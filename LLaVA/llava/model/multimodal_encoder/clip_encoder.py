import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import os
import numpy as np
import torch.nn.functional as F

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

        ## TODO
        ###################################################################################
        ## VITCOP
        self.use_vitcop = os.environ.get('USE_VITCOP', 'False') == 'True'
        self.vision_prune_ratio = float(os.environ.get('VISION_PRUNE_RARIO', 0.2))  # 初始剪枝比例
        self.cluster_percentage = float(os.environ.get('CLUSTER_PERCENTAGE', 0.08))  # 聚类中心点百分比数量
        ## VITCOP
        ###################################################################################

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    ## TODO
    ##########################################################################################
    ## VITCOP
    def compute_adaptive_threshold(self, attention_scores: torch.Tensor) -> float:
        """
        基于注意力分布计算自适应阈值
        
        Args:
            attention_scores: CLS token的注意力得分 [num_tokens]
            
        Returns:
            adaptive_prune_ratio: 自适应剪枝比例 (0.1, 0.3)
        """
        # 计算注意力分布的熵
        probs = F.softmax(attention_scores, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        
        # 计算方差
        variance = torch.var(attention_scores)
        
        # 根据熵和方差调整剪枝比例
        # 高熵(信息均匀分布)和低方差时减少剪枝
        # 低熵(信息集中)和高方差时增加剪枝
        entropy_norm = entropy / np.log(len(attention_scores))
        variance_norm = variance / torch.max(attention_scores)
        
        adaptive_ratio = self.initial_prune_ratio * (2 - entropy_norm) * (1 + variance_norm)
        adaptive_ratio = torch.clamp(adaptive_ratio, 0.1, 0.3)  # 限制在合理范围
        
        return adaptive_ratio.item()
    
    def cluster_visual_tokens(self, 
                             visual_tokens: torch.Tensor,
                             positions: torch.Tensor,
                             kept_indices: torch.Tensor,
                             cluster_method: str = "hierarchical") -> dict:
        """
        基于空间位置和特征相似度进行聚类
        
        Args:
            visual_tokens: 视觉token特征 [num_tokens, hidden_dim]
            positions: token的空间位置 [num_tokens, 2] (x, y)
            kept_indices: 保留的token索引
            cluster_method: 聚类方法 ("hierarchical", "kmeans", "dbscan")
            
        Returns:
            cluster_info: 包含聚类结果的字典
        """
        kept_tokens = visual_tokens[kept_indices]
        kept_positions = positions[kept_indices]
        
        # 标准化特征和位置
        tokens_norm = F.normalize(kept_tokens, p=2, dim=1)
        pos_norm = (kept_positions - kept_positions.mean(dim=0)) / (kept_positions.std(dim=0) + 1e-8)
        
        # 组合特征：拼接标准化后的视觉特征和位置特征
        combined_features = torch.cat([
            tokens_norm * self.feature_weight,
            pos_norm * self.spatial_weight
        ], dim=1)
        
        # 转换为numpy进行聚类
        features_np = combined_features.cpu().numpy()
        
        if cluster_method == "hierarchical":
            clusters = self._hierarchical_clustering(features_np)
        elif cluster_method == "kmeans":
            clusters = self._kmeans_clustering(features_np)
        elif cluster_method == "dbscan":
            clusters = self._dbscan_clustering(features_np)
        else:
            raise ValueError(f"Unsupported clustering method: {cluster_method}")
        
        # 组织聚类结果
        cluster_groups = {}
        for token_idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(token_idx)
        
        # 转换为列表格式，过滤噪声点（cluster_id = -1）
        cluster_list = [group for cluster_id, group in cluster_groups.items() if cluster_id != -1]
        
        # 计算聚类质量指标
        cluster_quality = self._evaluate_clustering_quality(features_np, clusters)
        
        return {
            'clusters': cluster_list,
            'cluster_labels': clusters,
            'kept_indices': kept_indices,
            'cluster_quality': cluster_quality,
            'num_clusters': len(cluster_list)
        }
    
    def get_pos(self, X, W = 24, H = 24):
        n_samples = X.shape[1]

        assert n_samples == W * H, f"Number of samples {n_samples} not equal grid size {W}*{H}={W*H}"

        yy, xx = torch.meshgrid(
            torch.arange(H, device=X.device),
            torch.arange(W, device=X.device),
            indexing='ij'
        )

        pos = torch.stack([yy, xx], dim=-1).reshape(-1, 2).float()
        # 归一化到[0,1)范围附近，原代码的归一化方式
        pos_norm = pos / torch.tensor([H, W], device=X.device, dtype=torch.float)

        return pos_norm.unsqueeze(0).expand(X.shape[0], -1, -1)

    def visual_tokens_spatial_similarity(self, images):
        """
        input:  images: [B, C, H, W]  # 输入图像
        output: hidden_states_filtered: [B, num_tokens, hidden_dim]
        output: clustered_labels: [B, num_clusters]
        """
        #set hooks for extracting desired layer's k
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[-2].self_attn.k_proj.register_forward_hook(hook_k)
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
        attn_weights  = image_forward_outs.attentions[-2]
        hidden_states = image_forward_outs.hidden_states[-2]

        #extract desired layer's k compute similarity
        desired_layer_k = outputs["desired_k"]
        # print(f"desired_layer_k shape: {desired_layer_k.shape}")
        hook_handle_k.remove()

        metric = desired_layer_k
        device = hidden_states.device

        ### 过滤cls tokens
        cls_attention = attn_weights[:, :, 0, 1:]  
        cls_attention_sum = cls_attention.sum(dim=1)
        hidden_states_filtered = hidden_states[:, 1:, :].view(
            hidden_states.shape[0], -1, hidden_states.shape[2]
        )
        metric_filtered = metric[:, 1:, :].view(hidden_states_filtered.shape[0], -1, metric.shape[2])
        # metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)
        vision_pos = self.get_pos(hidden_states_filtered, W=24, H=24)

        ## 保留 vision_retained_token 个 token
        vision_retained_token = int(self.vision_prune_ratio * hidden_states_filtered.shape[1])
        topk_indices = cls_attention_sum.topk(vision_retained_token, dim=1).indices
        mask = torch.ones_like(hidden_states_filtered[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, topk_indices, False)
        hidden_states_filtered = hidden_states_filtered.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], vision_retained_token, hidden_states.shape[2])
        metric_filtered = metric_filtered.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], vision_retained_token, metric.shape[2])
        vision_pos_filtered = vision_pos.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], vision_retained_token, vision_pos.shape[2])

        ## 使用聚类算法进行token的选择
        clustered_labels = []
        for i in range(hidden_states_filtered.shape[0]):

            sample_metric = metric_filtered[i]
            sample_pos = vision_pos_filtered[i]
            
            # 执行VIC聚类
            from .utils import VIC
            dc = 8 # 直径阈值
            percentage=self.cluster_percentage # 聚类中心点百分比数量
            k=0.6 # 空间约束系数
            vic = VIC(dc=dc, percentage=percentage, k=k)
            vic.fit(X=sample_metric, pos=sample_pos)
            labels = vic.labels_

            labels_ = torch.tensor(labels, device=device, dtype=torch.long)
            clustered_labels.append(labels_)
            # print(f"dc: {dc}, k: {k}, percentage: {percentage}, labels shape: {labels_.shape}, unique labels: {labels_.unique()}")

        return hidden_states_filtered, torch.stack(clustered_labels, dim=0)
    ## VITCOP
    ##########################################################################################

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        ## TODO
        ###################################################################################
        ## VITCOP
        elif self.use_vitcop:
            image_features, cluster_labels = self.visual_tokens_spatial_similarity(images)
            ## 处理多张图像的情况
            if image_features.shape[0] > 1:
                # 创建偏移量张量
                offsets = torch.arange(image_features.shape[0], device=cluster_labels.device) * image_features.shape[1]
                # 使用广播机制加上偏移量
                cluster_labels = (cluster_labels + offsets.unsqueeze(1)).flatten().unsqueeze(0)
                # print(f"cluster_labels shape: {cluster_labels.shape}")
        ## VITCOP
        ###################################################################################
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features, cluster_labels if self.use_vitcop else None

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
