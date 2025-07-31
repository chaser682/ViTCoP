import math
import re
import time
import torch
import torch.nn as nn
import random
import numpy as np
import os
from types import SimpleNamespace

def normalize(tensor):
    norm = torch.norm(tensor, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8)  # To avoid division by zero
    return tensor / norm


def normal_compute_pairwise_distances(X,l_2=False):
  
    X = torch.nn.functional.normalize(X,p=2.0,)
    
    dot_product = torch.mm(X, X.t())
 
    cosine_distance = 1 - dot_product
    
    return cosine_distance

class VIC:
    def __init__(self, dc=8, percentage=0.08, k=0.6):
        """
        Vision Information Clustering
        视觉信息聚类算法
        
        Args:
            dc (float): 局部密度计算的截止距离阈值
            percentage (float): 选择作为聚类中心的点的固定百分比
        """
        self.dc = dc # 局部密度计算的截止距离阈值
        self.percentage = percentage # 选择聚类中心的百分比
        self.k = k  # 空间距离阈值

    def get_clusters(self):
        """
        获取聚类结果
        
        Returns:
            dict: 每个簇标签对应的样本索引字典
        """
        if not hasattr(self, 'labels_') or self.labels_ is None:
            raise ValueError("请先运行聚类算法")
        
        # 返回每个簇及其包含的样本索引
        clusters = {label: np.where(self.labels_ == label)[0] 
                   for label in np.unique(self.labels_)}
        return clusters

    def fit(self, X, pos=None):
        """
        执行视觉信息聚类
        
        Args:
            X (torch.Tensor): 特征输入 [n_samples, n_features]
            pos (torch.Tensor): 坐标输入 [n_samples, 2]，归一化 (x, y) 坐标
        """
        n_samples = X.shape[0]    
        # 处理空输入情况
        if n_samples == 0:
            self.labels_ = np.array([], dtype=int)
            return
        device = X.device

        # === 计算距离矩阵 ===
        # 计算特征空间中的欧几里得距离
        dist_feat = self._compute_pairwise_distances(X)
        dist_feat = torch.clamp(dist_feat, min=0.0)
        dist = dist_feat
        dist.fill_diagonal_(0)  # 自身距离设为0

        # === DPC核心算法 ===
        # 1. 计算局部密度 rho
        # 使用高斯核函数计算每个点的局部密度
        rho = torch.exp(-(dist / self.dc) ** 2).sum(dim=1)

        # === 2. 计算 delta（到更高密度点的最小距离，带空间约束）===
        # 构造密度比较矩阵：rho_compare[i,j] = True 表示 j 的密度不高于 i
        rho_expand = rho.unsqueeze(1).expand(n_samples, n_samples)
        rho_compare = ~torch.gt(rho_expand, rho_expand.t())
        rho_compare.fill_diagonal_(False)  # 排除自身比较

        # 初始化空间距离矩阵
        if pos is not None:
            dist_spatial = self._compute_pairwise_distances(pos)
            dist_spatial = torch.clamp(dist_spatial, min=0.0)
            # 添加空间距离约束：仅考虑空间距离 <= k 的点
            spatial_mask = dist_spatial <= self.k
            # 综合密度和空间距离约束：密度更高且空间上接近
            valid_mask = rho_compare & spatial_mask
            # 用 inf 屏蔽掉不满足条件的距离
            inf_mask = torch.full_like(dist, float('inf'))
            conditioned_dist = torch.where(valid_mask, dist, inf_mask)
            # 计算 delta 和最相似的最近高密度点索引
            delta, nearest_higher_density_indices = torch.min(conditioned_dist, dim=1)
        else:
            # 不使用空间约束，使用原始特征距离 + 密度
            inf_mask = torch.full_like(dist, float('inf'))
            conditioned_dist = torch.where(rho_compare, dist, inf_mask)
            delta, nearest_higher_density_indices = torch.min(conditioned_dist, dim=1)

        # 3. 计算重要性指标 gamma = delta × rho
        # gamma值越大的点越可能是聚类中心
        gamma = delta * rho
        gamma_sorted_indices = torch.argsort(-gamma)  # 按gamma值降序排列

        # 4. 选择聚类中心
        # 根据设定的百分比选择gamma值最大的点作为聚类中心
        num_cluster_centers = max(1, int(self.percentage * n_samples))
        cluster_centers = gamma_sorted_indices[:num_cluster_centers]

        # 5. 标签传播
        # 初始化标签为 -1
        labels = torch.full((n_samples,), -1, dtype=torch.long, device=device)
        
        # 聚类中心索引，设为其标签值（即：中心点的下标）
        cluster_centers = cluster_centers.to(device)
        labels[cluster_centers] = torch.arange(len(cluster_centers), device=device)

        # 迭代传播标签，直到没有新的标签更新（并行方式）
        # 创建一个 mask 来标记是否还需要继续传播
        unassigned_mask = (labels == -1)
        updated = True

        while updated:
            # 找到尚未分配标签的点中，其对应的高密度邻居
            to_update = unassigned_mask & (labels[nearest_higher_density_indices] != -1)

            if not to_update.any():
                updated = False
                break

            # 把高密度邻居的标签赋值给当前未标记点
            src_labels = labels[nearest_higher_density_indices[to_update]]
            labels[to_update] = src_labels

            # 更新 mask
            unassigned_mask = (labels == -1)

        # 再次确保聚类中心标签没被更改
        labels[cluster_centers] = torch.arange(len(cluster_centers), device=device)

        # 将结果转换为numpy数组并保存
        self.labels_ = labels.cpu().numpy()

    def _compute_pairwise_distances(self, X):
        """
        计算特征空间中的成对欧几里得距离
        
        Args:
            X (torch.Tensor): 输入特征 [n_samples, n_features]
            
        Returns:
            torch.Tensor: 距离矩阵 [n_samples, n_samples]
        """
        # 使用广播计算所有点对之间的欧几里得距离
        X_expanded = X.unsqueeze(1)  # [n_samples, 1, n_features]
        X_transposed = X.unsqueeze(0)  # [1, n_samples, n_features]
        
        # 计算平方欧几里得距离
        dist_squared = torch.sum((X_expanded - X_transposed) ** 2, dim=2)
        
        # 返回欧几里得距离
        return torch.sqrt(dist_squared + 1e-8)  # 添加小常数避免数值不稳定

    def get_cluster_info(self):
        """
        获取聚类的详细信息
        
        Returns:
            dict: 包含聚类数量、噪声点数量等信息的字典
        """
        if not hasattr(self, 'labels_') or self.labels_ is None:
            raise ValueError("请先运行聚类算法")
        
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels)
        n_noise = np.sum(self.labels_ == -1) if -1 in unique_labels else 0
        
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[label] = np.sum(self.labels_ == label)
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': cluster_sizes,
            'total_samples': len(self.labels_)
        }