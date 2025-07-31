#!bin/bash

set -x

PAPER_TABLE=coco2017_cap_val,flickr30k_test,gqa,mmbench_en_dev,mme,mmmu_val,nocaps_val,ok_vqa_val2014,pope,scienceqa_img,seedbench,textvqa_val,mmvet,vqav2_val,vizwiz_vqa_val,qbench_dev

################################################################
#FASTV 在数据集 seedbench_lite 存在问题, 由于FastV只能处理单张照片#
################################################################
LITE_TABLE=coco2017_cap_val_lite,flickr30k_test_lite,gqa_lite,mmbench_en_dev_lite,ok_vqa_val2014_lite,refcoco_bbox_val_lite,textcaps_val_lite,textvqa_val_lite,vizwiz_vqa_val_lite,vqav2_val_lite

LOG_DIR=./logs/fastv
MODEL_NAME="/home/chaser/model/llava-v1.5-7b"
RUN_NAME=fastv_llava_1.5_7b

################################################
## FASTV
## 默认在第二层之后进行剪枝
## 剪枝率计算 rato = (k + (rank / 576) * (32 - k) ) / 32
## 对应剪枝率 1/3，2/9，1/6, 1/9
RATIO=("1_3" "2_9" "1_6" "1_9")
RANKS=(166 98 64 30)
################################################

for ((i=0; i<${#RANKS[@]}; i++)); do
    RANK=${RANKS[$i]}
    RATIO_VALUE=${RATIO[$i]}
    CUDA_VISIBLE_DEVICES=0 \
    USE_FAST_V=True \
    FAST_V_AGG_LAYER=2 \
    FAST_V_ATTENTION_RANK=${RANK} \
    python3 -m accelerate.commands.launch \
        --num_processes=4 \
        -m lmms_eval \
        --model llava \
        --model_args pretrained=$MODEL_NAME \
        --tasks coco2017_cap_val_lite \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix ${RUN_NAME}_${RATIO_VALUE} \
        --output_path $LOG_DIR/${RUN_NAME}_${RATIO_VALUE}
done