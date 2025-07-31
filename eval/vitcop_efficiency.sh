#!bin/bash
set -x
EXP_TABLE=coco2017_cap_val
LOG_DIR=./logs/vitcop
MODEL_NAME="/home/chaser/model/llava-v1.5-7b"
RUN_NAME=vitcop_llava_1.5_7b_efficiency

################################################
## VITCOP
## 计算剪枝率为 1/9 的token 下的效率分析
RATIO=("1_9")
PRUNED_RARIOS=(0.1111)
VISION_PRUNE_RARIOS=(0.3)
CLUSTER_PERCENTAGES=(0.12)
SHALLOW_PRUNED_LAYER=2
DEEP_PRUNED_LAYER=22  
################################################

for ((i=0; i<${#RATIO[@]}; i++)); do
    RATIO_VALUE=${RATIO[$i]}
    CUDA_VISIBLE_DEVICES=0 \
    USE_VITCOP=True \
    COMPUTE_EFFICIENCY=True \
    VITCOP_PRUNED_RARIO=${PRUNED_RARIOS[$i]} \
    VISION_PRUNE_RARIO=${VISION_PRUNE_RARIOS[$i]} \
    CLUSTER_PERCENTAGE=${CLUSTER_PERCENTAGES[$i]} \
    SHALLOW_PRUNED_LAYER=$SHALLOW_PRUNED_LAYER \
    DEEP_PRUNED_LAYER=$DEEP_PRUNED_LAYER \
    python3 -m accelerate.commands.launch \
        --num_processes=4 \
        -m lmms_eval \
        --model llava \
        --model_args pretrained=$MODEL_NAME \
        --tasks vqav2_val_lite \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix ${RUN_NAME}_${RATIO_VALUE} \
        --output_path $LOG_DIR/${RUN_NAME}_${RATIO_VALUE}
done