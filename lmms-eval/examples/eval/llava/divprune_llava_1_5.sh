#!bin/bash

set -x

PAPER_TABLE=coco2017_cap_val,flickr30k_test,gqa,mmbench_en_dev,mme,mmmu_val,nocaps_val,ok_vqa_val2014,pope,scienceqa_img,seedbench,textvqa_val,mmvet,vqav2_val,vizwiz_vqa_val,qbench_dev

LITE_TABLE=coco2017_cap_val_lite,flickr30k_test_lite,gqa_lite,mmbench_en_dev_lite,ok_vqa_val2014_lite,refcoco_bbox_val_lite,seedbench_lite,textcaps_val_lite,textvqa_val_lite,vizwiz_vqa_val_lite,vqav2_val_lite

LOG_DIR=./logs/divprune
MODEL_NAME="/home/chaser/model/llava-v1.5-7b"
RUN_NAME=divprune_llava_1.5_7b

################################################
## DIVPRUNE
## 默认在CLIP-ViT倒数第二层之后进行剪枝
## 对应剪枝率 1/3，1/6，1/9, 1/12
RATIO=("1_3" "1_6" "1_9" "1_12")
RETAINED_TOKENS=("0.3333" "0.1667" "0.1111" "0.0833")
################################################

for ((i=0; i<${#RATIO[@]}; i++)); do
    RATIO_VALUE=${RATIO[$i]}
    RETAINED_TOKENS_VALUE=${RETAINED_TOKENS[$i]}
    CUDA_VISIBLE_DEVICES=0 \
    USE_DIVPRUNE=True \
    NOT_ADAPTIVE=${RETAINED_TOKENS_VALUE} \
    python3 -m accelerate.commands.launch \
        --num_processes=4 \
        -m lmms_eval \
        --model llava \
        --model_args pretrained=$MODEL_NAME \
        --tasks $LITE_TABLE \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix ${RUN_NAME}_${RATIO_VALUE} \
        --output_path $LOG_DIR/${RUN_NAME}_${RATIO_VALUE}
done