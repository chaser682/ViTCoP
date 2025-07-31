#!bin/bash

set -x

PAPER_TABLE=coco2017_cap_val,flickr30k_test,gqa,mmbench_en_dev,mme,mmmu_val,nocaps_val,ok_vqa_val2014,pope,scienceqa_img,seedbench,textvqa_val,mmvet,vqav2_val,vizwiz_vqa_val,qbench_dev

EXP_TABLE=gqa,mmbench_en_dev,mme,mmmu_val,ok_vqa_val2014,pope,scienceqa_img,seedbench_lite,textvqa_val,vqav2_val,vizwiz_vqa_val,qbench_dev

LITE_TABLE=coco2017_cap_val_lite,flickr30k_test_lite,gqa_lite,mmbench_en_dev_lite,ok_vqa_val2014_lite,refcoco_bbox_val_lite,seedbench_lite,textcaps_val_lite,textvqa_val_lite,vizwiz_vqa_val_lite,vqav2_val_lite

LOG_DIR=./logs/dssp
MODEL_NAME="/home/chaser/model/llava-v1.5-7b"
RUN_NAME=dssp_llava_1.5_7b

################################################
## DSSP
RATIO=("9_10" "8_10" "7_10" "6_10" "5_10" "4_10" "3_10" "2_10" "1_10")
RETAINED_TOKENS=(518 461 403 346 288 230 173 115 57)
RATIO=("10_10")
RETAINED_TOKENS=(576)
################################################

for ((i=0; i<${#RATIO[@]}; i++)); do
    RATIO_VALUE=${RATIO[$i]}
    CUDA_VISIBLE_DEVICES=0 \
    USE_DSSP=True \
    RETAINED_TOKEN=${RETAINED_TOKENS[$i]} \
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