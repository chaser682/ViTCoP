#!bin/bash

set -x

PAPER_TABLE=coco2017_cap_val,flickr30k_test,gqa,mmbench_en_dev,mme,mmmu_val,nocaps_val,ok_vqa_val2014,pope,scienceqa_img,seedbench,textvqa_val,mmvet,vqav2_val,vizwiz_vqa_val,qbench_dev

LITE_TABLE=coco2017_cap_val_lite,flickr30k_test_lite,gqa_lite,mmbench_en_dev_lite,ok_vqa_val2014_lite,refcoco_bbox_val_lite,seedbench_lite,textcaps_val_lite,textvqa_val_lite,vizwiz_vqa_val_lite,vqav2_val_lite

LOG_DIR=./logs
MODEL_NAME="/home/chaser/model/llava-v1.5-7b"
RUN_NAME=llava_1.5_7b


#To run other models use: liuhaotian/llava-v1.5-13b, liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.5-7b
CUDA_VISIBLE_DEVICES=0 python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained=$MODEL_NAME \
    --tasks ok_vqa_val2014_lite \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path $LOG_DIR/$RUN_NAME \
    # --verbosity DEBUG