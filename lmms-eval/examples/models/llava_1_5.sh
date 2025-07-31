# # install lmms_eval without building dependencies
# cd lmms_eval;
# pip install --no-deps -U -e .

# # install LLaVA without building dependencies
# cd LLaVA
# pip install --no-deps -U -e .

# # install all the requirements that require for reproduce llava results
# pip install -r llava_repr_requirements.txt

# Run and exactly reproduce llava_v1.5 results!
# mme as an example
# accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="/home/um202477016/models/llava-v1.5-7b"   --tasks mme  --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/llava

#!bin/bash

set -x

PAPER_TABLE=coco2017_cap_val,flickr30k_test,gqa,mmbench_en_dev,mme,mmmu_val,nocaps_val,ok_vqa_val2014,pope,scienceqa_img,seedbench,textvqa_val,mmvet,vqav2_val,vizwiz_vqa_val,qbench_dev

LITE_TABLE=coco2017_cap_val_lite,flickr30k_test_lite,gqa_lite,mmbench_en_dev_lite,ok_vqa_val2014_lite,refcoco_bbox_val_lite,seedbench_lite,textcaps_val_lite,textvqa_val_lite,vqav2_val_lite

LOG_DIR=./logs
MODEL_NAME="/home/um202477016/models/llava-v1.5-7b"
RUN_NAME=llava_1.5_7b

#To run other models use: liuhaotian/llava-v1.5-13b, liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.5-7b
CUDA_VISIBLE_DEVICES=0 python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained=$MODEL_NAME \
    --tasks $LITE_TABLE \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path $LOG_DIR/$RUN_NAME \
    # --verbosity DEBUG