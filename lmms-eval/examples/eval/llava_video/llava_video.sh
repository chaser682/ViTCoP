export HF_HOME="/home/chaser/.cache/huggingface"

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

LOG_DIR=./logs
MODEL_NAME="/home/chaser/model/LLaVA-NeXT-Video-7B"
RUN_NAME=llava_video_7b

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=${MODEL_NAME},video_decode_backend=decord,max_frames_num=32,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
    --tasks video_dc499 \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path $LOG_DIR/${RUN_NAME}