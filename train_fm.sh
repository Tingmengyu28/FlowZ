#!/bin/bash
PROJECT_ROOT="/data1/azt/cv/recoverZ"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_MEMORY_ALLOCATION_MODE=legacy
export CUDA_VISIBLE_DEVICES=0,1,2,4

accelerate launch \
    --main_process_port=29501 \
    --num_processes=4 \
    --mixed_precision=fp16 \
    run/train_fm.py \
    --config configs/params.yaml