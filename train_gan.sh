#!/bin/bash
PROJECT_ROOT="/data1/azt/cv/recoverZ"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch \
    --main_process_port=29500 \
    --num_processes=4 \
    --mixed_precision=fp16 \
    run/train_gan.py \
    --config configs/train.yaml
