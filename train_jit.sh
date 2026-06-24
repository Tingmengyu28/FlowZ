#!/bin/bash
PROJECT_ROOT="/data1/azt/cv/recoverZ"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,6

accelerate launch \
    --main_process_port=29501 \
    --num_machines=1 \
    --num_processes=2 \
    --mixed_precision=fp16 \
    run/train_jit.py \
    --config configs/train.yaml
