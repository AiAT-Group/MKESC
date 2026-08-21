#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train.py \
    --config_name "${CONFIG_NAME:-strat}" \
    --inputter_name "${INPUTTER_NAME:-strat}" \
    --eval_input_file "${EVAL_INPUT_FILE:-./DATA/6_valid.txt}" \
    --seed "${SEED:-13}" \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 16 \
    --learning_rate 1.5e-5 \
    --num_epochs 10 \
    --warmup_steps 100 \
    --fp16 false \
    --loss_scale 0.0 \
    --pbar true \
    --use_all_persona False \
    --encode_context True
