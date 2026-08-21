#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python prepare.py \
    --config_name "${CONFIG_NAME:-strat}" \
    --inputter_name "${INPUTTER_NAME:-strat}" \
    --train_input_file "${TRAIN_INPUT_FILE:-./DATA/6_train.txt}" \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --use_all_persona False \
    --encode_context True
