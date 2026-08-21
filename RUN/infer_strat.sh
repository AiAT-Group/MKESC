#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to a trained checkpoint before running inference}"

NLG_EVAL_ARGS=()
if [[ "${ADD_NLG_EVAL:-false}" == "true" ]]; then
    NLG_EVAL_ARGS+=(--add_nlg_eval)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python infer.py \
    --config_name "${CONFIG_NAME:-strat}" \
    --inputter_name "${INPUTTER_NAME:-strat}" \
    --seed 0 \
    --load_checkpoint "${CHECKPOINT}" \
    --fp16 false \
    --max_input_length 512 \
    --max_decoder_input_length 40 \
    --max_length 40 \
    --min_length 10 \
    --infer_batch_size 16 \
    --infer_input_file "${INPUT_FILE:-./DATA/6_test.txt}" \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1.0 \
    --no_repeat_ngram_size 3 \
    --use_all_persona False \
    --encode_context True \
    "${NLG_EVAL_ARGS[@]}"
