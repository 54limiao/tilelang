#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-/root/venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-/publicdata/huggingface.co/Qwen/Qwen3-0.6B}"
PACKED_DIR="${PACKED_DIR:-/tmp/Qwen3-0.6B-static-calib-32x2048}"
EVAL_DATASET="${EVAL_DATASET:-fineweb}"
EVAL_COLUMN="${EVAL_COLUMN:-text}"
EVAL_PARQUET="${EVAL_PARQUET:-}"
EVAL_TOKENS="${EVAL_TOKENS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_BATCHES="${NUM_BATCHES:-1}"
CACHE_PROMPT="${CACHE_PROMPT:-你是一个有用而无害的聊天助手。}"
CALIB_DATASET="${CALIB_DATASET:-fineweb}"
CALIB_COLUMN="${CALIB_COLUMN:-text}"
CALIB_PARQUET="${CALIB_PARQUET:-}"
CALIB_SEQ_LEN="${CALIB_SEQ_LEN:-2048}"
CALIB_BATCHES="${CALIB_BATCHES:-32}"
CALIB_PREFIX_TOKENS="${CALIB_PREFIX_TOKENS:-512}"
FORCE_PACK="${FORCE_PACK:-0}"
START_TS="$(date +%s)"

PREPACK_ARGS=(
  examples/qwen3_int_only/utils/prepack.py
  --model-dir "${MODEL_DIR}"
  --out-dir "${PACKED_DIR}"
  --use-r1
  --use-r2
  --use-r3
  --calib-dataset "${CALIB_DATASET}"
  --calib-column "${CALIB_COLUMN}"
  --calib-seq-len "${CALIB_SEQ_LEN}"
  --calib-batches "${CALIB_BATCHES}"
  --calib-prefix-tokens "${CALIB_PREFIX_TOKENS}"
)

if [[ -n "${CALIB_PARQUET}" ]]; then
  PREPACK_ARGS+=(--calib-parquet "${CALIB_PARQUET}")
fi

if [[ "${FORCE_PACK}" == "1" ]]; then
  PREPACK_ARGS+=(--force)
fi

PPL_ARGS=(
  examples/qwen3_int_only/utils/ppl.py
  --model-dir "${MODEL_DIR}"
  --packed-dir "${PACKED_DIR}"
  --backend int-only
  --compare-backend hf
  --eval-dataset "${EVAL_DATASET}"
  --eval-column "${EVAL_COLUMN}"
  --max-tokens "${EVAL_TOKENS}"
  --batch-size "${BATCH_SIZE}"
  --num-batches "${NUM_BATCHES}"
  --cache-prompt "${CACHE_PROMPT}"
  --use-r1
  --use-r2
)

if [[ -n "${EVAL_PARQUET}" ]]; then
  PPL_ARGS+=(--eval-parquet "${EVAL_PARQUET}")
fi

echo "[1/2] pack static int-only weights and calibration"
PACK_START_TS="$(date +%s)"
"${PYTHON}" "${PREPACK_ARGS[@]}"
PACK_END_TS="$(date +%s)"

echo "[2/2] evaluate int-only against float/HF: ppl cos mse, tokens=${EVAL_TOKENS}"
EVAL_START_TS="$(date +%s)"
"${PYTHON}" "${PPL_ARGS[@]}"
END_TS="$(date +%s)"

echo "time pack=$((PACK_END_TS - PACK_START_TS))s eval=$((END_TS - EVAL_START_TS))s total=$((END_TS - START_TS))s"
