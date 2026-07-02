#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-/root/venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-/publicdata/huggingface.co/Qwen/Qwen3-0.6B}"
MODEL_NAME="$(basename "${MODEL_DIR}")"
PACKED_DIR="${PACKED_DIR:-/tmp/${MODEL_NAME}-static-calib-32x2048}"
EVAL_DATASET="${EVAL_DATASET:-fineweb}"
EVAL_COLUMN="${EVAL_COLUMN:-text}"
EVAL_PARQUET="${EVAL_PARQUET:-}"
BACKEND="${BACKEND:-int-only}"
EVAL_TOKENS="${EVAL_TOKENS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_BATCHES="${NUM_BATCHES:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"
PROFILE_REPEAT="${PROFILE_REPEAT:-16}"
PROFILE_WARMUP="${PROFILE_WARMUP:-1}"
PROFILE_LAYERS="${PROFILE_LAYERS:-0}"
CACHE_PROMPT="${CACHE_PROMPT:-你是一个有用而无害的聊天助手。}"
CALIB_DATASET="${CALIB_DATASET:-fineweb}"
CALIB_COLUMN="${CALIB_COLUMN:-text}"
CALIB_PARQUET="${CALIB_PARQUET:-}"
CALIB_SEQ_LEN="${CALIB_SEQ_LEN:-2048}"
CALIB_BATCHES="${CALIB_BATCHES:-32}"
CALIB_MICRO_BATCH="${CALIB_MICRO_BATCH:-1}"
CALIB_PREFIX_TOKENS="${CALIB_PREFIX_TOKENS:-512}"
FORCE_PACK="${FORCE_PACK:-0}"
USE_R1="${USE_R1:-1}"
USE_R2="${USE_R2:-1}"
LOG_DIR="${LOG_DIR:-/tmp/qwen3_int_only_logs}"
mkdir -p "${LOG_DIR}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${MODEL_NAME}-${BACKEND}"
PACK_LOG="${LOG_DIR}/${RUN_ID}.pack.log"
PPL_LOG="${LOG_DIR}/${RUN_ID}.ppl.log"
PROFILE_LOG="${LOG_DIR}/${RUN_ID}.profile.log"
START_TS="$(date +%s)"
TOTAL_STEPS=2
if [[ "${RUN_PROFILE}" == "1" ]]; then
  TOTAL_STEPS=3
fi

PREPACK_ARGS=(
  examples/qwen3_int_only/utils/prepack.py
  --model-dir "${MODEL_DIR}"
  --out-dir "${PACKED_DIR}"
  --calib-dataset "${CALIB_DATASET}"
  --calib-column "${CALIB_COLUMN}"
  --calib-seq-len "${CALIB_SEQ_LEN}"
  --calib-batches "${CALIB_BATCHES}"
  --calib-micro-batch "${CALIB_MICRO_BATCH}"
  --calib-prefix-tokens "${CALIB_PREFIX_TOKENS}"
  --cache-prompt "${CACHE_PROMPT}"
)

if [[ "${USE_R1}" == "1" ]]; then
  PREPACK_ARGS+=(--use-r1)
fi

if [[ "${USE_R2}" == "1" ]]; then
  PREPACK_ARGS+=(--use-r2)
fi

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
  --backend "${BACKEND}"
  --compare-backend hf
  --eval-dataset "${EVAL_DATASET}"
  --eval-column "${EVAL_COLUMN}"
  --max-tokens "${EVAL_TOKENS}"
  --batch-size "${BATCH_SIZE}"
  --num-batches "${NUM_BATCHES}"
  --cache-prompt "${CACHE_PROMPT}"
)

if [[ "${USE_R1}" == "1" ]]; then
  PPL_ARGS+=(--use-r1)
fi

if [[ "${USE_R2}" == "1" ]]; then
  PPL_ARGS+=(--use-r2)
fi

if [[ -n "${EVAL_PARQUET}" ]]; then
  PPL_ARGS+=(--eval-parquet "${EVAL_PARQUET}")
fi

echo "[1/${TOTAL_STEPS}] pack static int-only weights and calibration"
PACK_START_TS="$(date +%s)"
"${PYTHON}" "${PREPACK_ARGS[@]}" 2>&1 | tee "${PACK_LOG}"
PACK_END_TS="$(date +%s)"

echo "[2/${TOTAL_STEPS}] evaluate ${BACKEND} against float/HF: ppl cos mse, tokens=${EVAL_TOKENS}"
EVAL_START_TS="$(date +%s)"
"${PYTHON}" "${PPL_ARGS[@]}" 2>&1 | tee "${PPL_LOG}"
EVAL_END_TS="$(date +%s)"

PROFILE_END_TS="${EVAL_END_TS}"
if [[ "${RUN_PROFILE}" == "1" ]]; then
  echo "[3/${TOTAL_STEPS}] profile ${BACKEND} kernels: tokens=${EVAL_TOKENS}, repeat=${PROFILE_REPEAT}"
  PROFILE_START_TS="$(date +%s)"
  PROFILE_ARGS=(
    examples/qwen3_int_only/utils/profile_kernels.py
    --model-dir "${MODEL_DIR}"
    --packed-dir "${PACKED_DIR}"
    --backend "${BACKEND}"
    --max-tokens "${EVAL_TOKENS}"
    --layers "${PROFILE_LAYERS}"
    --warmup "${PROFILE_WARMUP}"
    --repeat "${PROFILE_REPEAT}"
    --cache-prompt "${CACHE_PROMPT}"
  )
  "${PYTHON}" "${PROFILE_ARGS[@]}" 2>&1 | tee "${PROFILE_LOG}"
  PROFILE_END_TS="$(date +%s)"
fi

echo ""
echo "== summary =="
grep '^backend=' "${PPL_LOG}" || true
if [[ "${RUN_PROFILE}" == "1" ]]; then
  grep '^profile seq_len=' "${PROFILE_LOG}" || true
fi
echo "logs pack=${PACK_LOG}"
echo "logs ppl=${PPL_LOG}"
if [[ "${RUN_PROFILE}" == "1" ]]; then
  echo "logs profile=${PROFILE_LOG}"
fi
echo "time pack=$((PACK_END_TS - PACK_START_TS))s eval=$((EVAL_END_TS - EVAL_START_TS))s profile=$((PROFILE_END_TS - EVAL_END_TS))s total=$((PROFILE_END_TS - START_TS))s"
