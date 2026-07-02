#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-/root/venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-/publicdata/huggingface.co/Qwen/Qwen3-0.6B}"
PACKED_DIR="${PACKED_DIR:-/tmp/Qwen3-0.6B-static-mlp-gated-i16-calib-1x2048}"
TMP_TEST_DIR="${TMP_TEST_DIR:-/tmp/qwen3_int_only_tests}"
RUN_PPL="${RUN_PPL:-1}"

echo "[1/3] py_compile"
"${PYTHON}" -m py_compile \
  examples/qwen3_int_only/kernels.py \
  examples/qwen3_int_only/model.py \
  examples/qwen3_int_only/utils/__init__.py \
  examples/qwen3_int_only/utils/prepack.py \
  examples/qwen3_int_only/utils/ppl.py \
  examples/qwen3_int_only/utils/profile_kernels.py \
  examples/qwen3_int_only/utils/trace_block.py \
  examples/qwen3_int_only/utils/quarot.py

if [[ -f "${TMP_TEST_DIR}/test_static_main_path.py" ]]; then
  echo "[2/3] static kernel smoke"
  "${PYTHON}" -m pytest -q "${TMP_TEST_DIR}/test_static_main_path.py" -q
else
  echo "[2/3] static kernel smoke skipped: ${TMP_TEST_DIR}/test_static_main_path.py not found"
fi

if [[ "${RUN_PPL}" == "1" ]]; then
  if [[ ! -f "${PACKED_DIR}/qwen3_int_only.safetensors" ]]; then
    echo "[3/3] ppl skipped: ${PACKED_DIR}/qwen3_int_only.safetensors not found"
    echo "      create it with examples/qwen3_int_only/utils/prepack.py or set PACKED_DIR"
  else
    echo "[3/3] short ppl"
    "${PYTHON}" examples/qwen3_int_only/utils/ppl.py \
      --model-dir "${MODEL_DIR}" \
      --packed-dir "${PACKED_DIR}" \
      --eval-dataset fineweb \
      --max-tokens 257 \
      --batch-size 1 \
      --num-batches 1
  fi
else
  echo "[3/3] ppl skipped by RUN_PPL=${RUN_PPL}"
fi
