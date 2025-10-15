#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH=${1:-vocos_vocoder.onnx}
ENGINE_PATH=${2:-vocos_vocoder.plan}
PRECISION=${PRECISION:-fp32}   # fp32 | fp16 | bf16

# 1) find trtexec
if command -v trtexec >/dev/null 2>&1; then
  TRTEXEC=$(command -v trtexec)
elif [[ -x /opt/tensorrt/bin/trtexec ]]; then
  TRTEXEC=/opt/tensorrt/bin/trtexec
elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
else
  echo "[-] trtexec не найден. Установите TensorRT или добавьте trtexec в PATH." >&2
  exit 1
fi

echo "[i] trtexec: $TRTEXEC"
echo "[i] ONNX:    $ONNX_PATH"
echo "[i] ENGINE:  $ENGINE_PATH"
echo "[i] PREC:    $PRECISION"

# 2) precision flags
PREC_FLAGS=""
case "$PRECISION" in
  fp16) PREC_FLAGS="--fp16" ;;
  bf16) PREC_FLAGS="--bf16" ;;
  fp32) PREC_FLAGS="" ;;
  *) echo "[-] Неизвестная PRECISION=$PRECISION (ожидаю fp32|fp16|bf16)"; exit 2 ;;
esac

# 3) dynamic shapes for mel [B, 100, T]
MIN_BATCH_SIZE=1; OPT_BATCH_SIZE=1; MAX_BATCH_SIZE=8
MIN_INPUT_LENGTH=1; OPT_INPUT_LENGTH=1000; MAX_INPUT_LENGTH=3000
MEL_MIN_SHAPE="${MIN_BATCH_SIZE}x100x${MIN_INPUT_LENGTH}"
MEL_OPT_SHAPE="${OPT_BATCH_SIZE}x100x${OPT_INPUT_LENGTH}"
MEL_MAX_SHAPE="${MAX_BATCH_SIZE}x100x${MAX_INPUT_LENGTH}"

# 4) build
"$TRTEXEC" --onnx="$ONNX_PATH" --saveEngine="$ENGINE_PATH" \
  --minShapes="mel:${MEL_MIN_SHAPE}" --optShapes="mel:${MEL_OPT_SHAPE}" --maxShapes="mel:${MEL_MAX_SHAPE}" \
  $PREC_FLAGS

