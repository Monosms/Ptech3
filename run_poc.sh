#!/usr/bin/env bash
set -euo pipefail

IMG_PATH=${1:-assets/sample.jpg}
OUT_DIR=${2:-outputs}
CKPT=${3:-weights/sam_vit_b_01ec64.pth}

mkdir -p "$(dirname "$CKPT")"
mkdir -p assets "$OUT_DIR"

if [ ! -f "$IMG_PATH" ]; then
  echo "[INFO] 샘플 이미지를 찾을 수 없습니다: $IMG_PATH"
  echo "[INFO] 자신의 정면 사진을 assets/sample.jpg 로 저장하거나, 경로를 인자로 전달하세요."
fi

python3 hair_assess.py --image "$IMG_PATH" --output-dir "$OUT_DIR" --sam-checkpoint "$CKPT" --variant vit_b --device cpu --download
