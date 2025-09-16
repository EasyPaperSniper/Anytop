#!/usr/bin/env bash
# Evaluate a trained Limb VQ-VAE and optionally export a reconstructed BVH window.
#
# Usage (env defaults; extra args forwarded):
#   CKPT=limb_embedding/checkpoints/limb_vqvae_e5.pt SUBSET=bipeds NUM_FRAMES=64 BATCH_SIZE=32 \
#   bash scripts/limb_emb_test.sh
#
# Export BVH window:
#   CKPT=... SUBSET=bipeds OBJECT_TYPE=Horse MOTION_NAME=Horse___Run_001.npy \
#   EXPORT_BVH=results/Horse_recon.bvh bash scripts/limb_emb_test.sh --window_start 0 --euler_order ZXY

set -euo pipefail

SUBSET=${SUBSET:-bipeds}
NUM_FRAMES=${NUM_FRAMES:-64}
BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE=${DEVICE:-}
CKPT=${CKPT:-}

if [[ -z "${CKPT}" ]]; then
  if ls limb_embedding/checkpoints/limb_vqvae_*.pt 1> /dev/null 2>&1; then
    CKPT=$(ls -t limb_embedding/checkpoints/limb_vqvae_*.pt | head -n1)
    echo "Using latest checkpoint: ${CKPT}"
  else
    echo "Error: CKPT not set and no checkpoints found under limb_embedding/checkpoints/" >&2
    exit 1
  fi
fi

EXTRA_ARGS=("--ckpt" "${CKPT}" "--subset" "${SUBSET}" "--num_frames" "${NUM_FRAMES}" "--batch_size" "${BATCH_SIZE}")

if [[ -n "${DEVICE}" ]]; then
  EXTRA_ARGS+=("--device" "${DEVICE}")
fi

if [[ -n "${EXPORT_BVH:-}" ]]; then
  : ${OBJECT_TYPE:?"OBJECT_TYPE must be set when EXPORT_BVH is provided"}
  : ${MOTION_NAME:?"MOTION_NAME must be set when EXPORT_BVH is provided"}
  EXTRA_ARGS+=("--export_bvh" "${EXPORT_BVH}" "--object_type" "${OBJECT_TYPE}" "--motion_name" "${MOTION_NAME}")
fi

python -m limb_embedding.eval_limb_vqvae "${EXTRA_ARGS[@]}" "$@"

