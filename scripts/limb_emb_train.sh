#!/usr/bin/env bash
# Train limb-level VQ-VAE on Truebones limb windows.
#
# Usage (env defaults; extra args forwarded):
#   SUBSET=bipeds EPOCHS=5 BATCH_SIZE=32 NUM_FRAMES=64 LR=3e-4 CODEBOOK_SIZE=1024 \
#   WANDB=0 WANDB_PROJECT=limb-vqvae WANDB_ENTITY= \
#   bash scripts/limb_emb_train.sh --save_dir limb_embedding/checkpoints
#
# Example with Weights & Biases:
#   WANDB=1 WANDB_PROJECT=my-proj RUN_NAME=run-001 bash scripts/limb_emb_train.sh

set -euo pipefail

SUBSET=${SUBSET:-bipeds}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_FRAMES=${NUM_FRAMES:-64}
LR=${LR:-3e-4}
CODEBOOK_SIZE=${CODEBOOK_SIZE:-1024}
DEVICE=${DEVICE:-}

# WandB controls via env
WANDB_FLAG=${WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-limb-vqvae}
WANDB_ENTITY=${WANDB_ENTITY:-}
RUN_NAME=${RUN_NAME:-}

EXTRA_ARGS=()
if [[ "${WANDB_FLAG}" == "1" ]]; then
  EXTRA_ARGS+=("--wandb" "--wandb_project" "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    EXTRA_ARGS+=("--wandb_entity" "${WANDB_ENTITY}")
  fi
  if [[ -n "${RUN_NAME}" ]]; then
    EXTRA_ARGS+=("--run_name" "${RUN_NAME}")
  fi
else
  EXTRA_ARGS+=("--no-wandb")
fi

if [[ -n "${DEVICE}" ]]; then
  EXTRA_ARGS+=("--device" "${DEVICE}")
fi

python -m limb_embedding.train_limb_vqvae \
  --subset "${SUBSET}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_frames "${NUM_FRAMES}" \
  --lr "${LR}" \
  --codebook_size "${CODEBOOK_SIZE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

