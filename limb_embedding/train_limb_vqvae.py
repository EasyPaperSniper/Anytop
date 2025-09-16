"""
Training script for Limb VQ-VAE.

This script trains a limb-wise motion tokenizer on Truebones limb windows.
It uses 6D joint rotations (features 3:9) as inputs and predicts the same.

Run
- python -m limb_embedding.train_limb_vqvae --subset bipeds --epochs 5 --batch_size 32

Notes
- Requires processed dataset under dataset/Truebones_processed and cond.npy.
- For balanced sampling across characters, uses LimbSampler.
"""
from __future__ import annotations

import argparse
import os
from os.path import join as pjoin
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from limb_embedding.models import LimbVQVAE, geodesic_loss_so3, rotation_6d_to_matrix
from data_loaders.truebones.data.limb_emb_dataset import LimbDataset, LimbSampler, collate_fn
from data_loaders.truebones.truebones_utils.get_opt import get_opt

# Optional Weights & Biases logging
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Limb VQ-VAE on Truebones limb windows")
    p.add_argument("--subset", type=str, default="bipeds", help="objects_subset key (see param_utils.OBJECT_SUBSETS_DICT)")
    p.add_argument("--num_frames", type=int, default=64, help="Temporal window length")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--codebook_size", type=int, default=1024)
    p.add_argument("--save_dir", type=str, default="limb_embedding/checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--vq_commit_w", type=float, default=0.25)
    p.add_argument("--vq_codebook_w", type=float, default=0.25)
    p.add_argument("--rot_w", type=float, default=1.0)
    # WandB
    p.add_argument("--wandb", dest="use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false", help="Disable Weights & Biases logging")
    p.set_defaults(use_wandb=False)
    p.add_argument("--wandb_project", type=str, default="limb-vqvae", help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (user or team)")
    p.add_argument("--run_name", type=str, default=None, help="W&B run name")
    return p.parse_args()


def build_dataloaders(subset: str, num_frames: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_set = LimbDataset(split="train", num_frames=num_frames, objects_subset=subset)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=LimbSampler(train_set),
                              num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    # For now, reuse train as val if no explicit split
    val_set = LimbDataset(split="test", num_frames=num_frames, objects_subset=subset)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=LimbSampler(val_set),
                            num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=False)
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # Dataloaders
    train_loader, val_loader = build_dataloaders(args.subset, args.num_frames, args.batch_size)

    # Model
    model = LimbVQVAE(d_joint_in=6, d_joint_out=6, codebook_size=args.codebook_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Setup W&B (optional)
    use_wandb = bool(args.use_wandb and _WANDB_AVAILABLE)
    if args.use_wandb and not _WANDB_AVAILABLE:
        print("[WARN] wandb requested but not installed; proceeding without logging.")
    if use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, config=vars(args))
        wandb.watch(model, log="gradients", log_freq=max(1, args.log_interval))
    # Codebook tracking
    codebook_size = getattr(model.vq, 'codebook_size', args.codebook_size)
    usage_counts_train = torch.zeros(codebook_size, dtype=torch.long)

    def _codebook_stats(counts: torch.Tensor):
        total = counts.sum().item()
        if total <= 0:
            return dict(utilization=0.0, perplexity=0.0, entropy=0.0)
        p = counts.float() / float(total)
        nz = p[p > 0]
        entropy = float(-(nz * torch.log(nz)).sum().item())
        perplexity = float(torch.exp(torch.tensor(entropy)).item())
        utilization = float((counts > 0).float().mean().item())
        return dict(utilization=utilization, perplexity=perplexity, entropy=entropy)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for it, (padded_limb_motion, joint_mask, local_parents, padded_bone_lengths, motion_length) in enumerate(train_loader, 1):
            X = padded_limb_motion.to(device)  # (B,W,J,6)
            mask = joint_mask.to(device)
            parents = local_parents[0].to(device)  # shared structure within batch
            bone_lengths = padded_bone_lengths.to(device)
            out = model(X, mask, parents, bone_lengths)

            # Losses
            recon = out["recon"][..., :6]
            R_hat = rotation_6d_to_matrix(recon)
            R_gt = rotation_6d_to_matrix(X)
            rot_loss = geodesic_loss_so3(R_hat, R_gt)
            vq = out["vq_losses"]
            vq_loss = args.vq_commit_w * vq["commit"] + args.vq_codebook_w * vq["codebook"]
            loss = args.rot_w * rot_loss + vq_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if it % args.log_interval == 0:
                with torch.no_grad():
                    # Codebook usage (approximate): unique in batch
                    uniq = out["indices"].unique().numel()
                msg = (f"[Epoch {epoch} | it {it}] loss={loss.item():.4f} rot={rot_loss.item():.4f} "
                       f"vq(c={vq['commit'].item():.4f},b={vq['codebook'].item():.4f}) cb_used={uniq}")
                print(msg)
                if use_wandb:
                    # Update running codebook counts and stats
                    batch_counts = torch.bincount(out["indices"].detach().cpu(), minlength=codebook_size)
                    usage_counts_train += batch_counts
                    cb_stats = _codebook_stats(usage_counts_train)
                    wandb.log({
                        "train/loss": float(loss.item()),
                        "train/rot_loss": float(rot_loss.item()),
                        "train/vq_commit": float(vq['commit'].item()),
                        "train/vq_codebook": float(vq['codebook'].item()),
                        "train/codebook_unique": int(uniq),
                        "train/codebook_utilization": cb_stats["utilization"],
                        "train/codebook_perplexity": cb_stats["perplexity"],
                        "train/codebook_entropy": cb_stats["entropy"],
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                    }, step=global_step)
            global_step += 1

        # Validation
        model.eval()
        rot_val_sum, n_batches = 0.0, 0
        with torch.no_grad():
            for padded_limb_motion, joint_mask, local_parents, padded_bone_lengths, motion_length in val_loader:
                X = padded_limb_motion.to(device)
                mask = joint_mask.to(device)
                parents = local_parents[0].to(device)
                bone_lengths = padded_bone_lengths.to(device)
                out = model(X, mask, parents, bone_lengths)
                recon = out["recon"][..., :6]
                rot_val_sum += geodesic_loss_so3(rotation_6d_to_matrix(recon), rotation_6d_to_matrix(X)).item()
                n_batches += 1
        rot_val = rot_val_sum / max(1, n_batches)
        print(f"[Epoch {epoch}] val geodesic(rot) = {rot_val:.4f}")
        if use_wandb:
            # Epoch-end codebook histogram and stats
            try:
                wandb.log({
                    "val/rot_geodesic": float(rot_val),
                    "epoch": epoch,
                    "train/codebook_hist": wandb.Histogram(usage_counts_train.numpy()),
                }, step=global_step)
            except Exception:
                wandb.log({"val/rot_geodesic": float(rot_val), "epoch": epoch}, step=global_step)
            # Reset counts for next epoch
            usage_counts_train.zero_()

        # Save
        ckpt_path = pjoin(args.save_dir, f"limb_vqvae_e{epoch}.pt")
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
        if use_wandb:
            try:
                art = wandb.Artifact("limb_vqvae_checkpoint", type="model")
                art.add_file(ckpt_path)
                wandb.log_artifact(art)
            except Exception as e:
                print(f"[WARN] Failed to log checkpoint artifact to wandb: {e}")


if __name__ == "__main__":
    main()
