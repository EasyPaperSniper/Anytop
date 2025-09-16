"""
Small sanity test for Limb VQ-VAE.

Two modes:
- Synthetic: run the model on random tensors to verify shapes.
- Data: run a single batch from LimbDataset if cond/motions are available.

Run
- python -m limb_embedding.test_small_run --mode synthetic
- python -m limb_embedding.test_small_run --mode data --subset bipeds
"""
from __future__ import annotations

import argparse
import torch

from limb_embedding.models import LimbVQVAE


def synthetic() -> None:
    B, W, J, Din = 2, 64, 6, 6
    X = torch.randn(B, W, J, Din)
    joint_mask = torch.ones(B, J)
    parents = torch.tensor([-1] + list(range(0, J - 1)))
    bone_lengths = torch.ones(B, J, 1) * 0.35
    model = LimbVQVAE(d_joint_in=6, d_joint_out=6)
    out = model(X, joint_mask, parents, bone_lengths)
    print("z_e:", out["z_e"].shape, "z_q:", out["z_q"].shape, "recon:", out["recon"].shape)


def data(subset: str) -> None:
    from torch.utils.data import DataLoader
    from data_loaders.truebones.data.limb_emb_dataset import LimbDataset, LimbSampler, collate_fn
    ds = LimbDataset(split="train", num_frames=64, objects_subset=subset)
    dl = DataLoader(ds, batch_size=4, sampler=LimbSampler(ds), collate_fn=collate_fn)
    model = LimbVQVAE(d_joint_in=6, d_joint_out=6)
    for batch in dl:
        padded_limb_motion, joint_mask, local_parents, padded_bone_lengths, motion_length = batch
        parents = local_parents[0]
        out = model(padded_limb_motion, joint_mask, parents, padded_bone_lengths)
        print("batch recon:", out["recon"].shape)
        break


def main() -> None:
    ap = argparse.ArgumentParser(description="Small sanity test for Limb VQ-VAE")
    ap.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "data"])
    ap.add_argument("--subset", type=str, default="bipeds")
    args = ap.parse_args()
    if args.mode == "synthetic":
        synthetic()
    else:
        data(args.subset)


if __name__ == "__main__":
    main()

