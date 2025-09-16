"""
Evaluation and export for Limb VQ-VAE.

Features
- Reconstruction metrics: geodesic rotation error + codebook usage.
- Full-character window reconstruction and BVH export for Blender visualization.

Run (metrics)
- python -m limb_embedding.eval_limb_vqvae --ckpt limb_embedding/checkpoints/limb_vqvae_e5.pt

Run (export BVH)
- python -m limb_embedding.eval_limb_vqvae --ckpt limb_embedding/checkpoints/limb_vqvae_e5.pt \
    --export_bvh out/recon.bvh --object_type Horse --motion_name Horse___Run_001.npy --num_frames 64
Then visualize with visualization/script.sh by pointing --bvh_path to the generated BVH.
"""
from __future__ import annotations

import argparse
import os
from os.path import join as pjoin
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from limb_embedding.models import LimbVQVAE, geodesic_loss_so3, rotation_6d_to_matrix
from data_loaders.truebones.data.limb_emb_dataset import LimbDataset, LimbSampler, collate_fn
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from utils.rotation_conversions import matrix_to_euler_angles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Limb VQ-VAE")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--subset", type=str, default="bipeds")
    p.add_argument("--num_frames", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Export BVH options
    p.add_argument("--export_bvh", type=str, default=None, help="If set, exports a reconstructed BVH to this path")
    p.add_argument("--object_type", type=str, default=None, help="Character name (must exist in cond.npy)")
    p.add_argument("--motion_name", type=str, default=None, help="Motion file name from motions dir (e.g., Horse___Run_001.npy)")
    p.add_argument("--window_start", type=int, default=0, help="Start frame for the reconstruction window")
    p.add_argument("--euler_order", type=str, default="ZXY", help="Euler order for BVH channels (e.g., ZXY, XYZ)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device)
    codebook_size = ckpt.get("args", {}).get("codebook_size", 1024)
    model = LimbVQVAE(d_joint_in=6, d_joint_out=6, codebook_size=codebook_size)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Data (for metrics)
    ds = LimbDataset(split="test", num_frames=args.num_frames, objects_subset=args.subset)
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=LimbSampler(ds),
                    num_workers=2, pin_memory=True, collate_fn=collate_fn)

    rot_sum, n, uniq_total = 0.0, 0, 0
    with torch.no_grad():
        for padded_limb_motion, joint_mask, local_parents, padded_bone_lengths, motion_length in dl:
            X = padded_limb_motion.to(device)
            mask = joint_mask.to(device)
            parents = local_parents[0].to(device)
            bone_lengths = padded_bone_lengths.to(device)
            out = model(X, mask, parents, bone_lengths)
            recon = out["recon"][..., :6]
            rot_sum += geodesic_loss_so3(rotation_6d_to_matrix(recon), rotation_6d_to_matrix(X)).item()
            uniq_total += out["indices"].unique().numel()
            n += 1
    print(f"Eval: geodesic(rot)={rot_sum/max(1,n):.4f}, codebook_unique_per_batch={uniq_total/max(1,n):.2f}")

    # Optional: export a reconstructed BVH for a single window
    if args.export_bvh is not None:
        export_reconstructed_bvh(model, args.object_type, args.motion_name, args.num_frames,
                                 args.window_start, args.euler_order, args.export_bvh, device=device)
        print(f"Saved reconstructed BVH to: {args.export_bvh}")


def export_reconstructed_bvh(model: LimbVQVAE,
                              object_type: str,
                              motion_name: str,
                              num_frames: int,
                              window_start: int,
                              euler_order: str,
                              out_path: str,
                              device: torch.device) -> None:
    """
    Reconstruct a full-character window by decoding all kinematic chains and export as BVH.

    - Uses ground-truth root positions from the original motion for realistic trajectory.
    - Merges overlapping joint predictions by averaging rotation matrices and projecting to SO(3).
    """
    assert object_type is not None and motion_name is not None, "object_type and motion_name must be provided for BVH export"

    # Load cond and motion
    opt = get_opt(None)
    cond_dict: Dict[str, Dict[str, Any]] = np.load(pjoin(opt.data_root, 'cond.npy'), allow_pickle=True).item()
    assert object_type in cond_dict, f"object_type '{object_type}' not found in cond.npy"
    cond = cond_dict[object_type]
    parents = np.array(cond['parents']).astype(int)
    offsets = np.array(cond['offsets']).astype(float)
    names = cond.get('joints_names', None)
    if isinstance(names, np.ndarray):
        names = names.tolist()
    if not names:
        names = [f"J{i}" for i in range(len(parents))]
    chains: List[List[int]] = cond['kinematic_chains']

    motion = np.load(pjoin(opt.motion_dir, motion_name))   # (T,J,F)
    T, J, F = motion.shape
    s = max(0, min(window_start, T - num_frames))
    e = min(T, s + num_frames)
    W = e - s
    assert W > 0, "Empty window after slicing"

    # Prepare accumulators for rotations
    R_accum = torch.zeros(W, J, 3, 3, device=device)
    R_count = torch.zeros(W, J, 1, device=device)

    # For each chain, run the model and accumulate rotation matrices
    with torch.no_grad():
        for chain in chains:
            chain = list(map(int, chain))
            X_chain = torch.from_numpy(motion[s:e, chain, 3:9]).float().unsqueeze(0).to(device)  # (1,W,Jc,6)
            mask = torch.ones(1, len(chain), device=device)
            # Local parents within the chain
            lp = []
            for j in chain:
                p = int(parents[j])
                if p in chain:
                    lp.append(chain.index(p))
                else:
                    lp.append(-1)
            local_parents = torch.tensor(lp, dtype=torch.long, device=device)
            bone_lengths = torch.from_numpy(np.linalg.norm(offsets[chain], axis=1)).float().view(1, len(chain), 1).to(device)

            out = model(X_chain, mask, local_parents, bone_lengths)
            recon6d = out['recon'][0, :, :, :6]  # (W,Jc,6)
            R = rotation_6d_to_matrix(recon6d)   # (W,Jc,3,3)
            # Accumulate into full arrays
            for ci, jidx in enumerate(chain):
                R_accum[:, jidx] += R[:, ci]
                R_count[:, jidx, 0] += 1.0

    # Normalize rotations: project averaged matrices to SO(3) via SVD
    R_avg = R_accum / torch.clamp(R_count, min=1.0)
    U, _, Vh = torch.linalg.svd(R_avg)
    R_proj = U @ Vh

    # Euler angles in degrees per joint, order matches BVH channel order
    euler = matrix_to_euler_angles(R_proj, euler_order)  # (W,J,3) radians
    euler_deg = torch.rad2deg(euler).cpu().numpy()

    # Use ground-truth root world positions from original motion
    root_pos = recover_root_positions_np(motion[s:e, 0, :])  # (W,3)

    # Write BVH
    write_bvh(out_path, parents.tolist(), offsets, names, euler_deg, root_pos, euler_order=euler_order)


def recover_root_positions_np(root_feats: np.ndarray) -> np.ndarray:
    """Recover root world positions from processed feature vector for the root joint.

    Expects root_feats shape (W, F) where rotations are at [:,3:9] and linear XZ velocities at [:,9] and [:,11],
    and root height at [:,1]. This follows the project’s processed feature convention.
    """
    from data_loaders.truebones.truebones_utils.motion_process import recover_root_quat_and_pos_np
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(root_feats)
    return r_pos


def write_bvh(path: str,
              parents: List[int],
              offsets: np.ndarray,
              names: List[str],
              euler_deg: np.ndarray,
              root_pos: np.ndarray,
              euler_order: str = 'ZXY',
              fps: int = 20) -> None:
    """
    Minimal BVH writer.

    Parameters
    - parents: list of length J with parent indices (-1 for root)
    - offsets: (J,3) rest offsets
    - names: list of joint names length J
    - euler_deg: (W,J,3) Euler angles in degrees, order given by euler_order
    - root_pos: (W,3) world positions for the root joint
    - euler_order: rotation order string (e.g., 'ZXY') used for CHANNELS
    - fps: frame rate
    """
    J = len(parents)
    children = {i: [] for i in range(J)}
    root = None
    for j, p in enumerate(parents):
        if p == -1:
            root = j
        else:
            children[p].append(j)
    assert root is not None, "No root joint found"

    def indent(n: int) -> str:
        return "\t" * n

    def write_joint(f, j: int, depth: int) -> None:
        jname = names[j] if j < len(names) else f"J{j}"
        if j == root:
            f.write("ROOT %s\n" % jname)
            f.write("{\n")
            f.write(f"{indent(depth+1)}OFFSET {offsets[j,0]:.6f} {offsets[j,1]:.6f} {offsets[j,2]:.6f}\n")
            f.write(f"{indent(depth+1)}CHANNELS 6 Xposition Yposition Zposition {euler_order[0]}rotation {euler_order[1]}rotation {euler_order[2]}rotation\n")
        else:
            f.write(f"{indent(depth)}JOINT {jname}\n")
            f.write(f"{indent(depth)}{{\n")
            f.write(f"{indent(depth+1)}OFFSET {offsets[j,0]:.6f} {offsets[j,1]:.6f} {offsets[j,2]:.6f}\n")
            f.write(f"{indent(depth+1)}CHANNELS 3 {euler_order[0]}rotation {euler_order[1]}rotation {euler_order[2]}rotation\n")

        for c in children[j]:
            write_joint(f, c, depth + 1)
        # End Site for leaves
        if len(children[j]) == 0:
            f.write(f"{indent(depth+1)}End Site\n")
            f.write(f"{indent(depth+1)}{{\n")
            f.write(f"{indent(depth+2)}OFFSET 0.000000 0.000000 0.000000\n")
            f.write(f"{indent(depth+1)}}}\n")
        f.write(f"{indent(depth)}}}\n")

    # Header
    with open(path, 'w', encoding='utf-8') as f:
        f.write("HIERARCHY\n")
        write_joint(f, root, 0)
        # Motion
        W = euler_deg.shape[0]
        f.write("MOTION\n")
        f.write(f"Frames: {W}\n")
        f.write(f"Frame Time: {1.0/float(fps):.8f}\n")
        # Channel order: For each joint in traversal order, with root having 6 channels
        # We need traversal order matching header; reuse a DFS
        order = []
        def dfs(j):
            order.append(j)
            for c in children[j]:
                dfs(c)
        dfs(root)
            for t in range(W):
                line_vals: List[str] = []
                for idx, j in enumerate(order):
                    if j == root:
                        px, py, pz = root_pos[t]
                    ex, ey, ez = euler_deg[t, j]
                        # Root: positions then rotations matching euler_order
                        line_vals += [f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}"]
                        # Map euler order to correct components
                    ed = { euler_order[0]: ex, euler_order[1]: ey, euler_order[2]: ez }
                        line_vals += [f"{ed[euler_order[0]]:.6f}", f"{ed[euler_order[1]]:.6f}", f"{ed[euler_order[2]]:.6f}"]
                    else:
                    ex, ey, ez = euler_deg[t, j]
                    ed = { euler_order[0]: ex, euler_order[1]: ey, euler_order[2]: ez }
                        line_vals += [f"{ed[euler_order[0]]:.6f}", f"{ed[euler_order[1]]:.6f}", f"{ed[euler_order[2]]:.6f}"]
                f.write(" ".join(line_vals) + "\n")


if __name__ == "__main__":
    main()
