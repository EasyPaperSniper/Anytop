Limb Embedding Dataset: Usage and Tests

Overview

- Goal: Provide a chain‑centric limb dataset for learning universal motion embeddings across arbitrary skeletal topologies. Each sample isolates a single kinematic chain (limb) from a full‑body motion, with standardized temporal and joint padding to enable batching across diverse skeletons.
- Source: Preprocessed motion clips stored as `.npy` tensors coupled with a per‑character condition dictionary that defines kinematic chains and skeletal metadata.

File: limb_emb_dataset.py

- Purpose: Build per‑limb training samples from full‑body motion for a chain‑motion embedding model. Wraps a low‑level dataset (`LimbMotionDataset`) with a convenience dataset (`LimbDataset`) and a balancing sampler (`LimbSampler`).

- Key Components:
  - `collate_fn(batch)`: Sorts a batch before default collation. Note: It currently sorts by `x[1]` (the second tuple element). If you need to sort by motion length, change to `key=lambda x: x[-1]`.
  - `LimbMotionDataset(opt, cond_dict, max_limb_joints=20)`: Core dataset that indexes pairs of `(motion_clip, limb_key)` where `limb_key` identifies one kinematic chain for a given character/object type.
  - `LimbDataset(split="train", num_frames=64, **kwargs)`: Thin wrapper that constructs `opt` via `get_opt`, loads the condition dictionary, optionally subsets characters, and instantiates `LimbMotionDataset`.
  - `LimbSampler(data_source)`: A `WeightedRandomSampler` that approximately balances sampling probability across object types (characters) despite different numbers of limbs/motions per type.

- Data Assumptions and Inputs:
  - Motion clips: `.npy` files in `opt.motion_dir` named with prefix `{object_type}_...`. Each file loads to a float array `full_motion` of shape `[T, J, F]` where:
    - `T`: frames, `J`: joints, `F`: feature channels per joint.
    - The dataset extracts 6D joint rotations from `full_motion[:, :, 3:9]` (Zhou et al., CVPR 2019). Ensure upstream preprocessing places the 6D representation in indices `[3:9]`.
  - Condition dictionary: `cond_dict = np.load(opt.cond_file, allow_pickle=True).item()` maps each `object_type` to a dict with:
    - `kinematic_chains`: list[list[int]] — each inner list is a limb chain (root→end effector) as joint indices in the full skeleton.
    - `parents`: array/list[int] of length `J` — parent index per joint (−1 for root).
    - `offsets`: array[float] of shape `[J, 3]` — static bone offsets (rest‑pose vectors); used to derive per‑bone lengths.
    - Optional project‑specific fields may exist but are not used here.

- Output Per Sample (`__getitem__` of `LimbMotionDataset`):
  - `padded_limb_motion`: `torch.FloatTensor` of shape `[T_w, J_max, 6]` where:
    - `T_w = opt.max_motion_length` (after slicing/padding to window length).
    - `J_max = max_limb_joints` (padding to a fixed joint budget per limb).
    - Contains the 6D rotations for the selected limb chain in the leftmost joint slice; rightmost joints are zero‑padded.
  - `joint_mask`: `torch.FloatTensor` of shape `[J_max]` — `1.0` for valid limb joints, `0.0` for padded joints.
  - `local_parents`: `torch.LongTensor` of shape `[J_max]` — parent indices re‑indexed into the limb’s local joint order; `-1` marks the limb root; padded joints are `-1`.
  - `padded_bone_lengths`: `torch.FloatTensor` of shape `[J_max, 1]` — Euclidean norms of `offsets` for the limb, padded with zeros.
  - `motion_length`: `int` — original limb temporal length before slicing/padding.

- Temporal Logic:
  - If original limb length `< max_motion_length`: zero‑pad the temporal dimension to `max_motion_length`.
  - Else: uniformly select a random start index and take a contiguous window of length `max_motion_length`.

- Joint Padding and Masks:
  - Given a limb chain of length `L`, the leftmost `L` joints are valid; the remaining `max_limb_joints − L` joints are zero‑padded. `joint_mask[:L] = 1`, others `0`.

- Parent Re‑indexing:
  - Builds a local parent array for the limb so that edges only connect nodes inside the limb. Parents outside the chain map to `-1`, designating the limb root.

- Bone Lengths:
  - Computed as `||offsets[j]||_2` for each joint `j` in the limb and padded to `[J_max, 1]`.

- Wrapper Dataset (`LimbDataset`):
  - Calls `get_opt(None)` to obtain `opt`, makes `opt.motion_dir` and `opt.data_root` relative to project root, and clamps `opt.max_motion_length = min(opt.max_motion_length, num_frames)`.
  - Optional subsetting: pass `objects_subset=...` to restrict to a predefined subset of object types via `opt.subsets_dict[name]`.

- Sampler (`LimbSampler`):
  - Computes per‑sample weights so that each object type contributes an equal share of total sampling probability, independent of its number of clips/limbs.

- Quickstart Example:
  - Instantiate dataset and sampler
    - `from torch.utils.data import DataLoader`
    - `from data_loaders.truebones.data.limb_emb_dataset import LimbDataset, LimbSampler, collate_fn`
    - `dataset = LimbDataset(split="train", num_frames=64)`
    - `sampler = LimbSampler(dataset)`
  - Create loader
    - `loader = DataLoader(dataset, batch_size=32, sampler=sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)`
  - Iterate
    - `for padded_limb_motion, joint_mask, local_parents, padded_bone_lengths, motion_length in loader:`
    - `    ...  # feed into limb embedding model`

File: test_limb_emb_dataset.py

- Purpose: Sanity‑check the dataset outputs against expected shapes and metadata and validate that padded areas and parent re‑indexing are correct.

- What It Checks:
  - Shapes: `(64, max_limb_joints, 6)` for motion, `(max_limb_joints,)` for `joint_mask` and `local_parents`, `(max_limb_joints, 1)` for `padded_bone_lengths`.
  - Mask correctness: `sum(joint_mask) == len(limb_chain)`.
  - Padding correctness: the padded joint slice `[:, L:, :]` is all zeros.
  - Parent re‑indexing: exactly one root (`-1`) within the valid joints slice.
  - Data integrity (short sequences): for motions with `motion_length < 64`, verifies the returned 6D data matches the original `.npy` slice (no temporal cropping case).

- How To Run:
  - From project root: `python -m data_loaders.truebones.data.test_limb_emb_dataset`
  - Requirements: `opt.motion_dir` must contain preprocessed `.npy` motion files; `opt.cond_file` must exist and contain the condition dictionary described above.

- Notes and Caveats:
  - The test uses `random.randint(...)` but imports `random` only under the `if __name__ == '__main__'` guard. Running the function via other harnesses (e.g., pytest) may require adding `import random` at module scope.
  - For long sequences (`motion_length ≥ 64`), the test cannot deterministically match a random crop and will skip the integrity comparison for that case.

Integration Tips

- Embedding Targets: The dataset returns only rotations (6D) for the limb. If your model conditions on other features (e.g., root velocities, joint positions), extend the preprocessing and adjust the feature indices accordingly.
- Collation: If your downstream model expects batches sorted by temporal length (e.g., RNNs with packing), modify `collate_fn` to sort by `motion_length` via `key=lambda x: x[-1]` and keep the `motion_length` in the batch.
- Balancing: Use `LimbSampler` to balance across characters; otherwise common species/rigs may dominate batches.
- Efficiency: For large datasets, consider memory‑mapping motion `.npy` files or caching limb windows if I/O becomes a bottleneck. Increase `num_workers` in the DataLoader and use `pin_memory=True` when training on GPU.

Common Pitfalls

- Feature Layout Mismatch: Ensure your preprocessing pipeline writes 6D rotations into channels `[3:9]`. A mismatch will silently produce incorrect inputs.
- Inconsistent Joint Indexing: `kinematic_chains`, `parents`, and `offsets` must all refer to the same global joint indexing used in the `.npy` files.
- Limb Budget: Set `max_limb_joints` high enough to cover the largest chain in your dataset; otherwise valid joints will be truncated.
- File Naming: Motion files must begin with `{object_type}_` to be discovered. Characters missing motions in `opt.motion_dir` will be skipped.

References

- On the Continuity of Rotation Representations in Neural Networks — Zhou, Barnes, Lu, Yang, Li, CVPR 2019. Introduces 6D continuous rotation representation used here for stable learning.

