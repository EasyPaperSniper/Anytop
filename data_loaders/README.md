# Data Loaders

This folder contains dataset wrappers and utilities for preparing motion data used across the project. It focuses on the Truebones Zoo dataset that has been preprocessed into a uniform format for both full‑body and limb‑level training.


## Directory Structure
- `get_data.py` — Convenience utilities for fetching/handling data locally.
- `tensors.py` — Lightweight tensor helpers shared by loaders.
- `truebones/` — Primary data loader implementations for the Truebones dataset.
  - `data/` — Dataset classes specialized for different training regimes.
    - `dataset.py` — Full‑body motion dataset (`Truebones`, `MotionDataset`, `TruebonesSampler`).
    - `limb_emb_dataset.py` — Limb‑level dataset (`LimbDataset`, `LimbMotionDataset`, `LimbSampler`) for chain‑motion embedding (VQ‑VAE).
    - `test_limb_emb_dataset.py` — Sanity tests and utilities (includes a function to list all characters and their kinematic chains).
    - `instruction.md` — Notes on how the limb dataset is constructed and used.
  - `truebones_utils/` — Configuration and preprocessing parameters.
    - `param_utils.py` — Paths, constants, subsets, and processing thresholds.
    - `get_opt.py` — Centralized option builder that resolves dataset paths and defaults.


## Data Layout (Processed)
Processed data is expected under `dataset/Truebones_processed` (see `truebones_utils/param_utils.py`).

- `motions/` — NumPy `.npy` motion tensors. Naming convention: `{ObjectType}_... .npy`.
  - Each file: `motion` of shape `[T, J, F]`:
    - `T`: frames, `J`: joints, `F`: feature channels per joint.
    - By convention, 6D joint rotations are stored in `motion[:, :, 3:9]`. This is what the limb dataset consumes.
- `cond.npy` — Condition dictionary describing each character’s skeleton and metadata. For every object type key, it typically includes:
  - `parents`: list[int] of length `J` — parent index per joint (`−1` for root).
  - `offsets`: array `[J, 3]` — rest pose offsets for each joint.
  - `tpos_first_frame`: array `[J, F]` — canonical T‑pose/first frame features.
  - `joints_names`: list[str|None] of length `J` — human‑readable joint names (optional/padded).
  - `joint_relations`, `joints_graph_dist`: graph structure metadata.
  - `kinematic_chains`: list[list[int]] — root→end‑effector chains for limbs.
  - `mean`, `std`: per‑feature normalization statistics.
- `animations/`, `bvhs/` — Optional references to original BVH/animation assets.

Paths are resolved via `get_opt.py`, which reads constants from `param_utils.py`.


## Full‑Body Dataset (Truebones)
- Entry class: `Truebones` in `truebones/data/dataset.py`.
- Wraps `MotionDataset` and provides a `TruebonesSampler` for balanced sampling across object types.
- Handles:
  - Z‑scoring by per‑object `mean` and `std`.
  - Random temporal cropping/padding to a fixed window (`opt.max_motion_length`).
  - Optional joint augmentation (removal/addition) during training.
  - Optional T5 joint‑name embeddings for semantic conditioning (see `model/conditioners` and `T5Conditioner`).

Example
```python
from torch.utils.data import DataLoader
from data_loaders.truebones.data.dataset import Truebones, TruebonesSampler, collate_fn

# Full‑body windows (example values)
train_set = Truebones(split="train", temporal_window=31, t5_name="t5-base", num_frames=64, balanced=True, objects_subset="bipeds")
train_loader = DataLoader(train_set, batch_size=16, sampler=TruebonesSampler(train_set), collate_fn=collate_fn, num_workers=4, pin_memory=True)
for batch in train_loader:
    motion, m_length, parents, tpos_first_frame, offsets, temporal_mask, joints_graph_dist, joints_relations, object_type, joints_names_embs, ind, mean, std, max_joints = batch
    # ... train full‑body model
```


## Limb‑Level Dataset (Chain Embedding)
- Entry class: `LimbDataset` in `truebones/data/limb_emb_dataset.py`.
- Purpose: isolate a single kinematic chain (limb) per sample to learn a universal chain‑motion embedding.
- For each motion clip in `motions/`, creates samples for each limb defined in `cond['kinematic_chains']`.
- Returns, per sample:
  - `padded_limb_motion`: `[W, J_max, 6]` — 6D rotations for the limb, joint‑padded to `J_max`.
  - `joint_mask`: `[J_max]` — 1 for valid joints, 0 for padded.
  - `local_parents`: `[J_max]` — re‑indexed parents within the limb (`-1` marks chain root).
  - `padded_bone_lengths`: `[J_max, 1]` — per‑joint bone lengths (from `offsets`), padded.
  - `motion_length`: `int` — original (pre‑crop) limb motion length.
- Use `LimbSampler` for approximate balance across characters.

Example
```python
from torch.utils.data import DataLoader
from data_loaders.truebones.data.limb_emb_dataset import LimbDataset, LimbSampler, collate_fn

dataset = LimbDataset(split="train", num_frames=64, objects_subset="bipeds")
loader = DataLoader(dataset, batch_size=32, sampler=LimbSampler(dataset), collate_fn=collate_fn, num_workers=4, pin_memory=True)
for padded_limb_motion, joint_mask, local_parents, padded_bone_lengths, motion_length in loader:
    # ... feed into limb VQ‑VAE
```


## Subsets and Sampling
`param_utils.py` defines canonical object subsets under `OBJECT_SUBSETS_DICT` (e.g., `all`, `bipeds`, `quadropeds`, `flying`, `millipeds`, and their `_clean` variants). Pass `objects_subset` to restrict loaders to a subset.

Both full‑body and limb datasets include `WeightedRandomSampler`s that balance sampling across object types, mitigating dataset skew.


## Tests and Utilities
- Limb dataset test: `python -m data_loaders.truebones.data.test_limb_emb_dataset` validates shapes, masks, padding, and parent re‑indexing.
- Kinematic chain listing (for inspection):
```python
from data_loaders.truebones.data.test_limb_emb_dataset import list_characters_kinematic_chains
list_characters_kinematic_chains(objects_subset="all", verbose=True, save_json=True, json_path="data_loaders/truebones/data/kinematic_chain.json")
```
This also writes a human‑readable `.txt` companion in the same directory.


## Feature Conventions
- 6D rotations (Zhou et al., CVPR 2019) are stored in channels `[3:9]` of each joint’s feature vector. Ensure your preprocessing adheres to this layout.
- For limb‑level training, only rotations are consumed by default; extend the dataset and model if you require contact or other channels.


## Tips for Scale and Performance
- Increase `num_workers` and enable `pin_memory=True` when training on GPU.
- Consider memory‑mapping motion `.npy` files for large deployments.
- Ensure `max_limb_joints` in the limb dataset covers the largest chain in your subset to avoid truncation.


## References
- Y. Zhou, C. Barnes, J. Lu, J. Yang, H. Li. “On the Continuity of Rotation Representations in Neural Networks.” CVPR 2019 — 6D rotation representation used for stable learning.

