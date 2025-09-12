# Project Status

High-level: The repository contains a working baseline from Anytop and a functional prototype of a limb‑wise VQ‑VAE tokenizer. Data loaders support both full‑body and limb windows. The core missing piece is a training and integration pipeline that conditions the topology‑aware decoder on limb tokens, plus tooling to export and consume tokens at scale.


## What We Have
- Data and preprocessing
  - Truebones processing utilities (`data_loaders/truebones/truebones_utils/*`, `utils/process_new_skeleton.py`), and paths configured via `param_utils.py`/`get_opt.py`.
  - Processed dataset convention in this repo: `dataset/Truebones_processed/{motions, animations, bvhs, cond.npy}`.

- Limb tokenizer prototype
  - `limb_embedding/temp_limb_VQVAE.py`: Per‑limb VQ‑VAE with variable‑joint masking, EMA codebook, and FK utilities from 6D rotations.
  - `data_loaders/truebones/data/limb_emb_dataset.py`: Limb window dataset that pads joints, builds joint masks, and remaps local parents per chain.

- Anytop backbone (decoder)
  - `model/anytop.py`, `model/motion_transformer.py`: Graph‑aware motion decoder; `train/training_loop.py` and `train/train_anytop.py` for training.
  - `diffusion/*`: Diffusion utilities intact for training a generative decoder.

- Tooling
  - Evaluation metrics and benchmarks in `eval/` for Truebones.
  - Visualization utilities (`visualization/`) including Blender scripts.
  - Joint‑name T5 embeddings integrated in full‑body dataset for semantic conditioning.


## What’s Missing / To Implement
- LimbVQVAE training pipeline
  - Training script/CLI, config management, and logging (EMA VQ losses, rotation/FK/contact/smoothness losses, codebook utilization).
  - Explicit contact feature handling (limb dataset currently extracts 6D rotations from features 3:9). Optionally extend inputs/heads to include contact, root deltas, etc.

- Token export and storage
  - Offline tokenization pass over the dataset to save per‑limb, per‑window token indices (and optional code vectors) with timestamps and limb identifiers.
  - Define a durable format (e.g., `.npz` with {motion_id, limb_id, start, length, token_ids}).

- Token‑conditioned decoder
  - Integrate token sequences into the decoder: cross‑attention over limb tokens, or learned token‑to‑frame alignment modules.
  - Update `train/train_anytop.py` and collate logic to feed token streams (instead of or in addition to raw full‑body features).
  - Experiment with single vs multi‑stream conditioning (per limb) and fusion strategies.

- Inference/generation utilities
  - Sampling from token sequences (existing or generated) to produce full‑body motion on new skeletons.
  - Window stitching and smoothing strategies for temporal coherence across overlapping tokens.

- Evaluation extensions
  - Limb‑level reconstruction, token perplexity/usage, transition smoothness between windows.
  - Cross‑topology transfer evaluation: train on subset A, evaluate on held‑out skeletons/species.

- Reproducibility
  - Configs for subsets (bipeds/quadrupeds/etc.), codebook sizes, window lengths/strides, and training schedules.
  - Small scripts to regenerate key qualitative videos in `results/`.


## Risks / Open Questions
- Chain definitions and consistency: Quality of `kinematic_chains` in `cond.npy` across species; handling of tails/wings/antennae.
- Token granularity: Best window size/stride and overlap to balance editability and smoothness.
- Multi‑limb coordination: Ensuring coherent global pose when decoding many token streams; inter‑limb constraints.
- Objectives: Balancing rotation (SO(3)), FK position, contact, and smoothness for both tokenizer and decoder.
- Licensing: Truebones dataset redistribution restrictions (keep processing scripts, not data).


## Near‑Term Milestones (suggested)
1. LimbVQVAE baseline (bipeds subset)
   - Train with 6D rotations only, window=64, stride=32, codebook=1024. Export tokens. (1–2 weeks)
2. Token export (all subsets)
   - Build offline pipeline over processed motions; store `.npz` token files. (1 week)
3. Token‑conditioned decoder prototype
   - Cross‑attention from decoder to concatenated limb token streams. Train on bipeds; validate on held‑out skeletons. (2 weeks)
4. End‑to‑end demo
   - Given a target skeleton + tokenized sequence, render full‑body motion; add Blender videos. (2–3 weeks)
5. Ablations and metrics
   - Codebook size, window stride, overlap/stitching, per‑subset vs global codebook. (1–2 weeks)


## Decisions Needed
- Global vs subset‑specific codebooks.
- Window length/stride and overlap policy for deployment.
- Conditioning design: early fusion vs decoder cross‑attention vs alignment module.
- Loss mix and weighting for tokenizer/decoder.
