# Chain-Motion-Based Embedding for Any-Topology Motion Generation

This repository extends the Anytop project with a new research idea: factorizing whole‑body motion into reusable limb (kinematic chain) tokens, and using a topology‑aware decoder to realize full‑body motion on arbitrary skeletons. The goal is a universal, data‑driven representation that transfers across characters with different joint counts and graph structures.


## Motivation
- Problem: Standard motion models often entangle motion dynamics with a specific skeleton topology, limiting transfer to new creatures or rigs.
- Idea: Decompose a character into kinematic chains (root→end‑effector paths). Learn a limb‑wise motion tokenizer that produces discrete codes from short temporal windows. A separate decoder consumes codes plus the target skeleton’s structure to generate full‑body motion.
- Benefits: Cross‑skeleton generalization, scalable training across species, modular reuse of motion patterns (walk, flap, crawl) regardless of embodiment.


## Method Overview
- Limb Tokenizer (VQ‑VAE): Encodes a window of per‑joint motion features for a single limb, quantizes to a codebook index, and reconstructs the limb’s motion.
  - Variable joints per limb supported via joint masks and local parent re‑indexing.
  - Discrete tokens via EMA Vector Quantization enable compact, composable motion primitives.
- Topology‑Aware Decoder (Anytop backbone): Conditions on the target skeleton (parents/graph, offsets, joint names/text embeddings) and decodes motion. In this project, the decoder is designed to be conditioned by limb tokens rather than raw full‑body features.
- Data: Truebones “Zoo” motions provide diverse topologies (bipeds, quadrupeds, flying animals, millipeds, snakes), enabling broad generalization.


## Repository Map (what matters for this idea)
- `limb_embedding/`
  - `temp_limb_VQVAE.py`: Prototype of the limb‑wise motion tokenizer.
    - Encoder: per‑joint temporal Conv1D → per‑frame joint self‑attention → frame pooling → temporal block → window latent.
    - Quantizer: EMA VQ (straight‑through estimator).
    - Decoder: temporal seeding → per‑frame joint attention → ConvTranspose1D upsampling → per‑joint heads.
    - Optional FK from 6D rotations for geometry‑aware losses/metrics.
  - `AGENTS.md`: Concept notes and shapes/losses cheat sheet.
- `data_loaders/truebones/`
  - `data/limb_emb_dataset.py`: Builds limb windows and masks from processed Truebones motions; handles limb chain selection, joint padding, and local parent mapping.
  - `data/dataset.py`: Full‑body dataset used by the Anytop backbone; includes joint‑name T5 embeddings and graph conditioning.
  - `truebones_utils/`: Preprocessing and configuration for Truebones (create dataset, motion processing, params).
- `model/`
  - `anytop.py`, `motion_transformer.py`: Graph‑aware transformer decoder for motion; serves as the topology‑aware realization module.
- `diffusion/`: Standard diffusion utilities from Anytop; kept for training a diffusion‑based decoder when conditioning on limb tokens.
- `train/`: Training loop and script for the Anytop backbone (full‑body). Will be extended to support token‑conditioned training.
- `visualization/`: Blender utilities to render `.bvh` or processed motions (helpful for qualitative checks of limb/full‑body generations).
- `eval/`: Metrics and benchmarks for Truebones.


## End‑to‑End Workflow (intended)
1. Prepare data
   - Download raw Truebones and run preprocessing to produce `dataset/Truebones_processed/{motions, animations, bvhs, cond.npy}`.
2. Train limb tokenizer (VQ‑VAE)
   - Use `LimbDataset`/`LimbMotionDataset` to sample kinematic chain windows and train `LimbVQVAE` with reconstruction+VQ+geometry losses.
   - Export per‑window token indices and (optionally) codebook embeddings for each limb across the dataset.
3. Train topology‑aware decoder
   - Condition the Anytop decoder on the target skeleton graph and on sequences of limb tokens (per limb and over time). Train with diffusion or transformer objectives to reconstruct full‑body motion.
4. Inference
   - Given a target skeleton and a sequence (or sampler) of limb tokens, decode full‑body motion respecting the new topology.
5. Visualization & Evaluation
   - Render with Blender utilities; evaluate with existing metrics (NN retrieval, distances) and add limb‑aware consistency checks.


## Differences vs Upstream Anytop
- Representation: Adds a limb‑wise discrete motion representation (VQ‑VAE) to decouple dynamics from topology.
- Data Loader: New limb window dataset with joint padding/masking and local parent mapping.
- Conditioning: Plan to condition the Anytop decoder on token sequences, rather than only raw full‑body features and joint‑name embeddings.


## Data and Environment
- Dataset: Truebones processed pack. Paths configurable via `data_loaders/truebones/truebones_utils/param_utils.py` and `get_opt.py`.
- Features: Processed motions use a per‑joint feature length of 13 (see `FEATS_LEN`), with 6D rotations commonly used by the limb tokenizer.
- Setup: See root `README.md` for environment creation and preprocessing commands. Note that in this repo the processed dataset path is `dataset/Truebones_processed` (slight naming differences from upstream docs).


## Planned Research Questions
- Token granularity: Best window length and stride for stable limb codes? Overlap strategies for smoothness.
- Composition: How to synchronize multiple limb token streams into coherent full‑body motion.
- Universality vs specialization: Single shared codebook vs subset‑specific codebooks (bipeds, flying, millipeds).
- Conditioning pathways: Cross‑attention from decoder to token streams vs learned token‑to‑frame alignment.
- Objectives: Mix of rotation, position (via FK), contact, and smoothness losses for robust decoding.


## Quick Pointers
- Limb tokenizer entry: `limb_embedding/temp_limb_VQVAE.py`
- Limb dataset: `data_loaders/truebones/data/limb_emb_dataset.py`
- Anytop decoder: `model/anytop.py`, `model/motion_transformer.py`
- Preprocess utilities: `utils/process_new_skeleton.py`, `data_loaders/truebones/truebones_utils/`
- Viz: `visualization/`
