Limb Embedding (VQ‑VAE)

Overview
- Goal: Learn a universal limb‑motion embedding by tokenizing chain motions (root→end‑effector) into discrete codes via a VQ‑VAE.
- Input: Per‑joint 6D rotations for a single limb window, with joint masking for variable‑length chains.
- Output: Reconstructed 6D rotations and a discrete token index per window.

Architecture (Encoder → VQ → Decoder)
- Encoder
  - Per‑joint temporal Conv1D (downsample in time)
  - Per‑frame joint self‑attention (joint index PE)
  - Frame descriptor via attention pooling
  - Temporal transformer block
  - Attention pooling over time → latent z_e
- VQ (EMA)
  - Quantizes z_e into z_q with straight‑through estimator
  - Tracks codebook usage via EMA
- Decoder
  - Temporal seed (repeat z_q over downsampled frames + sinusoidal PE)
  - Per‑frame joint self‑attention (broadcast to joints)
  - Per‑joint ConvTranspose1D (upsample to original length)
  - Linear head to D_out (e.g., 6D rotations)

ASCII Diagram
  [W,J,6] ──Conv1D(time)──▶ [W',J,d_t]
                 │
                 └─ Joint Self‑Attn ─▶ [W',J,d_s]
                                   │
                 Frame Attn‑Pool ─▶ [W',d_s] ── Temporal Transformer ─▶ [W',d_g]
                                   │
                         Time Attn‑Pool ─▶ z_e ── VQ(EMA) ─▶ z_q
                                                        │
                           Seed+PE ─▶ [W',d0] ─ Joint Self‑Attn ─▶ [W',J,d_t]
                                                        │
                                      ConvT1D(time) ─▶ [W,J,d_t] ─ Linear ─▶ [W,J,6]

Files
- models/limb_vqvae.py: Model implementation (encoder, quantizer, decoder).
- train_limb_vqvae.py: Training script over Truebones limb windows.
- eval_limb_vqvae.py: Evaluation script for reconstruction loss and codebook usage.
- test_small_run.py: Quick sanity test (synthetic or single batch from dataset).

Train
- python -m limb_embedding.train_limb_vqvae --subset bipeds --epochs 5 --batch_size 32

Eval
- python -m limb_embedding.eval_limb_vqvae --ckpt limb_embedding/checkpoints/limb_vqvae_e5.pt --subset bipeds

Small Test
- python -m limb_embedding.test_small_run --mode synthetic
- python -m limb_embedding.test_small_run --mode data --subset bipeds

Notes
- The dataloader extracts 6D rotations from features 3:9. Ensure preprocessing matches this layout.
- For efficiency, prefer GPU with pin_memory=True and num_workers>0.

