#  LimbVQVAE: Motion Tokenizer for Any-Topology Characters

This directory implements a **Vector-Quantized Variational Autoencoder (VQ-VAE)** designed for motion sequences of arbitrary skeletons (different numbers of joints, morphologies, etc.).  
Instead of flattening the entire pose, we **decompose characters into limbs**, and learn **per-limb motion tokens**. These tokens can later be used for motion generation with diffusion models or other autoregressive architectures.

---

##  Key Ideas

1. **Encoder**
   - Takes raw motion features (e.g., per-joint 6D rotations).
   - Applies:
     - **Per-joint temporal Conv1D** → encodes each joint's trajectory, downsamples in time.
     - **Per-frame joint self-attention** → joints talk to each other at each frame (with joint-index positional encoding).
     - **Frame pooling** → compress joints → one descriptor per frame.
     - **Temporal block** → models dependencies across frames.
     - **Window pooling** → compress frames → one latent per window.
   - Outputs latent vector `z_e ∈ ℝ^(B × d_z)`.

2. **Vector Quantization**
   - EMA codebook maps `z_e` into discrete tokens.
   - Provides both quantized vector `z_q` and token index.
   - Loss = commitment + codebook.

3. **Decoder**
   - Mirrors the encoder but in reverse:
     - **Step D0**: Seed a temporal sequence from a single latent code (FFN + repeat + temporal PE).
     - **Step D1**: Broadcast seeds to joints and run per-frame joint self-attention.
     - **Step D2**: Per-joint ConvTranspose1D upsamples back to original frame length.
     - **Step D3**: Output heads predict joint-level features (e.g., 6D rotations).
   - Optional **FK (forward kinematics)** is applied to compute joint positions for loss/metrics.

---

## 📐 Shapes Cheat Sheet

| Stage                           | Shape                               |
|---------------------------------|-------------------------------------|
| Input motion                    | (B, W, J, D_joint)                  |
| After temporal Conv1D           | (B, W′, J, d_t)                     |
| After joint attention            | (B, W′, J, d_s)                     |
| After frame pooling              | (B, W′, d_s)                        |
| After temporal Transformer       | (B, W′, d_g)                        |
| After time pooling (latent)      | (B, d_z)                            |
| Quantized latent (VQ)            | (B, d_z), plus indices (B,)         |
| After temporal seed (decode)     | (B, W′, d0)                         |
| After joint attention (decode)   | (B, W′, J, d_t)                     |
| After ConvTranspose1D (decode)   | (B, W, J, d_up)                     |
| Final prediction                 | (B, W, J, D_joint_out)              |

---

## 🛠 Training Losses

- **Rotation loss**: geodesic distance on SO(3).
- **End-effector loss**: L2 distance after FK.
- **Contact loss**: BCE for foot/hand contact flags.
- **Velocity/locking loss**: penalize motion when contact=1.
- **Smoothness loss**: temporal differences.
- **VQ losses**: codebook + commitment.

