# -*- coding: utf-8 -*-
# Limb-wise VQ-VAE for motion windows
# Encoder: per-joint temporal Conv1D (downsample) + per-frame joint self-attn → frame descriptors
#          tiny temporal block → attention pool over time → latent z
# Quantizer: EMA VQ
# Decoder: temporal seeding (repeat) → per-frame joint self-attn → per-joint ConvTranspose1D (upsample)
#          per-joint heads (e.g., 6D rotations [+ optional extras])
#
# Only motion inputs are used (e.g., local 6D rotations; optionally contact, root offset, etc.).
# Variable joint counts are supported with a joint mask. No relation/type/meta embeddings are used.

from typing import Optional, Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Utilities
# ----------------------------

def conv1d_out_length(L_in: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    return math.floor((L_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1)

def convt1d_out_length(L_in: int, kernel: int, stride: int, padding: int,
                       output_padding: int = 0, dilation: int = 1) -> int:
    return (L_in - 1) * stride - 2*padding + dilation*(kernel - 1) + output_padding + 1

def make_norm_joint_index(B: int, J: int, joint_mask: Optional[torch.Tensor] = None, device=None):
    """
    Per-sample normalized joint index in [0,1]: idx / (J_valid-1), masking pads.
    joint_mask: (B, J) with 1=valid, 0=pad. If None, assume all valid.
    Returns: (B, J, 1) float
    """
    device = device or (joint_mask.device if joint_mask is not None else "cpu")
    base = torch.arange(J, device=device).float().unsqueeze(0).expand(B, J)  # (B,J)
    if joint_mask is None:
        denom = (J - 1) if J > 1 else 1.0
        return (base / denom).unsqueeze(-1)
    valid_counts = joint_mask.sum(dim=1).clamp(min=1).float()   # (B,)
    denom = (valid_counts - 1).clamp(min=1.0).unsqueeze(1)      # (B,1)
    out = (base / denom) * joint_mask                           # (B,J)
    return out.unsqueeze(-1)                                     # (B,J,1)

def sinusoid_posenc(T: int, D: int, device=None) -> torch.Tensor:
    """
    Standard sinusoidal positional encoding (Vaswani et al. 2017) for length T and dim D.
    Returns: (1, T, D)
    """
    device = device or "cpu"
    pe = torch.zeros(T, D, device=device)
    position = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, T, D)

class AttentionPooling(nn.Module):
    """
    Attention pooling along a sequence dimension with optional mask.
    Given X: (B, T, D), returns pooled: (B, D).
    mask: (B, T) with 1=valid, 0=pad.
    """
    def __init__(self, dim: int, with_proj: bool = False):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))  # (D,)
        self.proj = nn.Linear(dim, dim) if with_proj else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B,T,D)
        B, T, D = x.shape
        q = self.query.view(1, 1, D).expand(B, T, D)    # (B,T,D)
        logits = (self.proj(x) * q).sum(dim=-1)         # (B,T)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(logits, dim=1)             # (B,T)
        pooled = (attn.unsqueeze(-1) * x).sum(dim=1)    # (B,D)
        return pooled

# 6D rotation utilities (Zhou et al., CVPR 2019)
def rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    x: (..., 6) -> rotation matrix (..., 3, 3)
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)  # (..., 3, 3)

def geodesic_loss_so3(R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    R_pred, R_gt: (..., 3, 3)
    Returns mean geodesic angle (radians).
    """
    M = torch.matmul(R_pred.transpose(-1, -2), R_gt)  # (...,3,3)
    cos = ((M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2]) - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos)
    return theta.mean()

# (Simplified) FK placeholder: replace by your rig’s FK for training targets/metrics
def fk_local_rotations_to_positions(R_local: torch.Tensor,
                                    parents: torch.Tensor,
                                    bone_lengths: torch.Tensor) -> torch.Tensor:
    """
    R_local: (B, W, J, 3, 3) local rotations
    parents: (J,) parent index per joint (-1 for limb root)
    bone_lengths: (B, J, 1) per-joint length; assumes rest direction along local z.
    Returns positions: (B, W, J, 3)
    """
    B, W, J = R_local.shape[:3]
    device = R_local.device
    pos = torch.zeros(B, W, J, 3, device=device)
    z = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 1, 3, 1)
    seg = torch.matmul(R_local, z).squeeze(-1)  # (B,W,J,3)
    for j in range(J):
        p = int(parents[j].item())
        if p == -1:
            pos[:, :, j, :] = 0.0
        else:
            pos[:, :, j, :] = pos[:, :, p, :] + seg[:, :, j, :] * bone_lengths[:, j, :]
    return pos

# ----------------------------
# Encoder
# ----------------------------

class PerJointTemporalConvDown(nn.Module):
    """
    Per-joint Conv1D along time with learned downsampling (stride on first layer).
    Input:  X  (B, W, J, D_joint)
    Output: Ht (B, W', J, d_t)
    """
    def __init__(self, d_joint: int, d_t: int, kernel_size=5, stride=2, padding=2,
                 num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = d_joint
        for i in range(num_layers):
            out_ch = d_t
            layers += [
                nn.Conv1d(in_ch, out_ch,
                          kernel_size=kernel_size,
                          stride=(stride if i == 0 else 1),
                          padding=padding),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, W, J, D = X.shape
        Xr = X.permute(0, 2, 3, 1).contiguous().view(B * J, D, W)  # (B*J, D, W)
        H = self.conv(Xr)                                          # (B*J, d_t, W')
        _, d_t, Wp = H.shape
        H = H.view(B, J, d_t, Wp).permute(0, 3, 1, 2)              # (B, W', J, d_t)
        return H

    def out_length(self, W: int) -> int:
        return conv1d_out_length(W, self.kernel_size, self.stride, self.padding)

class PerFrameJointSelfAttention(nn.Module):
    """
    Per-frame joint self-attention with joint-index PE (index / max_index).
    Input:  Ht (B, W', J, d_t), joint_mask (B, J)
    Output: Hj (B, W', J, d_s)
    """
    def __init__(self, d_t: int, d_s: int, n_heads=4, dropout=0.1):
        super().__init__()
        self.pe_proj = nn.Linear(1, d_t)  # scalar -> d_t
        self.attn = nn.MultiheadAttention(d_t, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(d_t),
                                 nn.Linear(d_t, d_s), nn.GELU(), nn.LayerNorm(d_s))

    def forward(self, Ht: torch.Tensor, joint_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, Wp, J, d_t = Ht.shape
        if joint_mask is None:
            joint_mask = Ht.new_ones(B, J)
        pe = self.pe_proj(make_norm_joint_index(B, J, joint_mask, device=Ht.device))  # (B,J,1)->(B,J,d_t)
        Ht_pe = Ht + pe.unsqueeze(1)                                                  # (B,1,J,d_t)
        X = Ht_pe.reshape(B * Wp, J, d_t)                                             # (B*W',J,d_t)
        kpm = (joint_mask == 0).unsqueeze(1).expand(B, Wp, J).reshape(B * Wp, J)      # (B*W',J)
        Y, _ = self.attn(X, X, X, key_padding_mask=kpm)                                # (B*W',J,d_t)
        Y = Y.view(B, Wp, J, d_t)
        Hj = self.ffn(Y)                                                              # (B,W',J,d_s)
        return Hj

class FrameDescriptor(nn.Module):
    """Attention pooling over joints → per-frame descriptor."""
    def __init__(self, d_s: int, with_proj: bool = False):
        super().__init__()
        self.pool = AttentionPooling(d_s, with_proj=with_proj)

    def forward(self, Hj: torch.Tensor, joint_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, Wp, J, d = Hj.shape
        if joint_mask is None:
            joint_mask = Hj.new_ones(B, J)
        mask = joint_mask.unsqueeze(1).expand(B, Wp, J)     # (B,W',J)
        Hj_  = Hj.reshape(B * Wp, J, d)
        mask_ = mask.reshape(B * Wp, J)
        F = self.pool(Hj_, mask_).reshape(B, Wp, d)         # (B,W',d)
        return F

class TemporalBlock(nn.Module):
    """Tiny temporal Transformer over downsampled frames (keeps length)."""
    def __init__(self, d_in: int, d_g: int, n_heads=4, n_layers=1, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_g) if d_in != d_g else nn.Identity()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_g, nhead=n_heads,
                                               dim_feedforward=4*d_g, batch_first=True,
                                               dropout=dropout, activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        G = self.in_proj(F)     # (B,W',d_g)
        return self.enc(G)      # (B,W',d_g)

class WindowLatent(nn.Module):
    """Attention pooling over time → window latent z."""
    def __init__(self, d_g: int, d_z: int, dropout=0.1):
        super().__init__()
        self.pool = AttentionPooling(d_g)
        self.proj = nn.Sequential(nn.LayerNorm(d_g), nn.Linear(d_g, d_z), nn.Dropout(dropout))

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        z_raw = self.pool(G)     # (B,d_g)
        z = self.proj(z_raw)     # (B,d_z)
        return z

class LimbWindowEncoder(nn.Module):
    """
    Encoder: (B,W,J,D_joint) -> (B,d_z)
    Also returns intermediates for inspection.
    """
    def __init__(self, d_joint: int, d_t=256, d_s=256, d_g=256, d_z=256,
                 kernel_size=5, stride=2, padding=2, num_temporal_layers=2,
                 n_heads_joint=4, n_heads_time=4, n_time_layers=1, dropout=0.1):
        super().__init__()
        self.temporal = PerJointTemporalConvDown(d_joint, d_t, kernel_size, stride, padding,
                                                 num_temporal_layers, dropout)
        self.joint_attn = PerFrameJointSelfAttention(d_t, d_s, n_heads_joint, dropout)
        self.frame_desc = FrameDescriptor(d_s, with_proj=False)
        self.temp_block = TemporalBlock(d_s, d_g, n_heads_time, n_time_layers, dropout)
        self.to_latent  = WindowLatent(d_g, d_z, dropout)

    def forward(self, X: torch.Tensor, joint_mask: Optional[torch.Tensor] = None):
        Ht = self.temporal(X)                     # (B,W',J,d_t)
        Hj = self.joint_attn(Ht, joint_mask)     # (B,W',J,d_s)
        F  = self.frame_desc(Hj, joint_mask)     # (B,W',d_s)
        G  = self.temp_block(F)                  # (B,W',d_g)
        z  = self.to_latent(G)                   # (B,d_z)
        return z, {"Ht": Ht, "Hj": Hj, "F": F, "G": G}

    def out_length(self, W: int) -> int:
        return self.temporal.out_length(W)

# ----------------------------
# EMA VQ quantizer
# ----------------------------

class VQEMA(nn.Module):
    """
    EMA Vector Quantization with straight-through estimator.
    """
    def __init__(self, codebook_size: int, d_z: int, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_z = d_z
        self.decay = decay
        self.eps = eps

        self.embed = nn.Embedding(codebook_size, d_z)
        nn.init.uniform_(self.embed.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", self.embed.weight.data.clone())

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, num_classes=self.codebook_size).type_as(z_e)  # (B,K)
        cluster_size = one_hot.sum(dim=0)                                         # (K,)
        embed_sum = one_hot.t() @ z_e                                             # (K,d_z)

        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum,    alpha=1 - self.decay)

        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embed.weight.data.copy_(embed_normalized)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        z_e: (B,d_z) encoder output
        Returns: z_q (B,d_z), indices (B,), losses
        """
        d = (z_e.pow(2).sum(dim=1, keepdim=True)
             - 2 * z_e @ self.embed.weight.t()
             + self.embed.weight.pow(2).sum(dim=1, keepdim=True).t())             # (B,K)
        indices = torch.argmin(d, dim=1)                                          # (B,)
        z_q = self.embed(indices)                                                 # (B,d_z)
        z_q_st = z_e + (z_q - z_e).detach()                                       # straight-through

        if self.training:
            self._ema_update(z_e.detach(), indices.detach())

        commit_loss   = F.mse_loss(z_e.detach(), z_q)         # ||sg[z_e] - z_q||^2
        codebook_loss = F.mse_loss(z_e, z_q.detach())         # ||z_e - sg[z_q]||^2
        return z_q_st, indices, {"commit": commit_loss, "codebook": codebook_loss}

# ----------------------------
# Decoder
# ----------------------------

class TemporalSeedRepeat(nn.Module):
    """
    Step D0: map (B,d_z) -> (B,W',d0) via FFN + repeat, then add sinusoidal time PE.
    """
    def __init__(self, d_z: int, Wp: int, d0: int):
        super().__init__()
        self.Wp = Wp
        self.d0 = d0
        self.ffn = nn.Sequential(nn.Linear(d_z, d0), nn.GELU(), nn.LayerNorm(d0))

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        B = z_q.shape[0]
        h = self.ffn(z_q)                                # (B,d0)
        Z = h.unsqueeze(1).expand(B, self.Wp, self.d0)   # (B,W',d0)
        Z = Z + sinusoid_posenc(self.Wp, self.d0, device=z_q.device)  # break symmetry
        return Z

class JointSelfAttentionDecode(nn.Module):
    """
    Step D1: per-frame joint attention (broadcast frame seed to joints, add joint-index PE).
    Input:  Z_seed (B,W',d0), J, joint_mask (B,J)
    Output: Hj (B,W',J,d_t)
    """
    def __init__(self, d0: int, d_t: int, n_heads=4, dropout=0.1):
        super().__init__()
        self.pe_proj = nn.Linear(1, d0)  # same scalar PE projection (index/max_index), added before attn
        self.attn = nn.MultiheadAttention(d0, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn  = nn.Sequential(nn.LayerNorm(d0), nn.Linear(d0, d_t), nn.GELU(), nn.LayerNorm(d_t))

    def forward(self, Z_seed: torch.Tensor, J: int, joint_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Wp, d0 = Z_seed.shape
        H = Z_seed.unsqueeze(2).expand(B, Wp, J, d0)                      # (B,W',J,d0)
        if joint_mask is None:
            joint_mask = Z_seed.new_ones(B, J)
        pe = self.pe_proj(make_norm_joint_index(B, J, joint_mask, device=Z_seed.device))  # (B,J,d0)
        H = H + pe.unsqueeze(1)                                           # add joint-index PE
        X = H.reshape(B * Wp, J, d0)                                      # (B*W',J,d0)
        kpm = (joint_mask == 0).unsqueeze(1).expand(B, Wp, J).reshape(B * Wp, J)
        Y, _ = self.attn(X, X, X, key_padding_mask=kpm)                   # (B*W',J,d0)
        Y = self.ffn(Y).view(B, Wp, J, -1)                                # (B,W',J,d_t)
        return Y

class PerJointTemporalConvUp(nn.Module):
    """
    Step D2: per-joint ConvTranspose1D to upsample time from W' to W.
    Input:  Hj (B, W', J, d_t)
    Output: Y  (B, W,  J, d_up)
    """
    def __init__(self, d_in: int, d_out: int, kernel_size=5, stride=2, padding=2,
                 output_padding=1, num_layers=1, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = d_in
        for i in range(num_layers):
            out_ch = d_out if i == num_layers - 1 else d_in
            layers += [
                nn.ConvTranspose1d(in_ch, out_ch,
                                   kernel_size=kernel_size,
                                   stride=(stride if i == 0 else 1),
                                   padding=padding,
                                   output_padding=(output_padding if i == 0 else 0)),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.deconv = nn.Sequential(*layers)

    def forward(self, Hj: torch.Tensor) -> torch.Tensor:
        B, Wp, J, d = Hj.shape
        X = Hj.permute(0, 2, 3, 1).contiguous().view(B * J, d, Wp)  # (B*J,d_in,W')
        Y = self.deconv(X)                                          # (B*J,d_out,W)
        _, d_out, W = Y.shape
        Y = Y.view(B, J, d_out, W).permute(0, 3, 1, 2).contiguous() # (B,W,J,d_out)
        return Y

class OutputHeads(nn.Module):
    """
    Step D3: per-joint heads to predict motion features (e.g., 6D rotations [+ optional]).
    Input:  Y (B,W,J,d_up)
    Output: X_hat (B,W,J,D_joint_out)
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out))

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        return self.head(Y)  # (B,W,J,D_out)

class LimbWindowDecoder(nn.Module):
    """
    Decoder: z_q (B,d_z) -> (B,W,J,D_joint_out)
    Mirrors the encoder’s downsampling with ConvTranspose1D upsampling.
    """
    def __init__(self, d_z: int, Wp: int, d0: int, d_t: int,
                 d_joint_out: int, kernel_size=5, stride=2, padding=2, output_padding=1,
                 n_heads_joint=4, dropout=0.1):
        super().__init__()
        self.seed = TemporalSeedRepeat(d_z, Wp, d0)
        self.joint_attn = JointSelfAttentionDecode(d0, d_t, n_heads_joint, dropout)
        self.up = PerJointTemporalConvUp(d_t, d_in=d_t, d_out=d_t,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, output_padding=output_padding,
                                         num_layers=1, dropout=dropout)
        self.out = OutputHeads(d_t, d_joint_out)

    def forward(self, z_q: torch.Tensor, J: int, joint_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Z_seed = self.seed(z_q)                               # (B,W',d0)
        Hj = self.joint_attn(Z_seed, J, joint_mask)           # (B,W',J,d_t)
        Y  = self.up(Hj)                                      # (B,W,J,d_t)
        X_hat = self.out(Y)                                   # (B,W,J,D_out)
        return X_hat

# ----------------------------
# Full wrapper
# ----------------------------

class LimbVQVAE(nn.Module):
    """
    End-to-end limb VQ-VAE.
    Inputs:
      X: (B,W,J,D_joint)        ground-truth per-joint motion features (e.g., 6D rotations)
      joint_mask: (B,J)         1=valid, 0=pad for variable J
      parents: (J,)             parent array (-1 for limb root)
      bone_lengths: (B,J,1)     per-joint length (for FK-based losses)
    Outputs:
      dict: {
        'z_e': (B,d_z), 'z_q': (B,d_z), 'indices': (B,),
        'recon': (B,W,J,D_joint_out),
        'vq_losses': {'commit': ..., 'codebook': ...},
        'aux': intermediates from encoder
      }
    """
    def __init__(self,
                 d_joint_in: int,         # input per-joint feature dim (e.g., 6)
                 d_joint_out: int,        # output per-joint feature dim (e.g., 6 [+ contact])
                 d_t: int = 256,
                 d_s: int = 256,
                 d_g: int = 256,
                 d_z: int = 256,
                 codebook_size: int = 1024,
                 kernel_size: int = 5,
                 stride: int = 2,
                 padding: int = 2,
                 n_heads_joint: int = 4,
                 n_heads_time: int = 4,
                 n_time_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        # Encoder
        self.encoder = LimbWindowEncoder(d_joint=d_joint_in, d_t=d_t, d_s=d_s, d_g=d_g, d_z=d_z,
                                         kernel_size=kernel_size, stride=stride, padding=padding,
                                         num_temporal_layers=2,
                                         n_heads_joint=n_heads_joint, n_heads_time=n_heads_time,
                                         n_time_layers=n_time_layers, dropout=dropout)
        # Quantizer
        self.vq = VQEMA(codebook_size=codebook_size, d_z=d_z, decay=0.99)
        # Decoder (Wp must match encoder’s temporal shrinkage for a given W; we’ll compute it on the fly)
        self._decoder_cfg = dict(d0=d_t, d_t=d_t, d_joint_out=d_joint_out,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 output_padding=1, n_heads_joint=n_heads_joint, dropout=dropout)
        self.decoder = None  # will be materialized per W via build_decoder(W)

    def build_decoder(self, W: int):
        Wp = self.encoder.out_length(W)
        self.decoder = LimbWindowDecoder(d_z=self.encoder.to_latent.proj[1].out_features if isinstance(self.encoder.to_latent.proj, nn.Sequential) else self.encoder.to_latent.proj.out_features,
                                         Wp=Wp, **self._decoder_cfg)

    def forward(self, X: torch.Tensor, joint_mask: Optional[torch.Tensor],
                parents: Optional[torch.Tensor], bone_lengths: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with on-the-fly decoder construction for given W.
        """
        B, W, J, D = X.shape
        if self.decoder is None:
            self.build_decoder(W)

        # Encoder
        z_e, aux = self.encoder(X, joint_mask)            # (B,d_z)
        # VQ
        z_q, indices, vq_losses = self.vq(z_e)            # (B,d_z), (B,)
        # Decoder
        X_hat = self.decoder(z_q, J=J, joint_mask=joint_mask)  # (B,W,J,D_out)

        out = {
            "z_e": z_e, "z_q": z_q, "indices": indices,
            "recon": X_hat, "vq_losses": vq_losses,
            "aux": aux
        }

        # Optional: return FK tensors for external loss computation
        if X_hat.shape[-1] >= 6 and parents is not None and bone_lengths is not None:
            R_hat = rotation_6d_to_matrix(X_hat[..., :6])     # (B,W,J,3,3)
            out["R_hat"] = R_hat
            out["P_hat"] = fk_local_rotations_to_positions(R_hat, parents, bone_lengths)  # (B,W,J,3)

        return out

# ----------------------------
# Example: construction & a single forward
# ----------------------------
if __name__ == "__main__":
    # Toy shapes
    B, W = 4, 64             # batch, window length
    J = 6                    # joints in the limb (padded in real training)
    D_in = 6                 # per-joint input features (e.g., 6D rotations)
    D_out = 6                # predict 6D rotations (minimal head; add extra dims if needed)

    X = torch.randn(B, W, J, D_in)           # input motion
    joint_mask = torch.ones(B, J)            # all joints valid in this toy example
    parents = torch.tensor([-1, 0, 1, 2, 3, 4])  # chain for illustration
    bone_lengths = torch.ones(B, J, 1) * 0.3

    model = LimbVQVAE(d_joint_in=D_in, d_joint_out=D_out,
                      d_t=256, d_s=256, d_g=256, d_z=256,
                      codebook_size=1024, stride=2, kernel_size=5, padding=2)
    out = model(X, joint_mask, parents, bone_lengths)

    print("Encoder latent z_e:", out["z_e"].shape)
    print("Quantized z_q:", out["z_q"].shape, "Token indices:", out["indices"].shape)
    print("Reconstruction:", out["recon"].shape)
    if "P_hat" in out:
        print("FK positions:", out["P_hat"].shape)
