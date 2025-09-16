"""
Limb VQ-VAE for chain-motion embedding.

This module implements an encoder that processes per-limb motion windows, an EMA VQ quantizer,
and a decoder to reconstruct per-joint motion features (e.g., 6D rotations). Variable joint
counts are supported via joint masks.

Usage example
-------------
>>> import torch
>>> from limb_embedding.models.limb_vqvae import LimbVQVAE
>>> B, W, J, Din = 2, 64, 8, 6
>>> X = torch.randn(B, W, J, Din)
>>> joint_mask = torch.ones(B, J)
>>> parents = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6])
>>> bone_lengths = torch.ones(B, J, 1) * 0.3
>>> model = LimbVQVAE(d_joint_in=6, d_joint_out=6)
>>> out = model(X, joint_mask, parents, bone_lengths)
>>> out['recon'].shape
torch.Size([2, 64, 8, 6])
"""
from __future__ import annotations

from typing import Optional, Dict, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1d_out_length(L_in: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    return math.floor((L_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


def convt1d_out_length(L_in: int, kernel: int, stride: int, padding: int,
                       output_padding: int = 0, dilation: int = 1) -> int:
    return (L_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1


def make_norm_joint_index(B: int, J: int, joint_mask: Optional[torch.Tensor] = None, device=None) -> torch.Tensor:
    """Create normalized joint index features in [0,1] per batch.

    Parameters
    - B: Batch size
    - J: Number of joints (including padding)
    - joint_mask: Optional mask of shape (B, J) with 1 for valid joints
    - device: Optional device

    Returns
    - Tensor of shape (B, J, 1)
    """
    device = device or (joint_mask.device if joint_mask is not None else "cpu")
    base = torch.arange(J, device=device).float().unsqueeze(0).expand(B, J)
    if joint_mask is None:
        denom = (J - 1) if J > 1 else 1.0
        return (base / denom).unsqueeze(-1)
    valid_counts = joint_mask.sum(dim=1).clamp(min=1).float()
    denom = (valid_counts - 1).clamp(min=1.0).unsqueeze(1)
    out = (base / denom) * joint_mask
    return out.unsqueeze(-1)


def sinusoid_posenc(T: int, D: int, device=None) -> torch.Tensor:
    """Standard sinusoidal positional encoding (1, T, D)."""
    device = device or "cpu"
    pe = torch.zeros(T, D, device=device)
    position = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D representation to rotation matrix (..., 3, 3)."""
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def geodesic_loss_so3(R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean geodesic distance between rotation matrices (..., 3, 3)."""
    M = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    cos = ((M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2]) - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos)
    return theta.mean()


def fk_local_rotations_to_positions(R_local: torch.Tensor,
                                    parents: torch.Tensor,
                                    bone_lengths: torch.Tensor) -> torch.Tensor:
    """Dummy FK: accumulate along the chain using local z as bone direction.

    Parameters
    - R_local: (B, W, J, 3, 3)
    - parents: (J,)
    - bone_lengths: (B, J, 1)

    Returns
    - Positions: (B, W, J, 3)
    """
    B, W, J = R_local.shape[:3]
    device = R_local.device
    pos = torch.zeros(B, W, J, 3, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
    for j in range(J):
        parent = int(parents[j].item()) if isinstance(parents, torch.Tensor) else int(parents[j])
        dir_local = (R_local[..., j, :, :] @ z_axis)
        seg = dir_local * bone_lengths[..., j, 0].unsqueeze(-1)
        if parent == -1:
            pos[..., j, :] = seg
        else:
            pos[..., j, :] = pos[..., parent, :] + seg
    return pos


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, with_proj: bool = False):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))
        self.proj = nn.Linear(dim, dim) if with_proj else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.query.view(1, 1, D).expand(B, T, D)
        logits = (self.proj(x) * q).sum(dim=-1)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(logits, dim=1)
        return (attn.unsqueeze(-1) * x).sum(dim=1)


class PerJointTemporalConvDown(nn.Module):
    def __init__(self, d_in: int, d_t: int, kernel_size=5, stride=2, padding=2, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        cur = d_in
        for _ in range(num_layers):
            layers += [nn.Conv1d(cur, d_t, kernel_size, stride=stride, padding=padding), nn.GELU(), nn.Dropout(dropout)]
            cur = d_t
        self.net = nn.Sequential(*layers)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, W, J, Din = X.shape
        H = X.permute(0, 2, 3, 1).reshape(B * J, Din, W)
        H = self.net(H)
        _, d_t, Wp = H.shape
        return H.view(B, J, d_t, Wp).permute(0, 3, 1, 2)

    def out_length(self, W: int) -> int:
        return conv1d_out_length(W, self.kernel_size, self.stride, self.padding)


class PerFrameJointSelfAttention(nn.Module):
    def __init__(self, d_t: int, d_s: int, n_heads=4, dropout=0.1):
        super().__init__()
        self.pe_proj = nn.Linear(1, d_t)
        self.attn = nn.MultiheadAttention(d_t, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(d_t), nn.Linear(d_t, d_s), nn.GELU(), nn.LayerNorm(d_s))

    def forward(self, Ht: torch.Tensor, joint_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, Wp, J, d_t = Ht.shape
        if joint_mask is None:
            joint_mask = Ht.new_ones(B, J)
        pe = self.pe_proj(make_norm_joint_index(B, J, joint_mask, device=Ht.device))
        Ht_pe = Ht + pe.unsqueeze(1)
        X = Ht_pe.reshape(B * Wp, J, d_t)
        kpm = (joint_mask == 0).unsqueeze(1).expand(B, Wp, J).reshape(B * Wp, J)
        Y, _ = self.attn(X, X, X, key_padding_mask=kpm)
        Y = Y.view(B, Wp, J, d_t)
        return self.ffn(Y)


class FrameDescriptor(nn.Module):
    def __init__(self, d_s: int, with_proj: bool = False):
        super().__init__()
        self.pool = AttentionPooling(d_s, with_proj=with_proj)

    def forward(self, Hj: torch.Tensor, joint_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, Wp, J, d = Hj.shape
        if joint_mask is None:
            joint_mask = Hj.new_ones(B, J)
        mask = joint_mask.unsqueeze(1).expand(B, Wp, J)
        return self.pool(Hj.reshape(B * Wp, J, d), mask.reshape(B * Wp, J)).reshape(B, Wp, d)


class TemporalBlock(nn.Module):
    def __init__(self, d_in: int, d_g: int, n_heads=4, n_layers=1, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_g) if d_in != d_g else nn.Identity()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_g, nhead=n_heads,
                                               dim_feedforward=4 * d_g, batch_first=True,
                                               dropout=dropout, activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        return self.enc(self.in_proj(F))


class WindowLatent(nn.Module):
    def __init__(self, d_g: int, d_z: int, dropout=0.1):
        super().__init__()
        self.pool = AttentionPooling(d_g)
        self.proj = nn.Sequential(nn.LayerNorm(d_g), nn.Linear(d_g, d_z), nn.Dropout(dropout))

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(G))


class LimbWindowEncoder(nn.Module):
    def __init__(self, d_joint: int, d_t=256, d_s=256, d_g=256, d_z=256,
                 kernel_size=5, stride=2, padding=2, num_temporal_layers=2,
                 n_heads_joint=4, n_heads_time=4, n_time_layers=1, dropout=0.1):
        super().__init__()
        self.temporal = PerJointTemporalConvDown(d_joint, d_t, kernel_size, stride, padding,
                                                 num_temporal_layers, dropout)
        self.joint_attn = PerFrameJointSelfAttention(d_t, d_s, n_heads_joint, dropout)
        self.frame_desc = FrameDescriptor(d_s, with_proj=False)
        self.temp_block = TemporalBlock(d_s, d_g, n_heads_time, n_time_layers, dropout)
        self.to_latent = WindowLatent(d_g, d_z, dropout)

    def forward(self, X: torch.Tensor, joint_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        Ht = self.temporal(X)
        Hj = self.joint_attn(Ht, joint_mask)
        F = self.frame_desc(Hj, joint_mask)
        G = self.temp_block(F)
        z = self.to_latent(G)
        return z, {"Ht": Ht, "Hj": Hj, "F": F, "G": G}

    def out_length(self, W: int) -> int:
        return self.temporal.out_length(W)


class VQEMA(nn.Module):
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
    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor) -> None:
        one_hot = F.one_hot(indices, num_classes=self.codebook_size).type_as(z_e)
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.t() @ z_e
        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embed.weight.data.copy_(embed_normalized)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        d = (z_e.pow(2).sum(dim=1, keepdim=True)
             - 2 * z_e @ self.embed.weight.t()
             + self.embed.weight.pow(2).sum(dim=1, keepdim=True).t())
        indices = torch.argmin(d, dim=1)
        z_q = self.embed(indices)
        z_q_st = z_e + (z_q - z_e).detach()
        if self.training:
            self._ema_update(z_e.detach(), indices.detach())
        commit_loss = F.mse_loss(z_e.detach(), z_q)
        codebook_loss = F.mse_loss(z_e, z_q.detach())
        return z_q_st, indices, {"commit": commit_loss, "codebook": codebook_loss}


class TemporalSeedRepeat(nn.Module):
    def __init__(self, d_z: int, Wp: int, d0: int):
        super().__init__()
        self.Wp = Wp
        self.d0 = d0
        self.ffn = nn.Sequential(nn.Linear(d_z, d0), nn.GELU(), nn.LayerNorm(d0))

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        B = z_q.shape[0]
        h = self.ffn(z_q)
        Z = h.unsqueeze(1).expand(B, self.Wp, self.d0)
        Z = Z + sinusoid_posenc(self.Wp, self.d0, device=z_q.device)
        return Z


class JointSelfAttentionDecode(nn.Module):
    def __init__(self, d0: int, d_t: int, n_heads=4, dropout=0.1):
        super().__init__()
        self.pe_proj = nn.Linear(1, d0)
        self.attn = nn.MultiheadAttention(d0, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(d0), nn.Linear(d0, d_t), nn.GELU(), nn.LayerNorm(d_t))

    def forward(self, Z_seed: torch.Tensor, J: int, joint_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Wp, d0 = Z_seed.shape
        H = Z_seed.unsqueeze(2).expand(B, Wp, J, d0)
        if joint_mask is None:
            joint_mask = Z_seed.new_ones(B, J)
        pe = self.pe_proj(make_norm_joint_index(B, J, joint_mask, device=Z_seed.device))
        H = H + pe.unsqueeze(1)
        X = H.reshape(B * Wp, J, d0)
        kpm = (joint_mask == 0).unsqueeze(1).expand(B, Wp, J).reshape(B * Wp, J)
        Y, _ = self.attn(X, X, X, key_padding_mask=kpm)
        return self.ffn(Y).view(B, Wp, J, -1)


class PerJointTemporalConvUp(nn.Module):
    def __init__(self, d_in: int, d_out: int, kernel_size=5, stride=2, padding=2,
                 output_padding=1, num_layers=1, dropout=0.1):
        super().__init__()
        layers = []
        cur = d_in
        for i in range(num_layers):
            layers += [nn.ConvTranspose1d(cur, d_out, kernel_size, stride=stride, padding=padding,
                                          output_padding=output_padding), nn.GELU(), nn.Dropout(dropout)]
            cur = d_out
        self.net = nn.Sequential(*layers)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, Hj: torch.Tensor, W_out: int) -> torch.Tensor:
        B, Wp, J, d_t = Hj.shape
        H = Hj.permute(0, 2, 3, 1).reshape(B * J, d_t, Wp)
        Y = self.net(H)
        Y = Y[..., :W_out]
        _, d_up, W = Y.shape
        return Y.view(B, J, d_up, W).permute(0, 3, 1, 2)

    def out_length(self, Wp: int) -> int:
        return convt1d_out_length(Wp, self.kernel_size, self.stride, self.padding, output_padding=self.output_padding)


class JointOutputHead(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out))

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        return self.proj(Y)


class LimbWindowDecoder(nn.Module):
    def __init__(self, d_z: int, Wp: int, d0: int, d_t: int, d_joint_out: int,
                 kernel_size=5, stride=2, padding=2, output_padding=1, n_heads_joint=4, dropout=0.1):
        super().__init__()
        self.seed = TemporalSeedRepeat(d_z, Wp, d0)
        self.joint_attn = JointSelfAttentionDecode(d0, d_t, n_heads=n_heads_joint, dropout=dropout)
        self.up = PerJointTemporalConvUp(d_in=d_t, d_out=d_t, kernel_size=kernel_size, stride=stride,
                                         padding=padding, output_padding=output_padding, num_layers=1, dropout=dropout)
        self.out_head = JointOutputHead(d_in=d_t, d_out=d_joint_out)

    def forward(self, z_q: torch.Tensor, J: int, joint_mask: Optional[torch.Tensor], W_out: Optional[int] = None) -> torch.Tensor:
        B = z_q.shape[0]
        Z_seed = self.seed(z_q)
        Hj = self.joint_attn(Z_seed, J=J, joint_mask=joint_mask)
        W_out = W_out or (self.seed.Wp * 2)  # heuristic if not provided
        Y = self.up(Hj, W_out=W_out)
        return self.out_head(Y)


class LimbVQVAE(nn.Module):
    def __init__(self,
                 d_joint_in: int,
                 d_joint_out: int,
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
        self.encoder = LimbWindowEncoder(d_joint=d_joint_in, d_t=d_t, d_s=d_s, d_g=d_g, d_z=d_z,
                                         kernel_size=kernel_size, stride=stride, padding=padding,
                                         num_temporal_layers=2,
                                         n_heads_joint=n_heads_joint, n_heads_time=n_heads_time,
                                         n_time_layers=n_time_layers, dropout=dropout)
        self.vq = VQEMA(codebook_size=codebook_size, d_z=d_z, decay=0.99)
        self._decoder_cfg = dict(d0=d_t, d_t=d_t, d_joint_out=d_joint_out,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 output_padding=1, n_heads_joint=n_heads_joint, dropout=dropout)
        self.decoder: Optional[LimbWindowDecoder] = None

    def build_decoder(self, W: int) -> None:
        Wp = self.encoder.out_length(W)
        self.decoder = LimbWindowDecoder(d_z=self.encoder.to_latent.proj[1].out_features if isinstance(self.encoder.to_latent.proj, nn.Sequential) else self.encoder.to_latent.proj.out_features,
                                         Wp=Wp, **self._decoder_cfg)

    def forward(self, X: torch.Tensor, joint_mask: Optional[torch.Tensor],
                parents: Optional[torch.Tensor], bone_lengths: Optional[torch.Tensor]) -> Dict[str, Any]:
        B, W, J, _ = X.shape
        if self.decoder is None:
            self.build_decoder(W)
        z_e, aux = self.encoder(X, joint_mask)
        z_q, indices, vq_losses = self.vq(z_e)
        X_hat = self.decoder(z_q, J=J, joint_mask=joint_mask, W_out=W)
        out: Dict[str, Any] = {"z_e": z_e, "z_q": z_q, "indices": indices, "recon": X_hat, "vq_losses": vq_losses, "aux": aux}
        if X_hat.shape[-1] >= 6 and parents is not None and bone_lengths is not None:
            R_hat = rotation_6d_to_matrix(X_hat[..., :6])
            out["R_hat"] = R_hat
            out["P_hat"] = fk_local_rotations_to_positions(R_hat, parents, bone_lengths)
        return out

