#!/usr/bin/env python3
"""
ACT (Action Chunking with Transformers) with Frozen Pretrained ResNet18 Features
and O(1) Causal Recurrent Tactile Inference.

Key design for fair, fast visual-tactile evaluation across 10-20 shot industrial tasks:
  - Frozen ImageNet ResNet18 up to layer3 + AdaptiveAvgPool2d((4, 4)) -> (B, 256, 4, 4)
    so 10-shot visual XY localization generalizes cleanly without from-scratch overfitting,
    and offline demo features are pre-extracted once (no ResNet forward/backward per batch).
  - Lightweight trainable SpatialSoftmax(4, 4) + AvgPool2d visual adapter inside ACT.
  - Identical CVAE (d_style=16, kl_weight=0.2) & 2L/2L Transformer (d_model=128, nhead=4,
    chunk_len=8) across all 5 modality combinations:
      ACT_RGB_Proprio, ACT_RGB_RawFT, ACT_RGB_MAE, ACT_RGB_JEPA, ACT_RGB_PAEC.
  - Exact O(1) step_encode_recurrent() helper matching CausalContactBackbone GRU (< 1e-5).
"""

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


RESNET18_LOCAL_CACHE = Path(r"C:\Users\DELL\.cache\torch\hub\checkpoints\resnet18-5c106cde.pth")


class FrozenResNet18Extractor(nn.Module):
    """Frozen ImageNet ResNet18 up to layer3 -> (B, 256, 4, 4) spatial feature maps."""

    def __init__(self, device: torch.device):
        super().__init__()
        if RESNET18_LOCAL_CACHE.exists():
            backbone = resnet18(weights=None)
            backbone.load_state_dict(torch.load(RESNET18_LOCAL_CACHE, map_location="cpu", weights_only=True))
        else:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, img_01: torch.Tensor) -> torch.Tensor:
        # img_01: (B, 3, 64, 64) in [0, 1] -> native layer3 (B, 256, 4, 4)
        x = (img_01 - self.mean) / self.std
        return self.stem(x)


class SpatialSoftmax(nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, width),
            np.linspace(-1.0, 1.0, height),
        )
        self.register_buffer("pos_x", torch.from_numpy(pos_x.reshape(-1)).float())
        self.register_buffer("pos_y", torch.from_numpy(pos_y.reshape(-1)).float())

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        attn = F.softmax(feat.view(B, C, H * W), dim=-1)
        ex = torch.sum(attn * self.pos_x[None, None, :], dim=-1)
        ey = torch.sum(attn * self.pos_y[None, None, :], dim=-1)
        return torch.cat([ex, ey], dim=-1)


class VisualFeatureAdapter(nn.Module):
    """Lightweight trainable adapter from frozen ResNet18 (B, 256, 4, 4) -> (B, d_model)."""

    def __init__(self, in_ch: int = 256, n_kp: int = 32, d_model: int = 96):
        super().__init__()
        self.kp_conv = nn.Conv2d(in_ch, n_kp, kernel_size=1)
        self.ssm = SpatialSoftmax(4, 4)
        self.proj = nn.Sequential(
            nn.Linear(2 * n_kp + in_ch, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

    def forward(self, vis_feat: torch.Tensor) -> torch.Tensor:
        kp = self.ssm(self.kp_conv(vis_feat))  # (B, 64)
        glob = F.adaptive_avg_pool2d(vis_feat, (1, 1)).flatten(1)  # (B, 256)
        return self.proj(torch.cat([kp, glob], dim=-1))


class ACTPolicy(nn.Module):
    """Action Chunking with Transformers (CVAE + Transformer Encoder-Decoder)."""

    def __init__(
        self,
        prop_dim: int = 17,
        contact_dim: int = 0,
        act_dim: int = 7,
        chunk_len: int = 8,
        d_model: int = 96,
        nhead: int = 4,
        num_enc_layers: int = 1,
        num_dec_layers: int = 2,
        d_style: int = 16,
    ):
        super().__init__()
        self.prop_dim = prop_dim
        self.contact_dim = contact_dim
        self.act_dim = act_dim
        self.chunk_len = chunk_len
        self.d_model = d_model
        self.d_style = d_style

        self.vis_adapter = VisualFeatureAdapter(in_ch=256, n_kp=32, d_model=d_model)
        self.prop_proj = nn.Sequential(
            nn.Linear(prop_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.contact_proj = (
            nn.Sequential(
                nn.Linear(contact_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            if contact_dim > 0
            else None
        )

        self.cvae_enc = nn.Sequential(
            nn.Linear(prop_dim + chunk_len * act_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.cvae_mu = nn.Linear(d_model, d_style)
        self.cvae_logvar = nn.Linear(d_model, d_style)
        self.style_proj = nn.Linear(d_style, d_model)

        n_tokens = 3 + (1 if contact_dim > 0 else 0)
        self.src_pos = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, chunk_len, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
        self.action_head = nn.Linear(d_model, act_dim)

    def encode_style(
        self,
        prop: torch.Tensor,
        act_chunk: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if act_chunk is None:
            z = torch.zeros(prop.shape[0], self.d_style, device=prop.device, dtype=prop.dtype)
            return z, None, None
        h = self.cvae_enc(torch.cat([prop, act_chunk.reshape(prop.shape[0], -1)], dim=-1))
        mu = self.cvae_mu(h)
        logvar = self.cvae_logvar(h).clamp(-6.0, 4.0)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return z, mu, logvar

    def forward(
        self,
        vis_feat: torch.Tensor,
        prop: torch.Tensor,
        contact_feat: Optional[torch.Tensor] = None,
        act_chunk: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B = vis_feat.shape[0]
        z_style, mu, logvar = self.encode_style(prop, act_chunk)
        toks = [
            self.style_proj(z_style),
            self.vis_adapter(vis_feat),
            self.prop_proj(prop),
        ]
        if self.contact_proj is not None:
            assert contact_feat is not None
            toks.append(self.contact_proj(contact_feat))
        src = torch.stack(toks, dim=1) + self.src_pos
        mem = self.encoder(src)
        tgt = torch.zeros(B, self.chunk_len, self.d_model, device=vis_feat.device, dtype=vis_feat.dtype)
        dec = self.decoder(tgt + self.query_pos, mem)
        pred_chunk = self.action_head(dec)
        return pred_chunk, mu, logvar

    def compute_loss(
        self,
        vis_feat: torch.Tensor,
        prop: torch.Tensor,
        act_chunk: torch.Tensor,
        contact_feat: Optional[torch.Tensor] = None,
        kl_weight: float = 0.2,
    ) -> Tuple[torch.Tensor, float]:
        pred, mu, logvar = self.forward(vis_feat, prop, contact_feat=contact_feat, act_chunk=act_chunk)
        l_rec = F.l1_loss(pred, act_chunk) + 0.5 * F.mse_loss(pred, act_chunk)
        assert mu is not None and logvar is not None
        l_kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = l_rec + kl_weight * l_kl
        return loss, float(l_rec.item())


@torch.no_grad()
def step_encode_recurrent(
    model: nn.Module,
    w_t: torch.Tensor,
    w_prev: Optional[torch.Tensor],
    prev_a_t: torch.Tensor,
    h_prev: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact O(1) single-step recurrent evaluation for CausalContactBackbone(GRU)
    matching model.encode(w_seq, pa_seq)[:, -1, :] to < 1e-5.
    """
    diff = torch.zeros_like(w_t) if w_prev is None else (w_t - w_prev)
    x_t = torch.cat([w_t, diff, prev_a_t], dim=-1).unsqueeze(1)  # (B, 1, in_dim)
    backbone = model.encoder
    e_t = backbone.in_proj(x_t)  # (B, 1, H)
    h_out, h_next = backbone.temporal(e_t, h_prev)  # (B, 1, H), (L, B, H)
    z_t = backbone.bottleneck(h_out + e_t).squeeze(1)  # (B, d_z)
    return z_t, h_next


class TemporalEnsembler:
    """Fast-reacting ACT exponential temporal ensembling (k_exp=0.25)."""

    def __init__(self, chunk_len: int = 8, act_dim: int = 7, k_exp: float = 0.25):
        self.chunk_len = chunk_len
        self.act_dim = act_dim
        self.k_exp = k_exp
        self.history: list = []

    def reset(self):
        self.history = []

    def step(self, chunk_pred: np.ndarray) -> np.ndarray:
        self.history.append(chunk_pred)
        if len(self.history) > self.chunk_len:
            self.history = self.history[-self.chunk_len :]
        num = np.zeros(self.act_dim, dtype=np.float64)
        den = 0.0
        for age, ch in enumerate(reversed(self.history)):
            if age < ch.shape[0]:
                w = math.exp(-self.k_exp * age)
                num += w * ch[age]
                den += w
        return (num / max(den, 1e-8)).astype(np.float32)
