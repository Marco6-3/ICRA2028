#!/usr/bin/env python3
"""
V4 Route 1: Physics-Anchored Action-Effect Contact Compression (PAEC)
& Matched Baselines (MAE, Abstract Latent JEPA + VICReg, Raw End-to-End).

Core principle:
  Escape the abstract-latent JEPA trilemma (collapse vs. VICReg noise inflation)
  by compressing causal contact history into a tight low-dimensional bottleneck
  z_t in R^{d_z} (d_z=16) trained to predict self-recorded, zero-human-label
  future physical contact mechanics deltas (6D Wrench + Gripper Force + Tactile):

      delta_y_hat_{t, d} = b_phi(z_t, d) + g_phi(z_t, a_{t:t+d}, d),
      where g_phi(z_t, 0, d) == 0 identically.

  At test time, frozen z_t also supports:
    - Zero-dither Virtual Directional Compliance Probe: g_phi(z_t, +/- e_k, d_1)
    - Calibrated Physical Wrench Surprise: ||delta_w_true - delta_w_hat||_2
"""

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalContactBackbone(nn.Module):
    """
    Strictly causal temporal encoder compressing contact+executed-action history
    into a compact bottleneck z_t in R^{d_z}.
    Supports backbone_type in {'gru', 'short_win', 'single_frame'}.
    """

    def __init__(
        self,
        in_dim: int,
        d_z: int = 16,
        hidden_dim: int = 96,
        backbone_type: str = "gru",
        short_win: int = 4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_z = d_z
        self.hidden_dim = hidden_dim
        self.backbone_type = backbone_type
        self.short_win = short_win

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        if backbone_type == "gru":
            self.temporal = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
            )
        elif backbone_type == "short_win":
            self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=short_win, padding=0)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=short_win, padding=0)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        elif backbone_type == "single_frame":
            self.frame_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
        else:
            raise ValueError(f"Unknown backbone_type={backbone_type}")

        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, d_z),
            nn.LayerNorm(d_z),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.in_proj(x)
        if self.backbone_type == "gru":
            h, _ = self.temporal(e)
        elif self.backbone_type == "short_win":
            u = e.transpose(1, 2)
            u = F.pad(u, (self.short_win - 1, 0))
            u = F.silu(self.ln1(self.conv1(u).transpose(1, 2))).transpose(1, 2)
            u = F.pad(u, (self.short_win - 1, 0))
            h = F.silu(self.ln2(self.conv2(u).transpose(1, 2)))
        else:
            h = self.frame_mlp(e)
        return self.bottleneck(h + e)


def build_causal_contact_input(
    contact_obs: torch.Tensor,
    act_exec: torch.Tensor,
    use_act_exec: bool = True,
    use_causal_diff: bool = True,
) -> torch.Tensor:
    """
    Construct strictly causal encoder input from (contact_obs, act_exec).
    Includes causal first difference (c_t - c_{t-1}, zero at t=0) which is
    analytically invariant to static sensor zero-point drift.
    """
    parts = [contact_obs]
    if use_causal_diff:
        diff = torch.zeros_like(contact_obs)
        diff[:, 1:, :] = contact_obs[:, 1:, :] - contact_obs[:, :-1, :]
        parts.append(diff)
    if use_act_exec:
        parts.append(act_exec)
    return torch.cat(parts, dim=-1)


class DecomposedPhysicsPredictor(nn.Module):
    """
    Predicts multi-horizon physical contact observable delta (y_{t+d} - y_t):
        delta_y_hat = b_phi(z_t, h) + [g_raw(z_t, a, h) - g_raw(z_t, 0, h)]
    Exact structural guarantee: g_phi(z_t, 0, h) == 0.0 bit-for-bit.
    """

    def __init__(
        self,
        d_z: int,
        act_dim: int,
        target_dim: int,
        n_horizons: int,
        use_bg_split: bool = True,
    ):
        super().__init__()
        self.d_z = d_z
        self.act_dim = act_dim
        self.target_dim = target_dim
        self.use_bg_split = use_bg_split
        self.h_emb = nn.Embedding(n_horizons, 16)

        if use_bg_split:
            self.baseline_net = nn.Sequential(
                nn.Linear(d_z + 16, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, target_dim),
            )
            self.effect_net = nn.Sequential(
                nn.Linear(d_z + act_dim + 16, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, target_dim, bias=False),
            )
        else:
            self.joint_net = nn.Sequential(
                nn.Linear(d_z + act_dim + 16, 160),
                nn.LayerNorm(160),
                nn.SiLU(),
                nn.Linear(160, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, target_dim),
            )

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        h_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix = z.shape[:-1]
        h = self.h_emb.weight[h_idx].view(*([1] * len(prefix)), -1).expand(*prefix, -1)
        if self.use_bg_split:
            b = self.baseline_net(torch.cat([z, h], dim=-1))
            g_a = self.effect_net(torch.cat([z, a, h], dim=-1))
            g_0 = self.effect_net(torch.cat([z, torch.zeros_like(a), h], dim=-1))
            g = g_a - g_0
            return b + g, b, g
        else:
            y = self.joint_net(torch.cat([z, a, h], dim=-1))
            return y, torch.zeros_like(y), y


class PAECModel(nn.Module):
    """
    Physics-Anchored Action-Effect Contact Compression (PAEC).
    """

    def __init__(
        self,
        contact_dim: int,
        wrench_dim: int = 6,
        act_dim: int = 7,
        d_z: int = 16,
        horizons: Tuple[int, ...] = (1, 3, 6),
        backbone_type: str = "gru",
        use_bg_split: bool = True,
        use_act_exec: bool = True,
        use_causal_diff: bool = True,
        drift_aug_std: float = 0.35,
    ):
        super().__init__()
        self.contact_dim = contact_dim
        self.wrench_dim = wrench_dim
        self.act_dim = act_dim
        self.d_z = d_z
        self.horizons = tuple(horizons)
        self.use_bg_split = use_bg_split
        self.use_act_exec = use_act_exec
        self.use_causal_diff = use_causal_diff
        self.drift_aug_std = drift_aug_std

        in_dim = contact_dim * (2 if use_causal_diff else 1) + (act_dim if use_act_exec else 0)
        self.encoder = CausalContactBackbone(in_dim=in_dim, d_z=d_z, hidden_dim=96, backbone_type=backbone_type)
        self.predictor = DecomposedPhysicsPredictor(
            d_z=d_z,
            act_dim=act_dim,
            target_dim=contact_dim,
            n_horizons=len(horizons),
            use_bg_split=use_bg_split,
        )
        # 6 cardinal virtual probes (+/-x, +/-y, +/-z) -> 3D force response each = 18D
        self.probe_dim = 18

    def encode(self, contact_obs: torch.Tensor, act_exec: torch.Tensor) -> torch.Tensor:
        x = build_causal_contact_input(
            contact_obs, act_exec, use_act_exec=self.use_act_exec, use_causal_diff=self.use_causal_diff
        )
        return self.encoder(x)

    @staticmethod
    def aggregate_action(act_next: torch.Tensor, delta: int) -> torch.Tensor:
        B, T, A = act_next.shape
        cs = torch.cumsum(act_next, dim=1)
        zeros = torch.zeros(B, 1, A, device=act_next.device, dtype=act_next.dtype)
        cs_pad = torch.cat([zeros, cs], dim=1)
        return (cs_pad[:, delta:T, :] - cs_pad[:, 0 : T - delta, :]) / math.sqrt(float(delta))

    def compute_loss(
        self,
        contact_obs: torch.Tensor,
        act_exec: torch.Tensor,
        act_next: torch.Tensor,
        contact_weight: Optional[torch.Tensor] = None,
        hold_thresh: float = 0.25,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Optional zero-point sensor drift augmentation on encoder input ONLY
        # (Target deltas y_{t+d} - y_t are analytically invariant to constant drift!)
        if self.training and self.drift_aug_std > 0.0:
            B, _, D = contact_obs.shape
            drift = torch.randn(B, 1, D, device=contact_obs.device) * self.drift_aug_std
            # Apply to 70% of batch episodes so clean zero-bias episodes are also seen
            mask = (torch.rand(B, 1, 1, device=contact_obs.device) < 0.7).float()
            obs_in = contact_obs + drift * mask
        else:
            obs_in = contact_obs

        z = self.encode(obs_in, act_exec)
        losses, w_losses, b_norms, g_norms = [], [], [], []
        for h_idx, delta in enumerate(self.horizons):
            if contact_obs.shape[1] <= delta + 1:
                continue
            z_cur = z[:, :-delta, :]
            a_agg = self.aggregate_action(act_next, delta)
            dy_true = contact_obs[:, delta:, :] - contact_obs[:, :-delta, :]
            dy_hat, b_term, g_term = self.predictor(z_cur, a_agg, h_idx)

            # Weight 6D wrench higher (2.0x) so core force/torque mechanics dominate over noisy high-dim embeddings
            dim_w = torch.ones(self.contact_dim, device=contact_obs.device, dtype=contact_obs.dtype)
            dim_w[: self.wrench_dim] = 2.0
            per_step = (F.smooth_l1_loss(dy_hat, dy_true, reduction="none") * dim_w).mean(dim=-1)
            w_err = F.mse_loss(dy_hat[..., : self.wrench_dim], dy_true[..., : self.wrench_dim])

            w = torch.ones_like(per_step)
            if contact_weight is not None:
                w = w * contact_weight[:, :-delta]
            if self.use_bg_split:
                is_hold = (torch.norm(a_agg, dim=-1) < hold_thresh).float()
                w = w * (1.0 + 1.0 * is_hold)

            losses.append((per_step * w).sum() / (w.sum() + 1e-6))
            w_losses.append(float(w_err.item()))
            b_norms.append(float(b_term.detach().norm(dim=-1).mean().item()))
            g_norms.append(float(g_term.detach().norm(dim=-1).mean().item()))

        total = torch.stack(losses).mean()
        return total, {
            "loss": float(total.item()),
            "wrench_delta_mse": float(sum(w_losses) / max(len(w_losses), 1)),
            "b_norm": float(sum(b_norms) / max(len(b_norms), 1)),
            "g_norm": float(sum(g_norms) / max(len(g_norms), 1)),
        }

    @torch.no_grad()
    def compute_virtual_probe(self, z: torch.Tensor, step_scale: float = 1.0) -> torch.Tensor:
        """
        Zero-dither virtual directional compliance probe:
        Queries g_phi(z_t, +/- step_scale * e_k, h=0) along x, y, z axes
        and returns predicted 3D force response [Fx, Fy, Fz] for all 6 pokes -> (..., 18).
        """
        prefix = z.shape[:-1]
        outs = []
        for axis in range(3):
            for sign in (1.0, -1.0):
                a_v = torch.zeros(*prefix, self.act_dim, device=z.device, dtype=z.dtype)
                a_v[..., axis] = sign * step_scale
                dy, _, g = self.predictor(z, a_v, h_idx=0)
                resp = g if self.use_bg_split else dy
                outs.append(resp[..., :3])
        return torch.cat(outs, dim=-1)

    @torch.no_grad()
    def compute_causal_surprise(
        self,
        contact_obs: torch.Tensor,
        act_exec: torch.Tensor,
        horizon_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Strictly causal rollout returning:
          z: (B, T, d_z)
          probe: (B, T, 18) virtual directional compliance readout
          surprise: (B, T) physical wrench prediction residual ||dw_true - dw_hat||_2
        """
        delta = self.horizons[horizon_idx]
        z = self.encode(contact_obs, act_exec)
        probe = self.compute_virtual_probe(z)
        B, T, _ = z.shape
        surprise = torch.zeros(B, T, device=z.device, dtype=z.dtype)
        if T > delta:
            cs = torch.cumsum(act_exec, dim=1)
            a_past = (cs[:, delta:, :] - cs[:, :-delta, :]) / math.sqrt(float(delta))
            dy_hat, _, _ = self.predictor(z[:, :-delta, :], a_past, horizon_idx)
            dw_true = contact_obs[:, delta:, : self.wrench_dim] - contact_obs[:, :-delta, : self.wrench_dim]
            dw_hat = dy_hat[..., : self.wrench_dim]
            surprise[:, delta:] = torch.norm(dw_true - dw_hat, dim=-1)
        return z, probe, surprise


class MAEContactRecon(nn.Module):
    """Matched-capacity Masked/Future Absolute Signal Reconstruction baseline (B2)."""

    def __init__(self, contact_dim: int, act_dim: int = 7, d_z: int = 16, mask_ratio: float = 0.30):
        super().__init__()
        self.contact_dim = contact_dim
        self.d_z = d_z
        self.mask_ratio = mask_ratio
        in_dim = contact_dim * 2 + act_dim
        self.encoder = CausalContactBackbone(in_dim=in_dim, d_z=d_z, hidden_dim=96, backbone_type="gru")
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, contact_dim))
        self.dec_cur = nn.Sequential(nn.Linear(d_z, 96), nn.SiLU(), nn.Linear(96, contact_dim))
        self.dec_fut = nn.Sequential(nn.Linear(d_z, 96), nn.SiLU(), nn.Linear(96, contact_dim))

    def encode(self, contact_obs: torch.Tensor, act_exec: torch.Tensor) -> torch.Tensor:
        x = build_causal_contact_input(contact_obs, act_exec, use_act_exec=True, use_causal_diff=True)
        return self.encoder(x)

    def compute_loss(self, contact_obs: torch.Tensor, act_exec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T, D = contact_obs.shape
        mask = (torch.rand(B, T, 1, device=contact_obs.device) < self.mask_ratio).float()
        obs_m = contact_obs * (1.0 - mask) + self.mask_tok * mask
        z = self.encode(obs_m, act_exec)
        l_cur = F.mse_loss(self.dec_cur(z), contact_obs)
        l_fut = F.mse_loss(self.dec_fut(z[:, :-1, :]), contact_obs[:, 1:, :])
        loss = l_cur + l_fut
        return loss, {"loss": float(loss.item()), "l_cur": float(l_cur.item()), "l_fut": float(l_fut.item())}


class AbstractLatentJEPAVICReg(nn.Module):
    """
    Matched Abstract-Latent JEPA + EMA + VICReg baseline (B3).
    Directly contrasts abstract latent z training against Physics-Anchored PAEC
    on the exact same F/T + tactile dataset.
    """

    def __init__(self, contact_dim: int, act_dim: int = 7, d_z: int = 64, horizons: Tuple[int, ...] = (1, 3, 6)):
        super().__init__()
        self.d_z = d_z
        self.horizons = tuple(horizons)
        in_dim = contact_dim * 2 + act_dim
        self.encoder = CausalContactBackbone(in_dim=in_dim, d_z=d_z, hidden_dim=96, backbone_type="gru")
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = DecomposedPhysicsPredictor(
            d_z=d_z, act_dim=act_dim, target_dim=d_z, n_horizons=len(horizons), use_bg_split=False
        )

    @torch.no_grad()
    def update_ema(self, decay: float = 0.992):
        for p_on, p_t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(decay).add_(p_on.data, alpha=1.0 - decay)

    def encode(self, contact_obs: torch.Tensor, act_exec: torch.Tensor) -> torch.Tensor:
        x = build_causal_contact_input(contact_obs, act_exec, use_act_exec=True, use_causal_diff=True)
        return self.encoder(x)

    def compute_loss(
        self, contact_obs: torch.Tensor, act_exec: torch.Tensor, act_next: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = build_causal_contact_input(contact_obs, act_exec, use_act_exec=True, use_causal_diff=True)
        z = self.encoder(x)
        with torch.no_grad():
            z_tgt = self.target_encoder(x)

        preds = []
        for h_idx, delta in enumerate(self.horizons):
            if contact_obs.shape[1] <= delta + 1:
                continue
            a_agg = PAECModel.aggregate_action(act_next, delta)
            dz_hat, _, _ = self.predictor(z[:, :-delta, :], a_agg, h_idx)
            preds.append(F.smooth_l1_loss(z[:, :-delta, :] + dz_hat, z_tgt[:, delta:, :]))
        l_pred = torch.stack(preds).mean()

        # VICReg anti-collapse
        flat = z.reshape(-1, self.d_z)
        flat = flat - flat.mean(dim=0, keepdim=True)
        std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
        l_var = F.relu(1.0 - std).mean()
        cov = (flat.T @ flat) / max(len(flat) - 1, 1)
        off = cov - torch.diag(torch.diag(cov))
        l_cov = (off ** 2).sum() / self.d_z
        loss = l_pred + 0.5 * l_var + 0.05 * l_cov
        return loss, {"loss": float(loss.item()), "l_pred": float(l_pred.item()), "std": float(std.mean().item())}


class DownstreamActionChunkPolicy(nn.Module):
    """
    Matched downstream Action-Chunk BC policy predicting next chunk_len actions.
    """

    def __init__(
        self,
        proprio_hist_dim: int,
        feat_dim: int,
        act_dim: int = 7,
        chunk_len: int = 8,
        mode: str = "frozen_feat",
        raw_contact_dim: int = 0,
    ):
        super().__init__()
        self.mode = mode
        self.chunk_len = chunk_len
        self.act_dim = act_dim

        if mode == "raw_end2end":
            self.raw_enc = CausalContactBackbone(in_dim=raw_contact_dim, d_z=16, hidden_dim=96, backbone_type="gru")
            eff_dim = 16
        elif mode == "frozen_feat":
            eff_dim = feat_dim
        elif mode == "proprio_only":
            eff_dim = 0
        else:
            raise ValueError(f"Unknown mode={mode}")

        in_dim = proprio_hist_dim + eff_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, chunk_len * act_dim),
        )

    def forward(
        self,
        prop_hist: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
        raw_win: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "proprio_only":
            x = prop_hist
        elif self.mode == "frozen_feat":
            assert feat is not None
            x = torch.cat([prop_hist, feat], dim=-1)
        else:
            assert raw_win is not None
            z = self.raw_enc(raw_win)[:, -1, :]
            x = torch.cat([prop_hist, z], dim=-1)
        return self.net(x).view(-1, self.chunk_len, self.act_dim)


def compute_effective_rank(z: torch.Tensor) -> float:
    with torch.no_grad():
        x = z.reshape(-1, z.shape[-1]).float()
        x = x - x.mean(dim=0, keepdim=True)
        if x.shape[0] > 8192:
            idx = torch.linspace(0, x.shape[0] - 1, 8192, device=x.device).long()
            x = x[idx]
        s = torch.linalg.svdvals(x)
        p = (s ** 2) / (torch.sum(s ** 2) + 1e-12)
        return float(torch.exp(-torch.sum(p * torch.log(p + 1e-12))).item())
