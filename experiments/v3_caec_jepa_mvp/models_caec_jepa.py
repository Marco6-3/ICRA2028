#!/usr/bin/env python3
"""
V3 CAEC-JEPA & Matched Baselines (PyTorch)

Implements:
1. CausalContactEncoder (pure contact stream -> z_tau, ZERO proprioception shortcut;
   supports backbone_type in {'gru', 'short_win', 'single_frame'} for E1 history ablations).
2. SlowContextEncoder (strictly causal rolling interaction statistics -> c_tau).
3. DecomposedActionEffectPredictor:
       z_hat_{t+d} = z_t + b_phi(z_t, c_t, d) + g_phi(z_t, c_t, a_{t:t+d}, d)
   with exact structural zero-action guarantee g_phi(..., a=0, ...) == 0.
4. CAECJEPA (full model + clean ablation switches: use_ctx, use_bg_split, use_idm, use_event_pool).
5. MAEReconContactModel (matched backbone, masked + future spectrogram reconstruction).
6. DownstreamActionChunkPolicy (matched downstream action-chunk BC policy for E2 low-shot tests).
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalContactEncoder(nn.Module):
    """
    Pure contact-stream causal encoder (NO proprioception input -> zero proprio shortcut).
    Input:  aud (B, T, audio_dim)
    Output: z   (B, T, d_z), e_in (B, T, d_z)
    """

    def __init__(
        self,
        audio_dim: int,
        d_z: int = 64,
        backbone_type: str = "gru",
        short_win: int = 4,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.d_z = d_z
        self.backbone_type = backbone_type
        self.short_win = short_win

        self.in_proj = nn.Sequential(
            nn.Linear(audio_dim, d_z),
            nn.LayerNorm(d_z),
            nn.SiLU(),
            nn.Linear(d_z, d_z),
            nn.LayerNorm(d_z),
            nn.SiLU(),
        )

        if backbone_type == "gru":
            self.temporal = nn.GRU(
                input_size=d_z,
                hidden_size=d_z,
                num_layers=2,
                batch_first=True,
            )
        elif backbone_type == "short_win":
            # Strictly causal 1D convolution with kernel_size = short_win
            self.conv1 = nn.Conv1d(d_z, d_z, kernel_size=short_win, padding=0)
            self.conv2 = nn.Conv1d(d_z, d_z, kernel_size=short_win, padding=0)
            self.ln1 = nn.LayerNorm(d_z)
            self.ln2 = nn.LayerNorm(d_z)
        elif backbone_type == "single_frame":
            self.frame_mlp = nn.Sequential(
                nn.Linear(d_z, d_z),
                nn.LayerNorm(d_z),
                nn.SiLU(),
                nn.Linear(d_z, d_z),
                nn.LayerNorm(d_z),
                nn.SiLU(),
            )
        else:
            raise ValueError(f"Unknown backbone_type={backbone_type}")

        self.out_proj = nn.Sequential(
            nn.Linear(d_z, d_z),
            nn.LayerNorm(d_z),
        )

    def forward(self, aud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.in_proj(aud)  # (B, T, d_z)
        if self.backbone_type == "gru":
            h, _ = self.temporal(e)
        elif self.backbone_type == "short_win":
            x = e.transpose(1, 2)  # (B, d_z, T)
            x = F.pad(x, (self.short_win - 1, 0))
            x = F.silu(self.ln1(self.conv1(x).transpose(1, 2))).transpose(1, 2)
            x = F.pad(x, (self.short_win - 1, 0))
            h = F.silu(self.ln2(self.conv2(x).transpose(1, 2)))
        else:
            h = self.frame_mlp(e)
        z = self.out_proj(h + e)
        return z, e


class SlowContextEncoder(nn.Module):
    """
    Strictly causal slow context encoder c_tau in R^{d_c}.
    Aggregates past W_ctx ticks of (contact_feat, executed_action) using strictly
    causal rolling mean & std (computed via cumulative sums with zero lookahead).
    """

    def __init__(self, e_dim: int, act_dim: int, d_c: int = 32, window: int = 64):
        super().__init__()
        self.d_c = d_c
        self.window = window
        self.step_net = nn.Sequential(
            nn.Linear(e_dim + act_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(48 * 2, d_c),
            nn.LayerNorm(d_c),
            nn.SiLU(),
            nn.Linear(d_c, d_c),
            nn.LayerNorm(d_c),
        )

    def _causal_rolling_mean_std(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact causal rolling mean & std over window W using float64 cumsum for stability.
        x: (B, T, D) -> mean, std: (B, T, D), strictly causal on [max(0, t-W+1) .. t].
        """
        B, T, D = x.shape
        W = self.window
        x64 = x.double()
        cs = torch.cumsum(x64, dim=1)
        cs2 = torch.cumsum(x64 * x64, dim=1)

        if T > W:
            cs_shift = torch.cat([torch.zeros(B, W, D, device=x.device, dtype=torch.float64), cs[:, :-W, :]], dim=1)
            cs2_shift = torch.cat([torch.zeros(B, W, D, device=x.device, dtype=torch.float64), cs2[:, :-W, :]], dim=1)
            win_sum = cs - cs_shift
            win_sum2 = cs2 - cs2_shift
        else:
            win_sum = cs
            win_sum2 = cs2

        counts = torch.arange(1, T + 1, device=x.device, dtype=torch.float64).clamp(max=W).view(1, T, 1)
        mean = win_sum / counts
        var = (win_sum2 / counts - mean * mean).clamp(min=1e-6)
        std = torch.sqrt(var)
        return mean.to(x.dtype), std.to(x.dtype)

    def forward(self, e_contact: torch.Tensor, act_exec: torch.Tensor) -> torch.Tensor:
        h = self.step_net(torch.cat([e_contact, act_exec], dim=-1))
        m, s = self._causal_rolling_mean_std(h)
        return self.ctx_proj(torch.cat([m, s], dim=-1))


class DecomposedActionEffectPredictor(nn.Module):
    """
    Predicts future latent z_{t+delta} given (z_t, c_t, a_{t:t+delta}, horizon_idx).

    When use_bg_split=True:
        z_hat = z_t + b_phi(z_t, c_t, h) + [g_raw(z_t, c_t, a, h) - g_raw(z_t, c_t, 0, h)]
    guaranteeing g_phi(..., a=0, ...) == 0.0 identically.
    """

    def __init__(
        self,
        d_z: int,
        d_c: int,
        act_dim: int,
        n_horizons: int,
        use_ctx: bool = True,
        use_bg_split: bool = True,
    ):
        super().__init__()
        self.d_z = d_z
        self.d_c = d_c
        self.act_dim = act_dim
        self.use_ctx = use_ctx
        self.use_bg_split = use_bg_split
        self.h_emb = nn.Embedding(n_horizons, 16)

        ctx_in = d_c if use_ctx else 0
        if use_bg_split:
            self.baseline_net = nn.Sequential(
                nn.Linear(d_z + ctx_in + 16, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, d_z),
            )
            self.effect_net = nn.Sequential(
                nn.Linear(d_z + ctx_in + act_dim + 16, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, d_z, bias=False),
            )
        else:
            self.joint_net = nn.Sequential(
                nn.Linear(d_z + ctx_in + act_dim + 16, 160),
                nn.LayerNorm(160),
                nn.SiLU(),
                nn.Linear(160, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, d_z),
            )

    def forward(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor],
        a: torch.Tensor,
        h_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z_hat, b_term, g_term)."""
        shape_prefix = z.shape[:-1]
        h = self.h_emb.weight[h_idx].view(*([1] * len(shape_prefix)), -1).expand(*shape_prefix, -1)
        base_parts = [z, c, h] if (self.use_ctx and c is not None) else [z, h]
        base_in = torch.cat(base_parts, dim=-1)

        if self.use_bg_split:
            b = self.baseline_net(base_in)
            g_a = self.effect_net(torch.cat([base_in, a], dim=-1))
            g_0 = self.effect_net(torch.cat([base_in, torch.zeros_like(a)], dim=-1))
            g = g_a - g_0
            return z + b + g, b, g
        else:
            delta = self.joint_net(torch.cat([base_in, a], dim=-1))
            return z + delta, torch.zeros_like(delta), delta


def vicreg_loss(z: torch.Tensor, var_target: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.08) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Label-free VICReg anti-collapse regularization on (..., D) tensor."""
    x = z.reshape(-1, z.shape[-1])
    N, D = x.shape
    x = x - x.mean(dim=0, keepdim=True)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-4)
    l_var = F.relu(var_target - std).mean()
    cov = (x.T @ x) / max(N - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    l_cov = (off_diag ** 2).sum() / D
    loss = var_weight * l_var + cov_weight * l_cov
    return loss, {
        "std_mean": float(std.mean().item()),
        "std_min": float(std.min().item()),
        "cov_offdiag": float(l_cov.item()),
    }


def compute_effective_rank(z: torch.Tensor) -> float:
    """Effective rank exp(H(p)) of centered representations (max = d_z)."""
    with torch.no_grad():
        x = z.reshape(-1, z.shape[-1]).float()
        x = x - x.mean(dim=0, keepdim=True)
        # Subsample if huge
        if x.shape[0] > 8192:
            idx = torch.linspace(0, x.shape[0] - 1, 8192, device=x.device).long()
            x = x[idx]
        s = torch.linalg.svdvals(x)
        p = (s ** 2) / (torch.sum(s ** 2) + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return float(torch.exp(entropy).item())


class CAECJEPA(nn.Module):
    """
    Context-Factorized Action-Effect Contact JEPA (CAEC-JEPA)
    Supports clean ablations via flags:
      - Vanilla JEPA: use_ctx=False, use_bg_split=False, use_idm=False, use_event_pool=False
      - Full CAEC-JEPA: all True
    """

    def __init__(
        self,
        audio_dim: int,
        act_dim: int = 7,
        d_z: int = 64,
        d_c: int = 32,
        horizons: Tuple[int, ...] = (2, 6, 16),
        ctx_window: int = 64,
        backbone_type: str = "gru",
        use_ctx: bool = True,
        use_bg_split: bool = True,
        use_idm: bool = True,
        use_event_pool: bool = True,
        ema_decay: float = 0.992,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.act_dim = act_dim
        self.d_z = d_z
        self.d_c = d_c
        self.horizons = tuple(horizons)
        self.use_ctx = use_ctx
        self.use_bg_split = use_bg_split
        self.use_idm = use_idm
        self.use_event_pool = use_event_pool
        self.ema_decay = ema_decay

        self.encoder = CausalContactEncoder(audio_dim, d_z=d_z, backbone_type=backbone_type)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        if use_ctx:
            self.ctx_encoder = SlowContextEncoder(d_z, act_dim, d_c=d_c, window=ctx_window)
        else:
            self.ctx_encoder = None

        self.predictor = DecomposedActionEffectPredictor(
            d_z=d_z,
            d_c=d_c,
            act_dim=act_dim,
            n_horizons=len(horizons),
            use_ctx=use_ctx,
            use_bg_split=use_bg_split,
        )

        if use_idm:
            ctx_in = d_c if use_ctx else 0
            self.idm_h_emb = nn.Embedding(len(horizons), 16)
            self.idm_head = nn.Sequential(
                nn.Linear(d_z * 2 + ctx_in + 16, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, act_dim),
            )
        else:
            self.idm_head = None

        # Output token dimension for downstream policy: always d_c + d_z + d_z = 160D when use_ctx
        # (or zeros for CTX when use_ctx=False so downstream policy width is 100% matched!)
        self.token_dim = d_c + d_z + d_z

    @torch.no_grad()
    def update_ema(self):
        d = self.ema_decay
        for p_on, p_tgt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(d).add_(p_on.data, alpha=1.0 - d)

    def encode_ticks(self, aud: torch.Tensor, act_exec: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Strictly causal tick-level encoding.
        aud: (B, T_ticks, audio_dim)
        act_exec: (B, T_ticks, act_dim) — causal executed action up to tick tau.
        """
        z, e = self.encoder(aud)
        c = self.ctx_encoder(e, act_exec) if self.use_ctx else None
        return z, c

    @staticmethod
    def aggregate_horizon_action(act_next: torch.Tensor, delta: int) -> torch.Tensor:
        """
        Cumulative action from tick tau to tau+delta: sum_{k=0}^{delta-1} act_next[:, tau+k].
        act_next: (B, T, act_dim) -> (B, T - delta, act_dim)
        """
        B, T, A = act_next.shape
        cs = torch.cumsum(act_next, dim=1)
        zeros = torch.zeros(B, 1, A, device=act_next.device, dtype=act_next.dtype)
        cs_pad = torch.cat([zeros, cs], dim=1)  # (B, T+1, A)
        return cs_pad[:, delta : T, :] - cs_pad[:, 0 : T - delta, :]

    def compute_loss(
        self,
        aud: torch.Tensor,
        act_exec: torch.Tensor,
        act_next: torch.Tensor,
        contact_weight: Optional[torch.Tensor] = None,
        hold_thresh: float = 0.25,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute full label-free CAEC-JEPA loss on tick sequences (B, T, ...).
        """
        z, c = self.encode_ticks(aud, act_exec)
        with torch.no_grad():
            z_tgt, _ = self.target_encoder(aud)

        pred_losses = []
        idm_losses = []
        b_norms = []
        g_norms = []

        for h_idx, delta in enumerate(self.horizons):
            if aud.shape[1] <= delta + 2:
                continue
            z_cur = z[:, :-delta, :]
            c_cur = c[:, :-delta, :] if c is not None else None
            z_fut = z_tgt[:, delta:, :]
            a_agg = self.aggregate_horizon_action(act_next, delta) / math.sqrt(float(delta))

            z_hat, b_term, g_term = self.predictor(z_cur, c_cur, a_agg, h_idx)
            per_step = F.smooth_l1_loss(z_hat, z_fut, reduction="none").mean(dim=-1)  # (B, T-delta)

            # Weighting: contact-active weighting + hold-anchor boost for low-action frames
            w = torch.ones_like(per_step)
            if contact_weight is not None:
                w = w * contact_weight[:, :-delta]
            if self.use_bg_split:
                a_norm = torch.norm(a_agg, dim=-1)
                is_hold = (a_norm < hold_thresh).float()
                w = w * (1.0 + 1.0 * is_hold)

            pred_losses.append((per_step * w).sum() / (w.sum() + 1e-6))
            b_norms.append(float(b_term.detach().norm(dim=-1).mean().item()))
            g_norms.append(float(g_term.detach().norm(dim=-1).mean().item()))

            if self.use_idm and self.idm_head is not None:
                h_e = self.idm_h_emb.weight[h_idx].view(1, 1, -1).expand(z_cur.shape[0], z_cur.shape[1], -1)
                idm_parts = [z_cur, z_fut, c_cur, h_e] if (c_cur is not None) else [z_cur, z_fut, h_e]
                a_pred = self.idm_head(torch.cat(idm_parts, dim=-1))
                idm_losses.append(F.smooth_l1_loss(a_pred, a_agg))

        l_pred = torch.stack(pred_losses).mean()
        l_idm = torch.stack(idm_losses).mean() if len(idm_losses) > 0 else torch.tensor(0.0, device=aud.device)
        l_vic, vic_info = vicreg_loss(z, var_target=1.0, var_weight=0.5, cov_weight=0.05)

        if self.use_ctx and c is not None and c.shape[1] > 16:
            # Slow consistency between t and t - 12 ticks (strictly within same trajectory)
            c_n = F.normalize(c, dim=-1)
            l_slow = F.mse_loss(c_n[:, 12:, :], c_n[:, :-12, :].detach())
            l_vic_c, _ = vicreg_loss(c, var_target=0.8, var_weight=0.2, cov_weight=0.04)
            l_slow = l_slow + l_vic_c
        else:
            l_slow = torch.tensor(0.0, device=aud.device)

        total = l_pred + 0.35 * l_idm + l_vic + 0.25 * l_slow
        stats = {
            "loss": float(total.item()),
            "l_pred": float(l_pred.item()),
            "l_idm": float(l_idm.item()),
            "l_vic": float(l_vic.item()),
            "l_slow": float(l_slow.item()),
            "b_norm": float(sum(b_norms) / max(len(b_norms), 1)),
            "g_norm": float(sum(g_norms) / max(len(g_norms), 1)),
            **vic_info,
        }
        return total, stats

    @torch.no_grad()
    def compute_tick_surprise(
        self,
        aud: torch.Tensor,
        act_exec: torch.Tensor,
        act_next: torch.Tensor,
        horizon_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Strictly causal tick-level prediction residual r_tau = ||z_tgt[tau] - z_hat[tau]||_2
        where z_hat[tau] was predicted at tau - delta using executed actions a[tau-delta : tau].
        Note: Uses act_exec / past actions ONLY -> 100% causal at tick tau!
        Returns: (z, surprise_raw, c) all of shape (B, T_ticks, ...).
        """
        delta = self.horizons[horizon_idx]
        z, c = self.encode_ticks(aud, act_exec)
        z_tgt, _ = self.target_encoder(aud)
        B, T, _ = z.shape
        surprise = torch.zeros(B, T, device=aud.device, dtype=z.dtype)
        if T > delta:
            # Action executed between (tau - delta) and tau is sum_{k=1}^{delta} act_exec[:, tau - delta + k]
            # which equals aggregate_horizon_action(act_exec[:, 1:], delta) aligned causally!
            cs = torch.cumsum(act_exec, dim=1)
            # sum of act_exec from index (t - delta + 1) to t inclusive:
            # = cs[:, t] - cs[:, t - delta]
            a_past = (cs[:, delta:, :] - cs[:, :-delta, :]) / math.sqrt(float(delta))
            z_prev = z[:, :-delta, :]
            c_prev = c[:, :-delta, :] if c is not None else None
            z_hat, _, _ = self.predictor(z_prev, c_prev, a_past, horizon_idx)
            r = torch.norm(z_tgt[:, delta:, :] - z_hat, dim=-1)
            surprise[:, delta:] = r
        return z, surprise, c

    @torch.no_grad()
    def extract_macro_tokens(
        self,
        aud_ticks: torch.Tensor,
        act_exec_ticks: torch.Tensor,
        substeps: int,
        temp: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress high-rate ticks (B, T_macro * substeps, ...) into 3-Token macro interface:
            tokens: (B, T_macro, d_c + d_z + d_z) = [CTX, STATE, EVENT]
            macro_surprise: (B, T_macro) max surprise within each macro interval.
        Strictly causal at every macro frame t!
        """
        z, surprise, c = self.compute_tick_surprise(aud_ticks, act_exec_ticks, act_exec_ticks, horizon_idx=0)
        B, T_all, _ = z.shape
        T_macro = T_all // substeps
        z_m = z[:, : T_macro * substeps, :].view(B, T_macro, substeps, self.d_z)
        s_m = surprise[:, : T_macro * substeps].view(B, T_macro, substeps)

        state_tok = z_m[:, :, -1, :]  # latest tick in macro frame
        if self.use_ctx and c is not None:
            ctx_tok = c[:, : T_macro * substeps, :].view(B, T_macro, substeps, self.d_c)[:, :, -1, :]
        else:
            ctx_tok = torch.zeros(B, T_macro, self.d_c, device=z.device, dtype=z.dtype)

        if self.use_event_pool:
            # Normalize surprise inside batch/seq gently and attention-pool
            w = torch.softmax(s_m / max(temp, 1e-3), dim=-1).unsqueeze(-1)  # (B, T_macro, substeps, 1)
            event_tok = (w * z_m).sum(dim=2)
        else:
            event_tok = z_m.mean(dim=2)

        tokens = torch.cat([ctx_tok, state_tok, event_tok], dim=-1)
        macro_surprise = s_m.max(dim=-1).values
        return tokens, macro_surprise


class MAEReconContactModel(nn.Module):
    """
    Matched-backbone Masked/Future Spectrogram Reconstruction baseline (B2).
    """

    def __init__(self, audio_dim: int, d_z: int = 64, d_c: int = 32, mask_ratio: float = 0.35):
        super().__init__()
        self.audio_dim = audio_dim
        self.d_z = d_z
        self.d_c = d_c
        self.mask_ratio = mask_ratio
        self.encoder = CausalContactEncoder(audio_dim, d_z=d_z, backbone_type="gru")
        self.mask_token = nn.Parameter(torch.zeros(1, 1, audio_dim))
        self.dec_cur = nn.Sequential(nn.Linear(d_z, 128), nn.SiLU(), nn.Linear(128, audio_dim))
        self.dec_fut = nn.Sequential(nn.Linear(d_z, 128), nn.SiLU(), nn.Linear(128, audio_dim))
        self.ctx_proj = nn.Linear(d_z, d_c)
        self.token_dim = d_c + d_z + d_z

    def compute_loss(self, aud: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T, D = aud.shape
        mask = (torch.rand(B, T, 1, device=aud.device) < self.mask_ratio).float()
        aud_in = aud * (1.0 - mask) + self.mask_token * mask
        z, _ = self.encoder(aud_in)
        rec_cur = self.dec_cur(z)
        l_cur = F.mse_loss(rec_cur, aud)
        rec_fut = self.dec_fut(z[:, :-2, :])
        l_fut = F.mse_loss(rec_fut, aud[:, 2:, :])
        loss = l_cur + l_fut
        return loss, {"loss": float(loss.item()), "l_cur": float(l_cur.item()), "l_fut": float(l_fut.item())}

    @torch.no_grad()
    def extract_macro_tokens(self, aud_ticks: torch.Tensor, substeps: int) -> torch.Tensor:
        z, _ = self.encoder(aud_ticks)
        B, T_all, _ = z.shape
        T_macro = T_all // substeps
        z_m = z[:, : T_macro * substeps, :].view(B, T_macro, substeps, self.d_z)
        state_tok = z_m[:, :, -1, :]
        mean_tok = z_m.mean(dim=2)
        # Causal rolling mean for matched ctx_tok slot
        cs = torch.cumsum(z.double(), dim=1)
        W = 64
        if T_all > W:
            cs_s = torch.cat([torch.zeros(B, W, self.d_z, device=z.device, dtype=torch.float64), cs[:, :-W, :]], dim=1)
            win_sum = cs - cs_s
        else:
            win_sum = cs
        cnt = torch.arange(1, T_all + 1, device=z.device, dtype=torch.float64).clamp(max=W).view(1, T_all, 1)
        ctx_all = self.ctx_proj((win_sum / cnt).to(z.dtype))
        ctx_tok = ctx_all[:, : T_macro * substeps, :].view(B, T_macro, substeps, self.d_c)[:, :, -1, :]
        return torch.cat([ctx_tok, state_tok, mean_tok], dim=-1)


class DownstreamActionChunkPolicy(nn.Module):
    """
    Matched downstream Action-Chunk Imitation Policy (predicts next H macro actions).
    Supports:
      - mode='proprio_only' (B0)
      - mode='frozen_tokens' (B2 MAE, B3 Vanilla JEPA, Ours CAEC-JEPA & ablations)
      - mode='raw_end2end' (B1 end-to-end audio encoder trained jointly by BC loss)
    """

    def __init__(
        self,
        proprio_dim: int,
        token_dim: int,
        act_dim: int = 7,
        chunk_len: int = 8,
        hist_len: int = 4,
        mode: str = "frozen_tokens",
        audio_dim: int = 68,
        substeps: int = 4,
        d_z: int = 64,
        d_c: int = 32,
    ):
        super().__init__()
        self.mode = mode
        self.chunk_len = chunk_len
        self.act_dim = act_dim
        self.hist_len = hist_len
        self.substeps = substeps

        if mode == "raw_end2end":
            self.raw_enc = CausalContactEncoder(audio_dim, d_z=d_z, backbone_type="gru")
            self.raw_ctx = nn.Linear(d_z, d_c)
            eff_tok_dim = d_c + d_z + d_z
        elif mode == "frozen_tokens":
            eff_tok_dim = token_dim
        elif mode == "proprio_only":
            eff_tok_dim = 0
        else:
            raise ValueError(f"Unknown mode={mode}")

        in_dim = proprio_dim * hist_len + eff_tok_dim
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
        contact_tok: Optional[torch.Tensor] = None,
        aud_win: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        prop_hist: (B, hist_len * proprio_dim)
        contact_tok: (B, token_dim) for frozen_tokens mode
        aud_win: (B, win_ticks, audio_dim) for raw_end2end mode
        Returns: (B, chunk_len, act_dim)
        """
        if self.mode == "proprio_only":
            x = prop_hist
        elif self.mode == "frozen_tokens":
            assert contact_tok is not None
            x = torch.cat([prop_hist, contact_tok], dim=-1)
        else:
            assert aud_win is not None
            z, _ = self.raw_enc(aud_win)
            state_tok = z[:, -1, :]
            mean_tok = z[:, -self.substeps :, :].mean(dim=1)
            ctx_tok = self.raw_ctx(z.mean(dim=1))
            x = torch.cat([prop_hist, ctx_tok, state_tok, mean_tok], dim=-1)

        out = self.net(x)
        return out.view(-1, self.chunk_len, self.act_dim)
