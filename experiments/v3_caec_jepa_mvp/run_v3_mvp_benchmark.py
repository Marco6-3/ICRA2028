#!/usr/bin/env python3
"""
V3 CAEC-JEPA MVP Audited Benchmark Suite (ManiWAV / Multi-Rate Contact Audio + Pose)

Executes:
  0. Strict unit audits (prefix causality == 0, zero-action identity == 0, grad flow).
  1. Label-free pretraining of matched models on TRAIN episodes only:
       B2_MAE, B3_VanillaJEPA, Abl_SingleFrame, Abl_ShortWin,
       Abl_NoCTX, Abl_NoBG, Abl_NoIDM, Ours_CAEC_JEPA.
  2. Representation geometry & anti-collapse audit on held-out TEST episodes.
  3. E1 pure-contact frozen probes + non-Markovian reversal & history-matching tests.
  4. Speed-controlled surprise event detection vs. raw audio delta & Vanilla JEPA.
  5. E2 downstream low-shot Action-Chunk BC (10% / 30% / 100% episodes x 3 seeds)
     evaluated on All, Contact-Active, Reversal, and Unseen Spectral-Shift OOD test sets.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from models_caec_jepa import (
    CAECJEPA,
    DownstreamActionChunkPolicy,
    MAEReconContactModel,
    compute_effective_rank,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 0. STRICT UNIT AUDITS
# ==============================================================================
def run_unit_audits(audio_dim: int, act_dim: int, device: torch.device) -> Dict[str, object]:
    print("\n=== [0] Running Strict Unit Audits ===")
    set_seed(123)
    model = CAECJEPA(audio_dim=audio_dim, act_dim=act_dim).to(device)
    model.eval()

    # Audit 1: Prefix Causality (T1=40 vs T2=96)
    B, T1, T2 = 2, 40, 96
    aud = torch.randn(B, T2, audio_dim, device=device)
    act = torch.randn(B, T2, act_dim, device=device)
    with torch.no_grad():
        z_full, s_full, c_full = model.compute_tick_surprise(aud, act, act, horizon_idx=0)
        z_pre, s_pre, c_pre = model.compute_tick_surprise(aud[:, :T1], act[:, :T1], act[:, :T1], horizon_idx=0)
    diff_z = float((z_full[:, :T1] - z_pre).abs().max().item())
    diff_c = float((c_full[:, :T1] - c_pre).abs().max().item())
    diff_s = float((s_full[:, :T1] - s_pre).abs().max().item())
    causal_ok = (diff_z < 1e-5) and (diff_c < 1e-5) and (diff_s < 1e-5)

    # Audit 2: Exact Zero-Action Identity g_phi(..., a=0) == 0.0
    with torch.no_grad():
        _, _, g_zero = model.predictor(z_full[:, :10], c_full[:, :10], torch.zeros(B, 10, act_dim, device=device), 0)
    zero_act_diff = float(g_zero.abs().max().item())
    zero_act_ok = (zero_act_diff == 0.0)

    # Audit 3: Full Gradient Flow across trainable params
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss, _ = model.compute_loss(aud, act, act)
    loss.backward()
    n_trainable, n_nonzero, n_nonfinite = 0, 0, 0
    zero_names = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_trainable += 1
        if p.grad is None or float(p.grad.abs().max().item()) == 0.0:
            zero_names.append(name)
        elif not torch.isfinite(p.grad).all():
            n_nonfinite += 1
        else:
            n_nonzero += 1
    grad_ok = (n_nonzero == n_trainable) and (n_nonfinite == 0)

    audit = {
        "prefix_causality_max_diff_z": diff_z,
        "prefix_causality_max_diff_c": diff_c,
        "prefix_causality_max_diff_surprise": diff_s,
        "prefix_causality_pass": causal_ok,
        "zero_action_g_max_abs": zero_act_diff,
        "zero_action_identity_pass": zero_act_ok,
        "trainable_params": n_trainable,
        "nonzero_grad_params": n_nonzero,
        "nonfinite_grad_params": n_nonfinite,
        "zero_grad_names": zero_names,
        "grad_audit_pass": grad_ok,
        "all_pass": bool(causal_ok and zero_act_ok and grad_ok),
    }
    print(json.dumps(audit, indent=2))
    assert audit["all_pass"], f"Unit audit failed: {audit}"
    return audit


# ==============================================================================
# DATA UTILITIES
# ==============================================================================
class EpisodeCache:
    """Holds episode-sliced normalized tensors strictly separated by split."""

    def __init__(self, cache_path: str):
        raw = np.load(cache_path, allow_pickle=False)
        self.meta = json.loads(str(raw["meta_json"]))
        self.substeps = int(self.meta["substeps"])
        self.audio_dim = int(self.meta["audio_dim"])
        self.n_mels = int(self.meta["n_mels"])
        self.n_mics = len(self.meta["mic_keys"])

        aud = raw["audio_feats"].astype(np.float32)       # (N, S, D_aud)
        rms = raw["audio_rms"].astype(np.float32)         # (N, S)
        prop = raw["proprio_state"].astype(np.float32)    # (N, 10)
        pdelta = raw["proprio_delta"].astype(np.float32)  # (N, 7)
        anext = raw["action_next"].astype(np.float32)     # (N, 7)

        self.aud_mean = raw["aud_mean"].astype(np.float32)
        self.aud_std = raw["aud_std"].astype(np.float32)
        self.prop_mean = raw["prop_mean"].astype(np.float32)
        self.prop_std = raw["prop_std"].astype(np.float32)
        self.act_mean = raw["act_mean"].astype(np.float32)
        self.act_std = raw["act_std"].astype(np.float32)

        # Normalize using strictly TRAIN statistics stored in cache
        aud_n = np.clip((aud - self.aud_mean) / self.aud_std, -8.0, 8.0)
        prop_n = np.clip((prop - self.prop_mean) / self.prop_std, -8.0, 8.0)
        pdelta_n = np.clip((pdelta - self.act_mean) / self.act_std, -8.0, 8.0)
        anext_n = np.clip((anext - self.act_mean) / self.act_std, -8.0, 8.0)

        ep_starts = raw["ep_starts"]
        ep_ends = raw["ep_ends"]
        self.train_eps = raw["train_eps"].tolist()
        self.val_eps = raw["val_eps"].tolist()
        self.test_eps = raw["test_eps"].tolist()

        # Store per-episode dict
        self.episodes: List[Dict[str, np.ndarray]] = []
        for s, e in zip(ep_starts, ep_ends):
            s, e = int(s), int(e)
            L = e - s
            # Expand macro action/delta to sub-steps (divided by substeps so cumulative sum matches macro delta)
            ep_aud_ticks = aud_n[s:e].reshape(L * self.substeps, self.audio_dim)
            ep_rms_ticks = rms[s:e].reshape(L * self.substeps)
            ep_act_exec_ticks = np.repeat(pdelta_n[s:e] / float(self.substeps), self.substeps, axis=0)
            ep_act_next_ticks = np.repeat(anext_n[s:e] / float(self.substeps), self.substeps, axis=0)

            # Causal contact-active weight per tick (based on local RMS flux + speed)
            rms_diff = np.abs(np.diff(ep_rms_ticks, prepend=ep_rms_ticks[0]))
            rms_norm = rms_diff / (np.percentile(rms_diff, 75) + 1e-6)
            energy_norm = ep_rms_ticks / (np.percentile(ep_rms_ticks, 75) + 1e-6)
            cw = np.clip(1.0 + 0.5 * np.minimum(rms_norm, 2.0) + 0.35 * np.minimum(energy_norm, 2.0), 1.0, 3.0).astype(np.float32)

            # Macro speed & reversal & jerk indicators (computed purely from raw kinematics)
            raw_v = anext[s:e, :3]  # local displacement per frame
            raw_speed = np.linalg.norm(raw_v, axis=-1)
            raw_grip_d = np.abs(anext[s:e, 6])
            raw_jerk = np.linalg.norm(np.diff(raw_v, axis=0, prepend=raw_v[:1]), axis=-1)

            # Motion reversal mask (window k=6 frames = 100 ms at 60 Hz): velocity before vs after flips direction
            k = 6
            rev_mask = np.zeros(L, dtype=bool)
            if L > 2 * k + 1:
                for t in range(k, L - k):
                    v_pre = raw_v[t - k : t].mean(axis=0)
                    v_post = raw_v[t : t + k].mean(axis=0)
                    n1, n2 = np.linalg.norm(v_pre), np.linalg.norm(v_post)
                    if n1 > 1e-4 and n2 > 1e-4:
                        cos_v = float(np.dot(v_pre, v_post) / (n1 * n2))
                        if cos_v < -0.25:
                            rev_mask[t] = True

            self.episodes.append(
                {
                    "len": L,
                    "aud_ticks": ep_aud_ticks,              # (L*S, D_aud)
                    "rms_ticks": ep_rms_ticks,              # (L*S,)
                    "act_exec_ticks": ep_act_exec_ticks,    # (L*S, 7)
                    "act_next_ticks": ep_act_next_ticks,    # (L*S, 7)
                    "contact_weight": cw,                   # (L*S,)
                    "prop_macro": prop_n[s:e],              # (L, 10)
                    "pdelta_macro": pdelta_n[s:e],          # (L, 7)
                    "act_macro": anext_n[s:e],              # (L, 7)
                    "raw_speed": raw_speed.astype(np.float32),
                    "raw_grip_d": raw_grip_d.astype(np.float32),
                    "raw_jerk": raw_jerk.astype(np.float32),
                    "rms_macro": rms[s:e].max(axis=-1).astype(np.float32),
                    "rev_mask": rev_mask,
                }
            )

        # Build a deterministic unseen spectral transfer-function perturbation (simulating tool/mic shift)
        # Applies a smooth frequency-dependent EQ tilt + resonant notch + gain shift to log-Mel channels
        per_ch = self.n_mels + 2
        freq_grid = np.linspace(-1.0, 1.0, self.n_mels, dtype=np.float32)
        # Spectral tilt (+0.65 low-to-high slope) + resonant peak/notch + constant gain offset
        eq_shift_1ch = np.zeros(per_ch, dtype=np.float32)
        eq_shift_1ch[: self.n_mels] = (
            0.55 * freq_grid
            + 0.70 * np.exp(-((freq_grid - 0.35) / 0.22) ** 2)
            - 0.65 * np.exp(-((freq_grid + 0.45) / 0.20) ** 2)
            + 0.40
        )
        eq_shift_1ch[self.n_mels] = 0.40      # log_rms gain shift
        eq_shift_1ch[self.n_mels + 1] = 0.15  # flux scale bias
        self.ood_bias = np.tile(eq_shift_1ch, self.n_mics).astype(np.float32)
        # Mild per-band scale modulation (0.85 ~ 1.25)
        scale_1ch = np.ones(per_ch, dtype=np.float32)
        scale_1ch[: self.n_mels] = 1.0 + 0.18 * np.sin(math.pi * freq_grid)
        self.ood_scale = np.tile(scale_1ch, self.n_mics).astype(np.float32)

    def apply_spectral_ood(self, aud_ticks: np.ndarray) -> np.ndarray:
        """Apply unseen spectral transfer-function shift (never seen during training)."""
        return np.clip(aud_ticks * self.ood_scale + self.ood_bias, -8.0, 8.0).astype(np.float32)

    def sample_tick_batch(
        self,
        ep_ids: List[int],
        batch_size: int,
        seq_macro: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        S = self.substeps
        T_ticks = seq_macro * S
        aud_b = np.zeros((batch_size, T_ticks, self.audio_dim), dtype=np.float32)
        aex_b = np.zeros((batch_size, T_ticks, 7), dtype=np.float32)
        anx_b = np.zeros((batch_size, T_ticks, 7), dtype=np.float32)
        cw_b = np.zeros((batch_size, T_ticks), dtype=np.float32)

        valid_eps = [e for e in ep_ids if self.episodes[e]["len"] > seq_macro + 2]
        chosen = rng.choice(valid_eps, size=batch_size, replace=True)
        for i, eid in enumerate(chosen):
            ep = self.episodes[int(eid)]
            max_start = ep["len"] - seq_macro - 1
            st = int(rng.integers(0, max_start + 1))
            ts, te = st * S, (st + seq_macro) * S
            aud_b[i] = ep["aud_ticks"][ts:te]
            aex_b[i] = ep["act_exec_ticks"][ts:te]
            anx_b[i] = ep["act_next_ticks"][ts:te]
            cw_b[i] = ep["contact_weight"][ts:te]

        return (
            torch.from_numpy(aud_b).to(device),
            torch.from_numpy(aex_b).to(device),
            torch.from_numpy(anx_b).to(device),
            torch.from_numpy(cw_b).to(device),
        )


# ==============================================================================
# 1. PRETRAINING
# ==============================================================================
def pretrain_caec_variant(
    name: str,
    cache: EpisodeCache,
    device: torch.device,
    steps: int = 1200,
    batch_size: int = 32,
    seq_macro: int = 40,
    lr: float = 1.2e-3,
    seed: int = 42,
    **model_kwargs,
) -> Tuple[CAECJEPA, Dict[str, float]]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = CAECJEPA(audio_dim=cache.audio_dim, act_dim=7, **model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)

    t0 = time.time()
    model.train()
    last_stats: Dict[str, float] = {}
    for step in range(1, steps + 1):
        aud, aex, anx, cw = cache.sample_tick_batch(cache.train_eps, batch_size, seq_macro, rng, device)
        opt.zero_grad()
        loss, stats = model.compute_loss(aud, aex, anx, contact_weight=cw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        model.update_ema()
        last_stats = stats
        if step % 400 == 0 or step == steps:
            print(
                f"  [{name}] step {step}/{steps} | loss={stats['loss']:.4f} "
                f"pred={stats['l_pred']:.4f} idm={stats['l_idm']:.4f} "
                f"std={stats['std_mean']:.3f} b={stats['b_norm']:.3f} g={stats['g_norm']:.3f}"
            )
    model.eval()
    last_stats["train_sec"] = round(time.time() - t0, 1)
    return model, last_stats


def pretrain_mae(
    cache: EpisodeCache,
    device: torch.device,
    steps: int = 1200,
    batch_size: int = 32,
    seq_macro: int = 40,
    lr: float = 1.2e-3,
    seed: int = 42,
) -> Tuple[MAEReconContactModel, Dict[str, float]]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = MAEReconContactModel(audio_dim=cache.audio_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)

    t0 = time.time()
    model.train()
    last_stats: Dict[str, float] = {}
    for step in range(1, steps + 1):
        aud, _, _, _ = cache.sample_tick_batch(cache.train_eps, batch_size, seq_macro, rng, device)
        opt.zero_grad()
        loss, stats = model.compute_loss(aud)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        last_stats = stats
        if step % 400 == 0 or step == steps:
            print(f"  [B2_MAE] step {step}/{steps} | loss={stats['loss']:.4f} cur={stats['l_cur']:.4f} fut={stats['l_fut']:.4f}")
    model.eval()
    last_stats["train_sec"] = round(time.time() - t0, 1)
    return model, last_stats


# ==============================================================================
# FEATURE EXTRACTION & DIAGNOSTICS (A, B, C)
# ==============================================================================
@torch.no_grad()
def extract_episode_features(
    model,
    cache: EpisodeCache,
    ep_ids: List[int],
    device: torch.device,
    ood_shift: bool = False,
    override_event_pool: Optional[bool] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Run strictly causal full-episode rollout to extract macro tokens, pure-contact z, and surprise."""
    out: Dict[int, Dict[str, np.ndarray]] = {}
    S = cache.substeps
    is_caec = isinstance(model, CAECJEPA)
    orig_epool = getattr(model, "use_event_pool", True)
    if is_caec and override_event_pool is not None:
        model.use_event_pool = override_event_pool

    # Pre-extraction supports optional policy_stride (e.g. 2 -> 30 Hz policy reading 8 ticks per step)
    if is_caec:
        tokens_240_step = S * 2  # pool 8 ticks (33.3 ms) for 30 Hz downstream tokens
    for eid in ep_ids:
        ep = cache.episodes[eid]
        aud_np = cache.apply_spectral_ood(ep["aud_ticks"]) if ood_shift else ep["aud_ticks"]
        aud_t = torch.from_numpy(aud_np).unsqueeze(0).to(device)
        aex_t = torch.from_numpy(ep["act_exec_ticks"]).unsqueeze(0).to(device)
        if is_caec:
            z_ticks, surp_ticks, c_ticks = model.compute_tick_surprise(aud_t, aex_t, aex_t, horizon_idx=0)
            L = ep["len"]
            z_m = z_ticks[0].view(L, S, -1)
            s_m = surp_ticks[0].view(L, S)
            z_macro = z_m[:, -1, :].cpu().numpy()
            macro_surp = s_m.max(dim=-1).values.cpu().numpy()
            # For every macro frame t, pool over causal 33.3 ms window [max(0,(t-1)*S) : (t+1)*S] (8 ticks)
            # Pad left by S copies of tick 0 so frame t=0 is strictly causal, and frame t>=1 covers [(t-1)*S : (t+1)*S]
            z_pad = torch.cat([z_ticks[0, :1].repeat(S, 1), z_ticks[0]], dim=0)  # (L*S + S, d_z)
            s_pad = torch.cat([surp_ticks[0, :1].repeat(S), surp_ticks[0]], dim=0)  # (L*S + S,)
            # Unfold sliding window of length 2*S with stride S -> (L, 2*S, d_z)
            z_win = z_pad.unfold(0, 2 * S, S).transpose(1, 2)  # (L, 2*S, d_z)
            s_win = s_pad.unfold(0, 2 * S, S)                  # (L, 2*S)
            state_tok = z_m[:, -1, :]
            if model.use_ctx and c_ticks is not None:
                ctx_tok = c_ticks[0].view(L, S, -1)[:, -1, :]
            else:
                ctx_tok = torch.zeros(L, model.d_c, device=device, dtype=state_tok.dtype)
            if model.use_event_pool:
                w = torch.softmax(s_win, dim=-1).unsqueeze(-1)
                event_tok = (w * z_win).sum(dim=1)
            else:
                event_tok = z_win.mean(dim=1)
            tokens = torch.cat([ctx_tok, state_tok, event_tok], dim=-1).cpu().numpy()
            out[eid] = {
                "tokens": tokens,
                "z_macro": z_macro,
                "surprise": macro_surp,
            }
        else:
            tokens = model.extract_macro_tokens(aud_t, substeps=S)
            z_ticks, _ = model.encoder(aud_t)
            z_macro = z_ticks[0].view(ep["len"], S, -1)[:, -1, :].cpu().numpy()
            out[eid] = {
                "tokens": tokens[0].cpu().numpy(),
                "z_macro": z_macro,
                "surprise": np.zeros(ep["len"], dtype=np.float32),
            }

    if is_caec:
        model.use_event_pool = orig_epool
    return out


def eval_geometry_and_probes(
    feats_train: Dict[int, Dict[str, np.ndarray]],
    feats_test: Dict[int, Dict[str, np.ndarray]],
    feats_test_ood: Dict[int, Dict[str, np.ndarray]],
    cache: EpisodeCache,
) -> Dict[str, float]:
    """
    Diag A (anti-collapse geometry on TEST) + Diag B (pure-contact z_macro frozen Ridge probes,
    zero proprioception input, including reversal windows & OOD spectral shift).
    """
    Z_tr = np.concatenate([feats_train[e]["z_macro"] for e in cache.train_eps], axis=0)
    Z_te = np.concatenate([feats_test[e]["z_macro"] for e in cache.test_eps], axis=0)
    Z_ood = np.concatenate([feats_test_ood[e]["z_macro"] for e in cache.test_eps], axis=0)

    # Anti-collapse geometry on held-out TEST
    z_t = torch.from_numpy(Z_te)
    eff_rank = compute_effective_rank(z_t)
    _, vic_te = vicreg_loss_np(Z_te)

    # Targets: 4-step cumulative future local displacement & gripper (strictly from kinematics, zero proprio in Z!)
    def build_targets(ep_ids: List[int]):
        ys_vel, ys_grip, revs = [], [], []
        for e in ep_ids:
            ep = cache.episodes[e]
            L = ep["len"]
            act = ep["act_macro"]
            # 4-step cumulative action ahead
            cs = np.vstack([np.zeros((1, 7), dtype=np.float32), np.cumsum(act, axis=0)])
            idx_end = np.minimum(np.arange(L) + 4, L)
            y4 = (cs[idx_end] - cs[:L]) / 2.0
            ys_vel.append(y4[:, :3])
            ys_grip.append(y4[:, 6:7])
            revs.append(ep["rev_mask"])
        return np.concatenate(ys_vel, axis=0), np.concatenate(ys_grip, axis=0), np.concatenate(revs, axis=0)

    Ytr_v, Ytr_g, _ = build_targets(cache.train_eps)
    Yte_v, Yte_g, rev_te = build_targets(cache.test_eps)

    # Ridge on pure-contact z_macro (fit strictly on TRAIN)
    reg_v = Ridge(alpha=10.0).fit(Z_tr, Ytr_v)
    reg_g = Ridge(alpha=10.0).fit(Z_tr, Ytr_g)

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2) + 1e-8
        return float(1.0 - ss_res / ss_tot)

    pv_te = reg_v.predict(Z_te)
    pg_te = reg_g.predict(Z_te)
    pv_ood = reg_v.predict(Z_ood)

    # Reversal subset R^2 and cosine direction accuracy
    if rev_te.sum() >= 20:
        rev_r2 = r2(Yte_v[rev_te], pv_te[rev_te])
        cos_sim = np.sum(Yte_v[rev_te] * pv_te[rev_te], axis=-1) / (
            np.linalg.norm(Yte_v[rev_te], axis=-1) * np.linalg.norm(pv_te[rev_te], axis=-1) + 1e-8
        )
        rev_cos = float(np.mean(cos_sim))
    else:
        rev_r2, rev_cos = float("nan"), float("nan")

    # Also test full 3-Token [CTX, STATE, EVENT] probe on clean & OOD
    T_tr = np.concatenate([feats_train[e]["tokens"] for e in cache.train_eps], axis=0)
    T_te = np.concatenate([feats_test[e]["tokens"] for e in cache.test_eps], axis=0)
    T_ood = np.concatenate([feats_test_ood[e]["tokens"] for e in cache.test_eps], axis=0)
    reg_tok = Ridge(alpha=20.0).fit(T_tr, Ytr_v)
    tok_r2_te = r2(Yte_v, reg_tok.predict(T_te))
    tok_r2_ood = r2(Yte_v, reg_tok.predict(T_ood))

    return {
        "test_eff_rank": round(eff_rank, 2),
        "test_std_mean": round(vic_te["std_mean"], 4),
        "test_cov_offdiag": round(vic_te["cov_offdiag"], 4),
        "pure_z_vel_r2_test": round(r2(Yte_v, pv_te), 4),
        "pure_z_grip_r2_test": round(r2(Yte_g, pg_te), 4),
        "pure_z_vel_r2_ood": round(r2(Yte_v, pv_ood), 4),
        "pure_z_rev_vel_r2": round(rev_r2, 4),
        "pure_z_rev_cos": round(rev_cos, 4),
        "tok3_vel_r2_test": round(tok_r2_te, 4),
        "tok3_vel_r2_ood": round(tok_r2_ood, 4),
    }


def vicreg_loss_np(z: np.ndarray):
    x = z - z.mean(axis=0, keepdims=True)
    std = np.sqrt(x.var(axis=0) + 1e-4)
    cov = (x.T @ x) / max(len(x) - 1, 1)
    off = cov - np.diag(np.diag(cov))
    return 0.0, {"std_mean": float(std.mean()), "cov_offdiag": float((off ** 2).sum() / z.shape[1])}


def eval_codex_history_matching(
    models_feats_test: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    cache: EpisodeCache,
) -> Dict[str, float]:
    """
    Codex-inspired non-Markovian stress test on held-out TEST episodes:
    Find pairs of frames (i, j) whose INSTANTANEOUS single-frame audio feature
    distance is very small (bottom 5%), yet their PAST 4-step motion direction
    is opposite (cos < -0.5) and their FUTURE 4-step motion differs significantly.
    Measure how much each model's latent space z separates these history-divergent pairs
    relative to random matched-history pairs (separation ratio > 1.0 means the latent
    captures non-Markovian history beyond current instantaneous audio!).
    """
    inst_aud, past_v, fut_v = [], [], []
    S = cache.substeps
    for e in cache.test_eps:
        ep = cache.episodes[e]
        L = ep["len"]
        if L < 16:
            continue
        aud_m = ep["aud_ticks"].reshape(L, S, -1)[:, -1, :]
        act = ep["act_macro"][:, :3]
        for t in range(6, L - 6):
            inst_aud.append(aud_m[t])
            past_v.append(act[t - 4 : t].mean(axis=0))
            fut_v.append(act[t : t + 4].mean(axis=0))

    inst_aud = np.asarray(inst_aud, dtype=np.float32)
    past_v = np.asarray(past_v, dtype=np.float32)
    fut_v = np.asarray(fut_v, dtype=np.float32)

    # Subsample up to 1500 frames deterministically for pairwise search
    rng = np.random.default_rng(7)
    N = len(inst_aud)
    sel = np.sort(rng.choice(N, size=min(N, 1500), replace=False))
    A = inst_aud[sel]
    P = past_v[sel]
    Fv = fut_v[sel]
    p_norm = np.linalg.norm(P, axis=-1, keepdims=True)
    active = (p_norm[:, 0] > 0.35)
    sel = sel[active]
    A, P, Fv = A[active], P[active], Fv[active]
    M = len(sel)
    if M < 50:
        return {"n_pairs": 0}

    # Pairwise instantaneous audio distance
    sq = np.sum(A ** 2, axis=-1, keepdims=True)
    d2_aud = np.maximum(sq + sq.T - 2.0 * (A @ A.T), 0.0)
    np.fill_diagonal(d2_aud, np.inf)
    thresh = np.percentile(d2_aud[np.isfinite(d2_aud)], 5.0)

    Pn = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-8)
    cos_p = Pn @ Pn.T

    div_i, div_j = np.where((d2_aud <= thresh) & (cos_p < -0.5))
    same_i, same_j = np.where((d2_aud <= thresh) & (cos_p > 0.7))
    # Keep upper triangle
    mask_d = div_i < div_j
    div_i, div_j = div_i[mask_d], div_j[mask_d]
    mask_s = same_i < same_j
    same_i, same_j = same_i[mask_s], same_j[mask_s]

    res: Dict[str, float] = {
        "n_matched_divergent_pairs": int(len(div_i)),
        "n_matched_same_pairs": int(len(same_i)),
    }
    if len(div_i) < 20 or len(same_i) < 20:
        return res

    # Subsample up to 3000 pairs for fast scoring
    if len(div_i) > 3000:
        idx = rng.choice(len(div_i), 3000, replace=False)
        div_i, div_j = div_i[idx], div_j[idx]
    if len(same_i) > 3000:
        idx = rng.choice(len(same_i), 3000, replace=False)
        same_i, same_j = same_i[idx], same_j[idx]

    # Verify raw future action difference ratio
    f_div = np.linalg.norm(Fv[div_i] - Fv[div_j], axis=-1).mean()
    f_same = np.linalg.norm(Fv[same_i] - Fv[same_j], axis=-1).mean()
    res["future_action_sep_ratio"] = round(float(f_div / (f_same + 1e-8)), 4)

    for mname, fdict in models_feats_test.items():
        zs = []
        for e in cache.test_eps:
            ep = cache.episodes[e]
            L = ep["len"]
            if L < 16:
                continue
            zs.append(fdict[e]["z_macro"][6 : L - 6])
        Z_all = np.concatenate(zs, axis=0)[sel]
        d_div = np.linalg.norm(Z_all[div_i] - Z_all[div_j], axis=-1).mean()
        d_same = np.linalg.norm(Z_all[same_i] - Z_all[same_j], axis=-1).mean()
        res[f"{mname}_sep_ratio"] = round(float(d_div / (d_same + 1e-8)), 4)

    return res


def eval_surprise_events(
    feats_train_caec: Dict[int, Dict[str, np.ndarray]],
    feats_test_caec: Dict[int, Dict[str, np.ndarray]],
    feats_train_van: Dict[int, Dict[str, np.ndarray]],
    feats_test_van: Dict[int, Dict[str, np.ndarray]],
    cache: EpisodeCache,
) -> Dict[str, float]:
    """
    Diag C: Speed-controlled regime-shift / transition event detection on held-out TEST episodes.
    Independent ground-truth transition events = top 10% frames of combined kinematic jerk +
    gripper-change onset (strictly kinematic, independent of audio/model).
    """
    def collect(ep_ids, f_caec, f_van):
        s_caec, s_van, speed, jerk, grip_d, raw_aud_d = [], [], [], [], [], []
        for e in ep_ids:
            ep = cache.episodes[e]
            # Skip first 8 frames of each episode (initial warmup)
            if ep["len"] <= 12:
                continue
            sl = slice(8, ep["len"])
            s_caec.append(f_caec[e]["surprise"][sl])
            s_van.append(f_van[e]["surprise"][sl])
            speed.append(ep["raw_speed"][sl])
            jerk.append(ep["raw_jerk"][sl])
            grip_d.append(ep["raw_grip_d"][sl])
            rms_m = ep["rms_macro"]
            rms_d = np.abs(np.diff(rms_m, prepend=rms_m[0]))
            raw_aud_d.append(rms_d[sl])
        return (
            np.concatenate(s_caec),
            np.concatenate(s_van),
            np.concatenate(speed),
            np.concatenate(jerk),
            np.concatenate(grip_d),
            np.concatenate(raw_aud_d),
        )

    tr_c, tr_v, tr_spd, tr_jrk, tr_gd, tr_ad = collect(cache.train_eps, feats_train_caec, feats_train_van)
    te_c, te_v, te_spd, te_jrk, te_gd, te_ad = collect(cache.test_eps, feats_test_caec, feats_test_van)

    # Independent physical transition label (computed from TRAIN 90th percentile thresholds!)
    j_norm_tr = tr_jrk / (np.std(tr_jrk) + 1e-8)
    g_norm_tr = tr_gd / (np.std(tr_gd) + 1e-8)
    j_norm_te = te_jrk / (np.std(tr_jrk) + 1e-8)
    g_norm_te = te_gd / (np.std(tr_gd) + 1e-8)
    trans_score_tr = j_norm_tr + g_norm_tr
    trans_score_te = j_norm_te + g_norm_te
    thresh = np.percentile(trans_score_tr, 90.0)
    y_event = (trans_score_te >= thresh).astype(np.int32)

    # Fit speed-conditioned z-score bins strictly on TRAIN
    bins = np.quantile(tr_spd, np.linspace(0.0, 1.0, 11))
    bins[0] = -1e9
    bins[-1] = 1e9

    def speed_norm(tr_score, te_score):
        out = np.zeros_like(te_score)
        tr_bin = np.digitize(tr_spd, bins) - 1
        te_bin = np.digitize(te_spd, bins) - 1
        for b in range(10):
            m_tr = tr_bin == b
            m_te = te_bin == b
            if m_tr.sum() >= 5 and m_te.sum() > 0:
                mu = tr_score[m_tr].mean()
                sd = tr_score[m_tr].std() + 1e-6
                out[m_te] = (te_score[m_te] - mu) / sd
            elif m_te.sum() > 0:
                out[m_te] = (te_score[m_te] - tr_score.mean()) / (tr_score.std() + 1e-6)
        return out

    te_c_norm = speed_norm(tr_c, te_c)
    te_v_norm = speed_norm(tr_v, te_v)
    te_ad_norm = speed_norm(tr_ad, te_ad)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "n_test_frames": int(len(y_event)),
        "event_positive_rate": round(float(y_event.mean()), 4),
        "raw_audio_delta_auroc": round(float(roc_auc_score(y_event, te_ad)), 4),
        "raw_audio_delta_speednorm_auroc": round(float(roc_auc_score(y_event, te_ad_norm)), 4),
        "vanilla_jepa_raw_auroc": round(float(roc_auc_score(y_event, te_v)), 4),
        "vanilla_jepa_speednorm_auroc": round(float(roc_auc_score(y_event, te_v_norm)), 4),
        "caec_jepa_raw_auroc": round(float(roc_auc_score(y_event, te_c)), 4),
        "caec_jepa_speednorm_auroc": round(float(roc_auc_score(y_event, te_c_norm)), 4),
        "caec_raw_speed_corr": round(corr(te_c, te_spd), 4),
        "caec_speednorm_speed_corr": round(corr(te_c_norm, te_spd), 4),
        "raw_audio_speed_corr": round(corr(te_ad, te_spd), 4),
    }


# ==============================================================================
# 2. DOWNSTREAM LOW-SHOT ACTION-CHUNK IMITATION LEARNING (E2)
# ==============================================================================
def build_downstream_samples(
    cache: EpisodeCache,
    ep_ids: List[int],
    feats_dict: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    hist_len: int = 4,
    chunk_len: int = 8,
    policy_stride: int = 2,
    aud_win_macro: int = 8,
    ood_shift: bool = False,
    need_aud_win: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Build downstream 30 Hz policy samples (policy_stride=2 on 60 Hz macro frames -> 30 Hz):
    - prop_hist: last hist_len control steps (spaced by policy_stride)
    - action_chunk: next chunk_len control steps (each step = sum of policy_stride macro deltas / sqrt(stride))
    - contact_tok: 3-Token [CTX, STATE, EVENT] covering 2*S = 8 high-rate 240 Hz ticks (33.3 ms)
    """
    prop_list, tok_list, audw_list, act_list, rms_list, rev_list = [], [], [], [], [], []
    S = cache.substeps
    st_p = policy_stride
    for e in ep_ids:
        ep = cache.episodes[e]
        L = ep["len"]
        start_t = max(hist_len * st_p, aud_win_macro)
        end_t = L - chunk_len * st_p
        if end_t <= start_t + 1:
            continue
        p_full = np.concatenate([ep["prop_macro"], ep["pdelta_macro"]], axis=-1)
        act_full = ep["act_macro"]
        cs_act = np.vstack([np.zeros((1, 7), dtype=np.float32), np.cumsum(act_full, axis=0)])
        aud_ticks = cache.apply_spectral_ood(ep["aud_ticks"]) if ood_shift else ep["aud_ticks"]
        toks = feats_dict[e]["tokens"] if feats_dict is not None else None

        for t in range(start_t, end_t, st_p):
            p_idx = [t - (hist_len - 1 - k) * st_p for k in range(hist_len)]
            prop_list.append(p_full[p_idx].reshape(-1))
            # Chunk of chunk_len 30 Hz actions (sum of st_p consecutive 60 Hz deltas)
            chunk = np.zeros((chunk_len, 7), dtype=np.float32)
            for h in range(chunk_len):
                t0 = t + h * st_p
                chunk[h] = (cs_act[t0 + st_p] - cs_act[t0]) / math.sqrt(float(st_p))
            act_list.append(chunk)
            rms_list.append(ep["rms_macro"][max(0, t - st_p + 1) : t + 1].max())
            rev_list.append(bool(ep["rev_mask"][max(0, t - st_p + 1) : t + 1].any()))
            if toks is not None:
                tok_list.append(toks[t])
            elif need_aud_win:
                t_end_tick = (t + 1) * S
                t_start_tick = t_end_tick - aud_win_macro * S
                audw_list.append(aud_ticks[t_start_tick:t_end_tick])

    rms_arr = np.asarray(rms_list, dtype=np.float32)
    out = {
        "prop": np.asarray(prop_list, dtype=np.float32),
        "act": np.asarray(act_list, dtype=np.float32),
        "rms": rms_arr,
        "rev": np.asarray(rev_list, dtype=bool),
    }
    if len(tok_list) > 0:
        out["tok"] = np.asarray(tok_list, dtype=np.float32)
    if len(audw_list) > 0:
        out["aud_win"] = np.asarray(audw_list, dtype=np.float32)
    return out


def train_and_eval_downstream(
    mode: str,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    test_clean: Dict[str, np.ndarray],
    test_ood: Dict[str, np.ndarray],
    contact_thresh: float,
    audio_dim: int,
    substeps: int,
    device: torch.device,
    seed: int = 0,
    epochs: int = 30,
    batch_size: int = 128,
) -> Dict[str, float]:
    set_seed(seed)
    prop_dim = train_data["prop"].shape[-1] // 4
    tok_dim = train_data["tok"].shape[-1] if "tok" in train_data else 160
    policy = DownstreamActionChunkPolicy(
        proprio_dim=prop_dim,
        token_dim=tok_dim,
        act_dim=7,
        chunk_len=8,
        hist_len=4,
        mode=mode,
        audio_dim=audio_dim,
        substeps=substeps,
    ).to(device)

    opt = torch.optim.AdamW(policy.parameters(), lr=1.5e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    N = len(train_data["prop"])
    p_tr = torch.from_numpy(train_data["prop"]).to(device)
    y_tr = torch.from_numpy(train_data["act"]).to(device)
    t_tr = torch.from_numpy(train_data["tok"]).to(device) if mode == "frozen_tokens" else None
    a_tr = torch.from_numpy(train_data["aud_win"]) if mode == "raw_end2end" else None  # keep on CPU if large

    best_val = float("inf")
    best_state = None
    bs = 64 if mode == "raw_end2end" else batch_size

    def eval_split(split_d: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
        policy.eval()
        M = len(split_d["prop"])
        errs = np.zeros(M, dtype=np.float32)
        with torch.no_grad():
            for i in range(0, M, 256):
                pb = torch.from_numpy(split_d["prop"][i : i + 256]).to(device)
                yb = torch.from_numpy(split_d["act"][i : i + 256]).to(device)
                tb = torch.from_numpy(split_d["tok"][i : i + 256]).to(device) if mode == "frozen_tokens" else None
                ab = torch.from_numpy(split_d["aud_win"][i : i + 256]).to(device) if mode == "raw_end2end" else None
                pred = policy(pb, contact_tok=tb, aud_win=ab)
                errs[i : i + 256] = ((pred - yb) ** 2).mean(dim=(1, 2)).cpu().numpy()
        all_mse = float(errs.mean())
        c_mask = split_d["rms"] >= contact_thresh
        c_mse = float(errs[c_mask].mean()) if c_mask.sum() > 0 else all_mse
        r_mask = split_d["rev"]
        r_mse = float(errs[r_mask].mean()) if r_mask.sum() > 0 else all_mse
        return all_mse, c_mse, r_mse

    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        policy.train()
        perm = rng.permutation(N)
        for i in range(0, N, bs):
            idx = perm[i : i + bs]
            idx_t = torch.from_numpy(idx).long().to(device)
            pb = p_tr[idx_t]
            yb = y_tr[idx_t]
            tb = t_tr[idx_t] if t_tr is not None else None
            ab = a_tr[idx].to(device) if a_tr is not None else None
            opt.zero_grad()
            pred = policy(pb, contact_tok=tb, aud_win=ab)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
            opt.step()
        sched.step()
        val_mse, _, _ = eval_split(val_data)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    policy.load_state_dict(best_state)
    all_c, con_c, rev_c = eval_split(test_clean)
    all_o, con_o, _ = eval_split(test_ood)
    return {
        "val_mse": round(best_val, 5),
        "test_all_mse": round(all_c, 5),
        "test_contact_mse": round(con_c, 5),
        "test_reversal_mse": round(rev_c, 5),
        "test_ood_all_mse": round(all_o, 5),
        "test_ood_contact_mse": round(con_o, 5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--pretrain-steps", type=int, default=1200)
    parser.add_argument("--bc-epochs", type=int, default=28)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    cache = EpisodeCache(args.cache)
    print(f"Loaded cache: {cache.meta}")

    # 0. Strict Unit Audits
    unit_audits = run_unit_audits(cache.audio_dim, 7, device)

    # 1. Pretrain all matched representation models on TRAIN episodes only
    print("\n=== [1] Pretraining Matched Representation Models ===")
    pretrain_stats = {}
    models = {}

    models["B2_MAE"], pretrain_stats["B2_MAE"] = pretrain_mae(cache, device, steps=args.pretrain_steps, seed=42)

    base_hz = dict(horizons=(4, 12, 32), ctx_window=96)
    caec_specs = {
        "B3_VanillaJEPA": dict(**base_hz, backbone_type="gru", use_ctx=False, use_bg_split=False, use_idm=False, use_event_pool=False),
        "Abl_SingleFrame": dict(**base_hz, backbone_type="single_frame", use_ctx=True, use_bg_split=True, use_idm=True, use_event_pool=True),
        "Abl_ShortWin": dict(**base_hz, backbone_type="short_win", use_ctx=True, use_bg_split=True, use_idm=True, use_event_pool=True),
        "Abl_NoCTX": dict(**base_hz, backbone_type="gru", use_ctx=False, use_bg_split=True, use_idm=True, use_event_pool=True),
        "Abl_NoBG": dict(**base_hz, backbone_type="gru", use_ctx=True, use_bg_split=False, use_idm=True, use_event_pool=True),
        "Abl_NoIDM": dict(**base_hz, backbone_type="gru", use_ctx=True, use_bg_split=True, use_idm=False, use_event_pool=True),
        "Ours_CAEC_JEPA": dict(**base_hz, backbone_type="gru", use_ctx=True, use_bg_split=True, use_idm=True, use_event_pool=True),
    }
    for mname, kw in caec_specs.items():
        models[mname], pretrain_stats[mname] = pretrain_caec_variant(
            mname, cache, device, steps=args.pretrain_steps, seed=42, **kw
        )

    # 2. Extract frozen episode features (train / val / test / test_ood)
    print("\n=== [2] Extracting Frozen Episode Features & Running Geometry/Probe Diagnostics ===")
    feats_tr, feats_va, feats_te, feats_te_ood = {}, {}, {}, {}
    probe_results = {}
    for mname, mdl in models.items():
        feats_tr[mname] = extract_episode_features(mdl, cache, cache.train_eps, device, ood_shift=False)
        feats_va[mname] = extract_episode_features(mdl, cache, cache.val_eps, device, ood_shift=False)
        feats_te[mname] = extract_episode_features(mdl, cache, cache.test_eps, device, ood_shift=False)
        feats_te_ood[mname] = extract_episode_features(mdl, cache, cache.test_eps, device, ood_shift=True)
        probe_results[mname] = eval_geometry_and_probes(feats_tr[mname], feats_te[mname], feats_te_ood[mname], cache)
        print(f"  [Probe] {mname:16s} | {probe_results[mname]}")

    # Also create Ours_NoEventPool (same weights, uniform mean pooling instead of surprise attention)
    feats_tr["Abl_NoEventPool"] = extract_episode_features(models["Ours_CAEC_JEPA"], cache, cache.train_eps, device, False, False)
    feats_va["Abl_NoEventPool"] = extract_episode_features(models["Ours_CAEC_JEPA"], cache, cache.val_eps, device, False, False)
    feats_te["Abl_NoEventPool"] = extract_episode_features(models["Ours_CAEC_JEPA"], cache, cache.test_eps, device, False, False)
    feats_te_ood["Abl_NoEventPool"] = extract_episode_features(models["Ours_CAEC_JEPA"], cache, cache.test_eps, device, True, False)

    # 3. Codex-inspired Non-Markovian History Matching Stress Test
    print("\n=== [3] Codex-Inspired Matched-Audio / Divergent-History Stress Test ===")
    history_test = eval_codex_history_matching(feats_te, cache)
    print(json.dumps(history_test, indent=2))

    # 4. Speed-Controlled Surprise Event Detection
    print("\n=== [4] Speed-Controlled Surprise Event Detection ===")
    surprise_test = eval_surprise_events(
        feats_tr["Ours_CAEC_JEPA"],
        feats_te["Ours_CAEC_JEPA"],
        feats_tr["B3_VanillaJEPA"],
        feats_te["B3_VanillaJEPA"],
        cache,
    )
    print(json.dumps(surprise_test, indent=2))

    # 5. E2 Downstream Low-Shot Action-Chunk BC Benchmark (10% / 30% / 100% x 3 seeds)
    print("\n=== [5] E2 Downstream Low-Shot Action-Chunk BC (10% / 30% / 100% x 3 seeds) ===")
    # Compute contact-active RMS threshold strictly on TRAIN episodes (75th percentile)
    tr_rms_all = np.concatenate([cache.episodes[e]["rms_macro"] for e in cache.train_eps])
    contact_thresh = float(np.percentile(tr_rms_all, 75.0))

    bc_methods = [
        ("B0_ProprioOnly", "proprio_only", None),
        ("B1_RawEnd2End", "raw_end2end", None),
        ("B2_MAE", "frozen_tokens", "B2_MAE"),
        ("B3_VanillaJEPA", "frozen_tokens", "B3_VanillaJEPA"),
        ("Abl_SingleFrame", "frozen_tokens", "Abl_SingleFrame"),
        ("Abl_NoCTX", "frozen_tokens", "Abl_NoCTX"),
        ("Abl_NoBG", "frozen_tokens", "Abl_NoBG"),
        ("Abl_NoIDM", "frozen_tokens", "Abl_NoIDM"),
        ("Abl_NoEventPool", "frozen_tokens", "Abl_NoEventPool"),
        ("Ours_CAEC_JEPA", "frozen_tokens", "Ours_CAEC_JEPA"),
    ]
    fractions = [0.10, 0.30, 1.00]
    downstream_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for frac in fractions:
        fkey = f"{int(round(frac * 100))}pct"
        downstream_results[fkey] = {}
        n_sub = max(3, int(round(len(cache.train_eps) * frac)))
        print(f"\n--- Budget: {fkey} ({n_sub}/{len(cache.train_eps)} train episodes) ---")

        for label, mode, mkey in bc_methods:
            seed_runs = []
            need_aw = (mode == "raw_end2end")
            # Pre-build val & test splits once per method
            va_d = build_downstream_samples(cache, cache.val_eps, feats_va[mkey] if mkey else None, ood_shift=False, need_aud_win=need_aw)
            te_d = build_downstream_samples(cache, cache.test_eps, feats_te[mkey] if mkey else None, ood_shift=False, need_aud_win=need_aw)
            te_ood_d = build_downstream_samples(cache, cache.test_eps, feats_te_ood[mkey] if mkey else None, ood_shift=True, need_aud_win=need_aw)

            for sd in args.seeds:
                rng_sub = np.random.default_rng(1000 + sd)
                sub_eps = sorted(rng_sub.choice(cache.train_eps, size=n_sub, replace=False).tolist())
                tr_d = build_downstream_samples(cache, sub_eps, feats_tr[mkey] if mkey else None, ood_shift=False, need_aud_win=need_aw)
                r = train_and_eval_downstream(
                    mode=mode,
                    train_data=tr_d,
                    val_data=va_d,
                    test_clean=te_d,
                    test_ood=te_ood_d,
                    contact_thresh=contact_thresh,
                    audio_dim=cache.audio_dim,
                    substeps=cache.substeps,
                    device=device,
                    seed=sd,
                    epochs=args.bc_epochs,
                )
                seed_runs.append(r)

            agg = {}
            for k in seed_runs[0].keys():
                vals = [x[k] for x in seed_runs]
                agg[f"{k}_mean"] = round(float(np.mean(vals)), 5)
                agg[f"{k}_std"] = round(float(np.std(vals)), 5)
            downstream_results[fkey][label] = agg
            print(
                f"  [{fkey}] {label:16s} | All={agg['test_all_mse_mean']:.5f}±{agg['test_all_mse_std']:.5f} "
                f"| Contact={agg['test_contact_mse_mean']:.5f} | Rev={agg['test_reversal_mse_mean']:.5f} "
                f"| OOD={agg['test_ood_all_mse_mean']:.5f}"
            )

    summary = {
        "meta": cache.meta,
        "unit_audits": unit_audits,
        "pretrain_stats": pretrain_stats,
        "probe_results": probe_results,
        "history_matching_test": history_test,
        "surprise_event_test": surprise_test,
        "downstream_lowshot_bc": downstream_results,
    }
    json_path = out_dir / "v3_mvp_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full JSON results to {json_path}")


if __name__ == "__main__":
    main()
