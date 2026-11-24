#!/usr/bin/env python3
"""
V4.1 Route 1 Audited Benchmark: Physics-Anchored Action-Effect Contact Compression (PAEC)
on real 6D Force/Torque + GelSight Tactile + 6-DOF Pose datasets (RDP peel / wipe / lift).

V4.1 Protocol Upgrades (over archived V4.0):
  - Physical zero-action alignment: zero-centered RMS/SI action scaling (delta_q == 0 <=> a_norm == 0)
    fixing g_phi(z, 0), is_hold, virtual probe axes, and Codex divergent-history velocity direction.
  - Fair matched baselines: B2b_MAE_DriftAug, B3_LatentJEPA_d16, B3b_LatentJEPA_d16_DriftAug,
    and B1b_RawWrench6D_End2End (to isolate 6D wrench vs 30D tactile noise).
  - Supports --contact-subset {all, wrench6d}, --con-norm-mode {zscore, si_fixed},
    --act-norm-mode {rms, si_fixed, zscore_legacy}, and scaling knobs (--hidden-dim, --num-layers, --d-z).
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models_paec import (
    AbstractLatentJEPAVICReg,
    DownstreamActionChunkPolicy,
    MAEContactRecon,
    PAECModel,
    build_causal_contact_input,
    compute_effective_rank,
)

# Canonical fixed SI physical scaling constants (for cross-dataset / cross-embodiment transfer)
# Force: 10 N, Torque: 0.5 N*m, Gripper force: 10 N
# Pos delta: 5 mm, Rot delta: 0.05 rad, Grip delta: 5 mm
SI_WRENCH_SCALE = np.array([10.0, 10.0, 10.0, 0.5, 0.5, 0.5], dtype=np.float32)
SI_GRIP_FORCE_SCALE = 10.0
SI_ACT_SCALE = np.array([0.005, 0.005, 0.005, 0.05, 0.05, 0.05, 0.005], dtype=np.float32)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


# ==============================================================================
# 0. STRICT UNIT AUDITS
# ==============================================================================
def run_unit_audits(
    contact_dim: int,
    wrench_dim: int,
    act_dim: int,
    device: torch.device,
    d_z: int = 16,
    hidden_dim: int = 96,
    num_layers: int = 2,
    cache: Optional["RDPEpisodeCache"] = None,
) -> Dict[str, object]:
    print("\n=== [0] Running Strict Unit Audits (PAEC V4.1) ===")
    set_seed(123)
    model = PAECModel(
        contact_dim=contact_dim,
        wrench_dim=wrench_dim,
        act_dim=act_dim,
        d_z=d_z,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        drift_aug_std=0.0,
    ).to(device)
    model.eval()

    B, T1, T2 = 2, 20, 48
    obs = torch.randn(B, T2, contact_dim, device=device)
    act = torch.randn(B, T2, act_dim, device=device)
    with torch.no_grad():
        z_f, p_f, s_f = model.compute_causal_surprise(obs, act, horizon_idx=0)
        z_p, p_p, s_p = model.compute_causal_surprise(obs[:, :T1], act[:, :T1], horizon_idx=0)
    dz = float((z_f[:, :T1] - z_p).abs().max().item())
    dp = float((p_f[:, :T1] - p_p).abs().max().item())
    ds = float((s_f[:, :T1] - s_p).abs().max().item())
    causal_ok = (dz < 1e-5) and (dp < 1e-5) and (ds < 1e-5)

    with torch.no_grad():
        zero_act = torch.zeros(B, 10, act_dim, device=device)
        _, _, g0 = model.predictor(z_f[:, :10], zero_act, 0)
        agg_zero_max = max(
            float(PAECModel.aggregate_action(torch.zeros(B, T2, act_dim, device=device), d).abs().max().item())
            for d in model.horizons
        )
    zact_diff = float(g0.abs().max().item())
    zact_ok = (zact_diff == 0.0) and (agg_zero_max == 0.0)

    # Physical zero-point alignment check on cache normalization (V4.1)
    phys_zero_err = 0.0
    phys_zero_ok = True
    if cache is not None and cache.act_norm_mode != "zscore_legacy":
        a0_norm = cache.normalize_action(np.zeros((1, act_dim), dtype=np.float32))
        phys_zero_err = float(np.abs(a0_norm).max())
        phys_zero_ok = (phys_zero_err == 0.0)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss, _ = model.compute_loss(obs, act, act)
    loss.backward()
    nt, nz, nf = 0, 0, 0
    znames = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nt += 1
        if p.grad is None or float(p.grad.abs().max().item()) == 0.0:
            znames.append(name)
        elif not torch.isfinite(p.grad).all():
            nf += 1
        else:
            nz += 1
    grad_ok = (nz == nt) and (nf == 0)

    audit = {
        "prefix_causality_max_diff_z": dz,
        "prefix_causality_max_diff_probe": dp,
        "prefix_causality_max_diff_surprise": ds,
        "prefix_causality_pass": causal_ok,
        "zero_action_g_max_abs": zact_diff,
        "zero_action_agg_max_abs": agg_zero_max,
        "zero_action_identity_pass": zact_ok,
        "physical_zero_action_norm_err": phys_zero_err,
        "physical_zero_action_pass": phys_zero_ok,
        "trainable_params": nt,
        "nonzero_grad_params": nz,
        "nonfinite_grad_params": nf,
        "zero_grad_names": znames,
        "grad_audit_pass": grad_ok,
        "all_pass": bool(causal_ok and zact_ok and phys_zero_ok and grad_ok),
    }
    print(json.dumps(audit, indent=2))
    assert audit["all_pass"], f"Unit audit failed: {audit}"
    return audit


# ==============================================================================
# DATA LOADER
# ==============================================================================
class RDPEpisodeCache:
    def __init__(
        self,
        cache_path: str,
        act_norm_mode: str = "rms",
        con_norm_mode: str = "zscore",
        contact_subset: str = "all",
    ):
        assert act_norm_mode in ("rms", "si_fixed", "zscore_legacy"), f"Unknown act_norm_mode={act_norm_mode}"
        assert con_norm_mode in ("zscore", "si_fixed"), f"Unknown con_norm_mode={con_norm_mode}"
        assert contact_subset in ("all", "wrench6d"), f"Unknown contact_subset={contact_subset}"
        self.act_norm_mode = act_norm_mode
        self.con_norm_mode = con_norm_mode
        self.contact_subset = contact_subset

        raw = np.load(cache_path, allow_pickle=False)
        self.meta = json.loads(str(raw["meta_json"]))
        self.wrench_dim = int(self.meta["wrench_dim"])
        self.proprio_dim = int(self.meta["proprio_dim"])
        self.act_dim = int(self.meta["act_dim"])

        con = raw["contact_obs"].astype(np.float32)
        wraw = raw["wrench_raw"].astype(np.float32)
        prop = raw["proprio_state"].astype(np.float32)
        pdelta = raw["proprio_delta"].astype(np.float32)
        anext = raw["action_next"].astype(np.float32)

        con_mean_raw = raw["con_mean"].astype(np.float32)
        con_std_raw = raw["con_std"].astype(np.float32)
        if contact_subset == "wrench6d":
            con = con[:, : self.wrench_dim]
            con_mean_raw = con_mean_raw[: self.wrench_dim]
            con_std_raw = con_std_raw[: self.wrench_dim]
        self.contact_dim = int(con.shape[1])

        self.prop_mean = raw["prop_mean"].astype(np.float32)
        self.prop_std = raw["prop_std"].astype(np.float32)
        self.act_mean = raw["act_mean"].astype(np.float32)
        self.act_std = raw["act_std"].astype(np.float32)

        ep_starts = raw["ep_starts"]
        ep_ends = raw["ep_ends"]
        self.train_eps = raw["train_eps"].tolist()
        self.val_eps = raw["val_eps"].tolist()
        self.test_eps = raw["test_eps"].tolist()

        # Train-only RMS for zero-centered physical action scaling (V4.1 fix)
        train_mask = np.zeros(len(anext), dtype=bool)
        for eid in self.train_eps:
            train_mask[int(ep_starts[eid]) : int(ep_ends[eid])] = True
        self.act_rms = np.maximum(np.sqrt(np.mean(anext[train_mask] ** 2, axis=0)), 1e-4).astype(np.float32)

        if con_norm_mode == "zscore":
            self.con_mean = con_mean_raw
            self.con_std = con_std_raw
        else:
            # Fixed SI physical units (zero-centered) to preserve N/m stiffness across datasets
            self.con_mean = np.zeros(self.contact_dim, dtype=np.float32)
            sc = con_std_raw.copy()
            w_n = min(self.wrench_dim, self.contact_dim, len(SI_WRENCH_SCALE))
            sc[:w_n] = SI_WRENCH_SCALE[:w_n]
            if self.contact_dim > self.wrench_dim and int(self.meta.get("grip_force_dim", 0)) > 0:
                sc[self.wrench_dim] = SI_GRIP_FORCE_SCALE
            self.con_std = sc.astype(np.float32)

        if act_norm_mode == "rms":
            self.act_center = np.zeros(self.act_dim, dtype=np.float32)
            self.act_scale = self.act_rms
        elif act_norm_mode == "si_fixed":
            self.act_center = np.zeros(self.act_dim, dtype=np.float32)
            sc_a = self.act_rms.copy()
            a_n = min(self.act_dim, len(SI_ACT_SCALE))
            sc_a[:a_n] = SI_ACT_SCALE[:a_n]
            self.act_scale = sc_a.astype(np.float32)
        else:
            self.act_center = self.act_mean
            self.act_scale = self.act_std

        con_n = self.normalize_contact(con)
        prop_n = np.clip((prop - self.prop_mean) / self.prop_std, -8.0, 8.0)
        pdelta_n = self.normalize_action(pdelta)
        anext_n = self.normalize_action(anext)

        # Deterministic unseen zero-drift bias vector (+1.5 sigma alternating signs)
        rng_ood = np.random.default_rng(2028)
        signs = rng_ood.choice([-1.0, 1.0], size=self.contact_dim).astype(np.float32)
        self.drift_bias = (1.5 * signs).astype(np.float32)

        self.episodes: List[Dict[str, np.ndarray]] = []
        for s, e in zip(ep_starts, ep_ends):
            s, e = int(s), int(e)
            L = e - s
            ep_w = wraw[s:e]
            ep_w_norm = np.linalg.norm(ep_w[:, :3], axis=-1)
            ep_dw_norm = np.linalg.norm(np.diff(ep_w, axis=0, prepend=ep_w[:1]), axis=-1)
            cw = np.clip(
                1.0 + 0.5 * (ep_w_norm / (np.percentile(ep_w_norm, 75) + 1e-5))
                + 0.5 * (ep_dw_norm / (np.percentile(ep_dw_norm, 75) + 1e-5)),
                1.0,
                3.5,
            ).astype(np.float32)

            raw_v = anext[s:e, :3]
            raw_speed = np.linalg.norm(raw_v, axis=-1).astype(np.float32)
            # Reversal mask (k=4 frames) in physical raw velocity coordinates
            k = 4
            rev_mask = np.zeros(L, dtype=bool)
            if L > 2 * k + 1:
                for t in range(k, L - k):
                    v1 = raw_v[t - k : t].mean(axis=0)
                    v2 = raw_v[t : t + k].mean(axis=0)
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-5 and n2 > 1e-5 and float(np.dot(v1, v2) / (n1 * n2)) < -0.25:
                        rev_mask[t] = True

            self.episodes.append(
                {
                    "len": L,
                    "con": con_n[s:e],
                    "wraw": ep_w,
                    "w_norm": ep_w_norm.astype(np.float32),
                    "dw_norm": ep_dw_norm.astype(np.float32),
                    "prop": prop_n[s:e],
                    "pdelta": pdelta_n[s:e],
                    "act": anext_n[s:e],
                    "act_raw": anext[s:e],
                    "cw": cw,
                    "speed": raw_speed,
                    "rev_mask": rev_mask,
                }
            )

    def normalize_contact(self, con: np.ndarray) -> np.ndarray:
        return np.clip((con - self.con_mean) / self.con_std, -8.0, 8.0).astype(np.float32)

    def normalize_action(self, act: np.ndarray) -> np.ndarray:
        return np.clip((act - self.act_center) / self.act_scale, -8.0, 8.0).astype(np.float32)

    def apply_ood(self, con: np.ndarray, ood_mode: str) -> np.ndarray:
        if ood_mode == "clean":
            return con
        elif ood_mode == "drift":
            return np.clip(con + self.drift_bias, -8.0, 8.0).astype(np.float32)
        elif ood_mode == "gain":
            return np.clip(con * 1.6, -8.0, 8.0).astype(np.float32)
        else:
            raise ValueError(ood_mode)

    def sample_batch(
        self, ep_ids: List[int], batch_size: int, seq_len: int, rng: np.random.Generator, device: torch.device
    ):
        valid = [e for e in ep_ids if self.episodes[e]["len"] > seq_len + 2]
        if len(valid) == 0:
            # Clamp seq_len to shortest episode if mini dataset has short episodes
            seq_len = min(self.episodes[e]["len"] for e in ep_ids) - 2
            valid = list(ep_ids)
        chosen = rng.choice(valid, size=batch_size, replace=True)
        con_b = np.zeros((batch_size, seq_len, self.contact_dim), dtype=np.float32)
        aex_b = np.zeros((batch_size, seq_len, self.act_dim), dtype=np.float32)
        anx_b = np.zeros((batch_size, seq_len, self.act_dim), dtype=np.float32)
        cw_b = np.zeros((batch_size, seq_len), dtype=np.float32)
        for i, eid in enumerate(chosen):
            ep = self.episodes[int(eid)]
            st = int(rng.integers(0, ep["len"] - seq_len))
            con_b[i] = ep["con"][st : st + seq_len]
            aex_b[i] = ep["pdelta"][st : st + seq_len]
            anx_b[i] = ep["act"][st : st + seq_len]
            cw_b[i] = ep["cw"][st : st + seq_len]
        return (
            torch.from_numpy(con_b).to(device),
            torch.from_numpy(aex_b).to(device),
            torch.from_numpy(anx_b).to(device),
            torch.from_numpy(cw_b).to(device),
        )


# ==============================================================================
# 1. PRETRAINING
# ==============================================================================
def pretrain_model(
    name: str,
    model: nn.Module,
    cache: RDPEpisodeCache,
    device: torch.device,
    steps: int = 1200,
    batch_size: int = 32,
    seq_len: int = 48,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> Dict[str, float]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    t0 = time.time()
    model.train()
    last = {}
    for step in range(1, steps + 1):
        con, aex, anx, cw = cache.sample_batch(cache.train_eps, batch_size, seq_len, rng, device)
        opt.zero_grad()
        if isinstance(model, PAECModel):
            loss, stats = model.compute_loss(con, aex, anx, contact_weight=cw)
        elif isinstance(model, MAEContactRecon):
            loss, stats = model.compute_loss(con, aex)
        elif isinstance(model, AbstractLatentJEPAVICReg):
            loss, stats = model.compute_loss(con, aex, anx)
        else:
            raise TypeError(type(model))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        if isinstance(model, AbstractLatentJEPAVICReg):
            model.update_ema()
        last = stats
        if step % 400 == 0 or step == steps:
            print(f"  [{name:16s}] step {step}/{steps} | " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()))
    model.eval()
    last["train_sec"] = round(time.time() - t0, 1)
    return last


# ==============================================================================
# 2. DIAGNOSTICS: FORWARD MECHANICS, GEOMETRY, CODEX HISTORY, SURPRISE
# ==============================================================================
@torch.no_grad()
def extract_features(model: nn.Module, cache: RDPEpisodeCache, ep_ids: List[int], device: torch.device, ood_mode: str = "clean"):
    out = {}
    for eid in ep_ids:
        ep = cache.episodes[eid]
        con_np = cache.apply_ood(ep["con"], ood_mode)
        con_t = torch.from_numpy(con_np).unsqueeze(0).to(device)
        aex_t = torch.from_numpy(ep["pdelta"]).unsqueeze(0).to(device)
        if isinstance(model, PAECModel):
            z, probe, surp = model.compute_causal_surprise(con_t, aex_t, horizon_idx=0)
            z_np = z[0].cpu().numpy()
            p_np = probe[0].cpu().numpy()
            s_np = surp[0].cpu().numpy()
            out[eid] = {
                "z": z_np,
                "z_probe": np.concatenate([z_np, p_np, s_np[:, None]], axis=-1).astype(np.float32),
                "surprise": s_np,
            }
        else:
            z_np = model.encode(con_t, aex_t)[0].cpu().numpy()
            out[eid] = {
                "z": z_np,
                "z_probe": z_np,
                "surprise": np.zeros(ep["len"], dtype=np.float32),
            }
    return out


@torch.no_grad()
def eval_forward_mechanics(model: PAECModel, cache: RDPEpisodeCache, device: torch.device) -> Dict[str, float]:
    """Direct 6D Wrench Delta prediction R^2 on held-out TEST episodes (Clean, Reversal, Drift OOD)."""
    res = {}
    for ood_mode in ["clean", "drift"]:
        trues, preds, revs = [], [], []
        for eid in cache.test_eps:
            ep = cache.episodes[eid]
            L = ep["len"]
            delta = model.horizons[1] if len(model.horizons) > 1 else model.horizons[0]  # 3-step horizon
            h_idx = 1 if len(model.horizons) > 1 else 0
            if L <= delta + 2:
                continue
            con_in = cache.apply_ood(ep["con"], ood_mode)
            con_t = torch.from_numpy(con_in).unsqueeze(0).to(device)
            aex_t = torch.from_numpy(ep["pdelta"]).unsqueeze(0).to(device)
            anx_t = torch.from_numpy(ep["act"]).unsqueeze(0).to(device)
            z = model.encode(con_t, aex_t)
            a_agg = PAECModel.aggregate_action(anx_t, delta)
            dy_hat, _, _ = model.predictor(z[:, :-delta, :], a_agg, h_idx)
            # True physical delta is invariant to constant drift
            dw_true = ep["con"][delta:, : cache.wrench_dim] - ep["con"][:-delta, : cache.wrench_dim]
            dw_hat = dy_hat[0, :, : cache.wrench_dim].cpu().numpy()
            trues.append(dw_true)
            preds.append(dw_hat)
            revs.append(ep["rev_mask"][:-delta])

        Y = np.concatenate(trues, axis=0)
        P = np.concatenate(preds, axis=0)
        R = np.concatenate(revs, axis=0)
        mse = float(np.mean((Y - P) ** 2))
        r2 = float(1.0 - np.sum((Y - P) ** 2) / (np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2) + 1e-8))
        res[f"dw3_{ood_mode}_mse"] = round(mse, 4)
        res[f"dw3_{ood_mode}_r2"] = round(r2, 4)
        if ood_mode == "clean" and R.sum() >= 15:
            r2_rev = float(1.0 - np.sum((Y[R] - P[R]) ** 2) / (np.sum((Y[R] - Y[R].mean(axis=0, keepdims=True)) ** 2) + 1e-8))
            res["dw3_rev_r2"] = round(r2_rev, 4)
    return res


def eval_codex_wrench_matching(models_feats_te: Dict[str, dict], cache: RDPEpisodeCache) -> Dict[str, float]:
    """
    Codex-inspired non-Markovian stress test:
    Find TEST frame pairs with nearly identical INSTANTANEOUS 6D wrench (bottom 5% distance),
    yet opposite PAST 4-step motion direction (cos < -0.5). Verify whether latent z separates
    divergent-history pairs better than same-history pairs.
    """
    inst_w, past_v, fut_dw = [], [], []
    for e in cache.test_eps:
        ep = cache.episodes[e]
        L = ep["len"]
        if L < 14:
            continue
        for t in range(5, L - 5):
            inst_w.append(ep["con"][t, : cache.wrench_dim])
            # Use physical zero-centered velocity scaled by act_rms (V4.1 fix: no mean subtraction!)
            past_v.append((ep["act_raw"][t - 4 : t, :3] / cache.act_rms[:3]).mean(axis=0))
            fut_dw.append(ep["con"][t + 3, : cache.wrench_dim] - ep["con"][t, : cache.wrench_dim])

    inst_w = np.asarray(inst_w, dtype=np.float32)
    past_v = np.asarray(past_v, dtype=np.float32)
    fut_dw = np.asarray(fut_dw, dtype=np.float32)
    rng = np.random.default_rng(7)
    N = len(inst_w)
    sel = np.sort(rng.choice(N, size=min(N, 1500), replace=False))
    W, P, FDW = inst_w[sel], past_v[sel], fut_dw[sel]
    active = np.linalg.norm(P, axis=-1) > 0.30
    sel, W, P, FDW = sel[active], W[active], P[active], FDW[active]
    if len(sel) < 40:
        return {"n_pairs": 0}

    sq = np.sum(W ** 2, axis=-1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (W @ W.T), 0.0)
    np.fill_diagonal(d2, np.inf)
    thr = np.percentile(d2[np.isfinite(d2)], 5.0)
    Pn = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-8)
    cos_p = Pn @ Pn.T

    di, dj = np.where((d2 <= thr) & (cos_p < -0.5))
    si, sj = np.where((d2 <= thr) & (cos_p > 0.7))
    di, dj = di[di < dj], dj[di < dj]
    si, sj = si[si < sj], sj[si < sj]
    res: Dict[str, float] = {"n_div_pairs": int(len(di)), "n_same_pairs": int(len(si))}
    if len(di) < 15 or len(si) < 15:
        return res
    if len(di) > 3000:
        idx = rng.choice(len(di), 3000, replace=False)
        di, dj = di[idx], dj[idx]
    if len(si) > 3000:
        idx = rng.choice(len(si), 3000, replace=False)
        si, sj = si[idx], sj[idx]

    res["true_fut_dw_sep_ratio"] = round(
        float(np.linalg.norm(FDW[di] - FDW[dj], axis=-1).mean() / (np.linalg.norm(FDW[si] - FDW[sj], axis=-1).mean() + 1e-8)),
        4,
    )
    for mname, fd in models_feats_te.items():
        zs = [fd[e]["z"][5 : cache.episodes[e]["len"] - 5] for e in cache.test_eps if cache.episodes[e]["len"] >= 14]
        Z = np.concatenate(zs, axis=0)[sel]
        ratio = float(np.linalg.norm(Z[di] - Z[dj], axis=-1).mean() / (np.linalg.norm(Z[si] - Z[sj], axis=-1).mean() + 1e-8))
        res[f"{mname}_sep_ratio"] = round(ratio, 4)
    return res


def eval_physical_surprise(feats_tr: dict, feats_te: dict, cache: RDPEpisodeCache) -> Dict[str, float]:
    """Speed-controlled detection of sudden contact-transition events (top 10% wrench-jerk transients)."""
    def collect(ep_ids, fd):
        s_mod, spd, dw_n, w_jrk = [], [], [], []
        for e in ep_ids:
            ep = cache.episodes[e]
            if ep["len"] <= 10:
                continue
            sl = slice(5, ep["len"])
            s_mod.append(fd[e]["surprise"][sl])
            spd.append(ep["speed"][sl])
            dw_n.append(ep["dw_norm"][sl])
            # Second derivative of wrench = abrupt contact make/break/jam/snap transient
            w = ep["wraw"]
            d1 = np.diff(w, axis=0, prepend=w[:1])
            d2 = np.linalg.norm(np.diff(d1, axis=0, prepend=d1[:1]), axis=-1)
            w_jrk.append(d2[sl])
        return np.concatenate(s_mod), np.concatenate(spd), np.concatenate(dw_n), np.concatenate(w_jrk)

    tr_s, tr_spd, tr_dw, tr_jrk = collect(cache.train_eps, feats_tr)
    te_s, te_spd, te_dw, te_jrk = collect(cache.test_eps, feats_te)
    thr = np.percentile(tr_jrk, 90.0)
    y_ev = (te_jrk >= thr).astype(np.int32)

    bins = np.quantile(tr_spd, np.linspace(0.0, 1.0, 11))
    bins[0], bins[-1] = -1e9, 1e9

    def sp_norm(tr_v, te_v):
        out = np.zeros_like(te_v)
        tb = np.digitize(tr_spd, bins) - 1
        eb = np.digitize(te_spd, bins) - 1
        for b in range(10):
            mt, me = (tb == b), (eb == b)
            if mt.sum() >= 5 and me.sum() > 0:
                out[me] = (te_v[me] - tr_v[mt].mean()) / (tr_v[mt].std() + 1e-6)
            elif me.sum() > 0:
                out[me] = (te_v[me] - tr_v.mean()) / (tr_v.std() + 1e-6)
        return out

    te_s_sn = sp_norm(tr_s, te_s)
    te_dw_sn = sp_norm(tr_dw, te_dw)
    return {
        "n_test_frames": int(len(y_ev)),
        "raw_dw_auroc": round(float(roc_auc_score(y_ev, te_dw)), 4),
        "raw_dw_speednorm_auroc": round(float(roc_auc_score(y_ev, te_dw_sn)), 4),
        "paec_surprise_raw_auroc": round(float(roc_auc_score(y_ev, te_s)), 4),
        "paec_surprise_speednorm_auroc": round(float(roc_auc_score(y_ev, te_s_sn)), 4),
        "paec_speed_corr_raw": round(float(np.corrcoef(te_s, te_spd)[0, 1]), 4),
        "paec_speed_corr_norm": round(float(np.corrcoef(te_s_sn, te_spd)[0, 1]), 4),
    }


# ==============================================================================
# 3. DOWNSTREAM LOW-SHOT ACTION-CHUNK BC BENCHMARK
# ==============================================================================
def build_bc_samples(
    cache: RDPEpisodeCache,
    ep_ids: List[int],
    feats_dict: Optional[dict] = None,
    feat_key: str = "z",
    hist_len: int = 4,
    chunk_len: int = 8,
    raw_win_len: int = 12,
    ood_mode: str = "clean",
    need_raw_win: bool = False,
    raw_wrench_only: bool = False,
) -> Dict[str, np.ndarray]:
    props, feats, rawws, acts, wnorms, revs = [], [], [], [], [], []
    for e in ep_ids:
        ep = cache.episodes[e]
        L = ep["len"]
        st = max(hist_len, raw_win_len)
        en = L - chunk_len
        if en <= st:
            continue
        p_full = np.concatenate([ep["prop"], ep["pdelta"]], axis=-1)
        act_full = ep["act"]
        con_in = cache.apply_ood(ep["con"], ood_mode)
        # Raw causal input matching PAEC input format for B1_RawFT_End2End / B1b_RawWrench6D_End2End
        if need_raw_win:
            c_use = con_in[:, : cache.wrench_dim] if raw_wrench_only else con_in
            diff_c = np.zeros_like(c_use)
            diff_c[1:] = c_use[1:] - c_use[:-1]
            raw_seq = np.concatenate([c_use, diff_c, ep["pdelta"]], axis=-1)
        f_arr = feats_dict[e][feat_key] if feats_dict is not None else None

        for t in range(st, en):
            props.append(p_full[t - hist_len + 1 : t + 1].reshape(-1))
            acts.append(act_full[t : t + chunk_len])
            wnorms.append(ep["w_norm"][t])
            revs.append(ep["rev_mask"][t])
            if f_arr is not None:
                feats.append(f_arr[t])
            elif need_raw_win:
                rawws.append(raw_seq[t - raw_win_len + 1 : t + 1])

    out = {
        "prop": np.asarray(props, dtype=np.float32),
        "act": np.asarray(acts, dtype=np.float32),
        "w_norm": np.asarray(wnorms, dtype=np.float32),
        "rev": np.asarray(revs, dtype=bool),
    }
    if len(feats) > 0:
        out["feat"] = np.asarray(feats, dtype=np.float32)
    if len(rawws) > 0:
        out["raw_win"] = np.asarray(rawws, dtype=np.float32)
    return out


def train_eval_bc(
    mode: str,
    tr_d: dict,
    va_d: dict,
    te_clean: dict,
    te_drift: dict,
    te_gain: dict,
    w_thresh: float,
    raw_contact_dim: int,
    device: torch.device,
    seed: int = 0,
    epochs: int = 35,
    batch_size: int = 128,
) -> Dict[str, float]:
    set_seed(seed)
    p_dim = tr_d["prop"].shape[-1]
    f_dim = tr_d["feat"].shape[-1] if "feat" in tr_d else 0
    policy = DownstreamActionChunkPolicy(
        proprio_hist_dim=p_dim,
        feat_dim=f_dim,
        act_dim=7,
        chunk_len=8,
        mode=mode,
        raw_contact_dim=raw_contact_dim,
    ).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=1.5e-3, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    N = len(tr_d["prop"])
    p_tr = torch.from_numpy(tr_d["prop"]).to(device)
    y_tr = torch.from_numpy(tr_d["act"]).to(device)
    f_tr = torch.from_numpy(tr_d["feat"]).to(device) if mode == "frozen_feat" else None
    r_tr = torch.from_numpy(tr_d["raw_win"]).to(device) if mode == "raw_end2end" else None

    def eval_sp(d: dict) -> Tuple[float, float, float]:
        policy.eval()
        M = len(d["prop"])
        errs = np.zeros(M, dtype=np.float32)
        with torch.no_grad():
            for i in range(0, M, 256):
                pb = torch.from_numpy(d["prop"][i : i + 256]).to(device)
                yb = torch.from_numpy(d["act"][i : i + 256]).to(device)
                fb = torch.from_numpy(d["feat"][i : i + 256]).to(device) if mode == "frozen_feat" else None
                rb = torch.from_numpy(d["raw_win"][i : i + 256]).to(device) if mode == "raw_end2end" else None
                pred = policy(pb, feat=fb, raw_win=rb)
                errs[i : i + 256] = ((pred - yb) ** 2).mean(dim=(1, 2)).cpu().numpy()
        cm = d["w_norm"] >= w_thresh
        rm = d["rev"]
        return (
            float(errs.mean()),
            float(errs[cm].mean()) if cm.sum() > 0 else float(errs.mean()),
            float(errs[rm].mean()) if rm.sum() > 0 else float(errs.mean()),
        )

    best_val = float("inf")
    best_st = None
    rng = np.random.default_rng(seed)
    bs = min(batch_size, max(16, N // 4))
    for _ in range(epochs):
        policy.train()
        perm = rng.permutation(N)
        for i in range(0, N, bs):
            idx = torch.from_numpy(perm[i : i + bs]).long().to(device)
            opt.zero_grad()
            pred = policy(p_tr[idx], feat=f_tr[idx] if f_tr is not None else None, raw_win=r_tr[idx] if r_tr is not None else None)
            loss = F.mse_loss(pred, y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
            opt.step()
        sched.step()
        v_all, _, _ = eval_sp(va_d)
        if v_all < best_val:
            best_val = v_all
            best_st = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    policy.load_state_dict(best_st)
    c_all, c_con, c_rev = eval_sp(te_clean)
    d_all, _, _ = eval_sp(te_drift)
    g_all, _, _ = eval_sp(te_gain)
    return {
        "val_mse": round(best_val, 5),
        "test_all_mse": round(c_all, 5),
        "test_contact_mse": round(c_con, 5),
        "test_rev_mse": round(c_rev, 5),
        "test_drift_ood_mse": round(d_all, 5),
        "test_gain_ood_mse": round(g_all, 5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--pretrain-steps", type=int, default=1200)
    parser.add_argument("--pretrain-seed", type=int, default=42)
    parser.add_argument("--bc-epochs", type=int, default=35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--act-norm-mode", type=str, default="rms", choices=["rms", "si_fixed", "zscore_legacy"])
    parser.add_argument("--con-norm-mode", type=str, default="zscore", choices=["zscore", "si_fixed"])
    parser.add_argument("--contact-subset", type=str, default="all", choices=["all", "wrench6d"])
    parser.add_argument("--d-z", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--save-ckpt", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    cache = RDPEpisodeCache(
        args.cache,
        act_norm_mode=args.act_norm_mode,
        con_norm_mode=args.con_norm_mode,
        contact_subset=args.contact_subset,
    )
    print(
        f"Loaded cache (V4.1): subset={args.contact_subset} (contact_dim={cache.contact_dim}), "
        f"act_norm={args.act_norm_mode}, con_norm={args.con_norm_mode}, meta={cache.meta}"
    )
    unit_audits = run_unit_audits(
        cache.contact_dim,
        cache.wrench_dim,
        cache.act_dim,
        device,
        d_z=args.d_z,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        cache=cache,
    )

    print("\n=== [1] Pretraining Matched Models (TRAIN episodes only, V4.1 Fair Suite) ===")
    d_z, h_dim, n_lay = args.d_z, args.hidden_dim, args.num_layers
    models = {
        "B2_MAE": MAEContactRecon(cache.contact_dim, cache.act_dim, d_z=d_z, hidden_dim=h_dim, num_layers=n_lay, drift_aug_std=0.0).to(device),
        "B2b_MAE_DriftAug": MAEContactRecon(cache.contact_dim, cache.act_dim, d_z=d_z, hidden_dim=h_dim, num_layers=n_lay, drift_aug_std=0.35).to(device),
        "B3_LatentJEPA_d16": AbstractLatentJEPAVICReg(cache.contact_dim, cache.act_dim, d_z=d_z, hidden_dim=h_dim, num_layers=n_lay, drift_aug_std=0.0).to(device),
        "B3b_LatentJEPA_d16_DriftAug": AbstractLatentJEPAVICReg(cache.contact_dim, cache.act_dim, d_z=d_z, hidden_dim=h_dim, num_layers=n_lay, drift_aug_std=0.35).to(device),
        "B3_LatentJEPA_d64": AbstractLatentJEPAVICReg(cache.contact_dim, cache.act_dim, d_z=64, hidden_dim=h_dim, num_layers=n_lay, drift_aug_std=0.0).to(device),
        "B4_NoBG": PAECModel(cache.contact_dim, cache.wrench_dim, cache.act_dim, d_z=d_z, hidden_dim=h_dim, num_layers=n_lay, use_bg_split=False, drift_aug_std=0.35).to(device),
        "B5_SingleFrame": PAECModel(cache.contact_dim, cache.wrench_dim, cache.act_dim, d_z=d_z, hidden_dim=h_dim, num_layers=n_lay, backbone_type="single_frame", drift_aug_std=0.35).to(device),
        "B6_ShortWin": PAECModel(cache.contact_dim, cache.wrench_dim, cache.act_dim, d_z=d_z, hidden_dim=h_dim, num_layers=n_lay, backbone_type="short_win", drift_aug_std=0.35).to(device),
        "Abl_NoDriftAug": PAECModel(cache.contact_dim, cache.wrench_dim, cache.act_dim, d_z=d_z, hidden_dim=h_dim, num_layers=n_lay, use_bg_split=True, drift_aug_std=0.0).to(device),
        "Ours_PAEC": PAECModel(cache.contact_dim, cache.wrench_dim, cache.act_dim, d_z=d_z, hidden_dim=h_dim, num_layers=n_lay, use_bg_split=True, drift_aug_std=0.35).to(device),
    }
    pretrain_stats = {}
    for name, mdl in models.items():
        pretrain_stats[name] = pretrain_model(name, mdl, cache, device, steps=args.pretrain_steps, seed=args.pretrain_seed)

    if args.save_ckpt:
        ckpt_path = out_dir / "pretrained_models.pt"
        torch.save(
            {
                "protocol_version": "v4.1",
                "args": vars(args),
                "meta": cache.meta,
                "con_mean": cache.con_mean,
                "con_std": cache.con_std,
                "act_center": cache.act_center,
                "act_scale": cache.act_scale,
                "act_mean": cache.act_mean,
                "act_std": cache.act_std,
                "act_rms": cache.act_rms,
                "state_dicts": {k: v.state_dict() for k, v in models.items()},
            },
            ckpt_path,
        )
        print(f"Saved pretrained checkpoints to {ckpt_path}")

    print("\n=== [2] Forward Contact Mechanics Fidelity & Geometry (TEST episodes) ===")
    feats_tr, feats_va, feats_te, feats_drift, feats_gain = {}, {}, {}, {}, {}
    geom_and_mech = {}
    for name, mdl in models.items():
        feats_tr[name] = extract_features(mdl, cache, cache.train_eps, device, "clean")
        feats_va[name] = extract_features(mdl, cache, cache.val_eps, device, "clean")
        feats_te[name] = extract_features(mdl, cache, cache.test_eps, device, "clean")
        feats_drift[name] = extract_features(mdl, cache, cache.test_eps, device, "drift")
        feats_gain[name] = extract_features(mdl, cache, cache.test_eps, device, "gain")

        Z_te = np.concatenate([feats_te[name][e]["z"] for e in cache.test_eps], axis=0)
        info = {"test_eff_rank": round(compute_effective_rank(torch.from_numpy(Z_te)), 2), "d_z": int(Z_te.shape[1])}
        if isinstance(mdl, PAECModel):
            info.update(eval_forward_mechanics(mdl, cache, device))
        geom_and_mech[name] = info
        print(f"  [{name:27s}] {info}")

    print("\n=== [3] Codex-Inspired Matched-Wrench / Divergent-History Test ===")
    codex_test = eval_codex_wrench_matching(feats_te, cache)
    print(json.dumps(codex_test, indent=2))

    print("\n=== [4] Calibrated Physical Wrench Surprise Event Detection ===")
    surprise_test = eval_physical_surprise(feats_tr["Ours_PAEC"], feats_te["Ours_PAEC"], cache)
    print(json.dumps(surprise_test, indent=2))

    print("\n=== [5] Downstream Low-Shot Action-Chunk BC Benchmark ===")
    tr_w = np.concatenate([cache.episodes[e]["w_norm"] for e in cache.train_eps])
    w_thresh = float(np.percentile(tr_w, 65.0))
    raw_in_dim = cache.contact_dim * 2 + cache.act_dim
    raw_w6_in_dim = cache.wrench_dim * 2 + cache.act_dim

    bc_specs = [
        ("B0_ProprioOnly", "proprio_only", None, "z", False),
        ("B1_RawFT_End2End", "raw_end2end", None, "z", False),
    ]
    if cache.contact_dim > cache.wrench_dim:
        bc_specs.append(("B1b_RawWrench6D_E2E", "raw_end2end", None, "z", True))
    bc_specs.extend(
        [
            ("B2_MAE", "frozen_feat", "B2_MAE", "z", False),
            ("B2b_MAE_DriftAug", "frozen_feat", "B2b_MAE_DriftAug", "z", False),
            ("B3_JEPA_d16", "frozen_feat", "B3_LatentJEPA_d16", "z", False),
            ("B3b_JEPA_d16_Drift", "frozen_feat", "B3b_LatentJEPA_d16_DriftAug", "z", False),
            ("B3_JEPA_d64", "frozen_feat", "B3_LatentJEPA_d64", "z", False),
            ("B4_NoBG", "frozen_feat", "B4_NoBG", "z", False),
            ("B5_SingleFrame", "frozen_feat", "B5_SingleFrame", "z", False),
            ("Abl_NoDriftAug", "frozen_feat", "Abl_NoDriftAug", "z", False),
            ("Ours_PAEC_z", "frozen_feat", "Ours_PAEC", "z", False),
            ("Ours_PAEC_Probe", "frozen_feat", "Ours_PAEC", "z_probe", False),
        ]
    )

    n_tr_tot = len(cache.train_eps)
    fracs = [0.20, 0.50, 1.00] if n_tr_tot < 25 else [0.10, 0.30, 1.00]
    bc_results = {}
    for frac in fracs:
        fk = f"{int(round(frac * 100))}pct"
        n_sub = max(2, int(round(n_tr_tot * frac)))
        bc_results[fk] = {}
        print(f"\n--- Budget: {fk} ({n_sub}/{n_tr_tot} train episodes) ---")
        for label, mode, mkey, fkey, raw_w_only in bc_specs:
            need_rw = (mode == "raw_end2end")
            cur_raw_dim = raw_w6_in_dim if raw_w_only else raw_in_dim
            va_d = build_bc_samples(cache, cache.val_eps, feats_va[mkey] if mkey else None, fkey, ood_mode="clean", need_raw_win=need_rw, raw_wrench_only=raw_w_only)
            te_c = build_bc_samples(cache, cache.test_eps, feats_te[mkey] if mkey else None, fkey, ood_mode="clean", need_raw_win=need_rw, raw_wrench_only=raw_w_only)
            te_dr = build_bc_samples(cache, cache.test_eps, feats_drift[mkey] if mkey else None, fkey, ood_mode="drift", need_raw_win=need_rw, raw_wrench_only=raw_w_only)
            te_gn = build_bc_samples(cache, cache.test_eps, feats_gain[mkey] if mkey else None, fkey, ood_mode="gain", need_raw_win=need_rw, raw_wrench_only=raw_w_only)

            runs = []
            for sd in args.seeds:
                rng_s = np.random.default_rng(1000 + sd)
                sub_eps = sorted(rng_s.choice(cache.train_eps, size=n_sub, replace=False).tolist())
                tr_d = build_bc_samples(cache, sub_eps, feats_tr[mkey] if mkey else None, fkey, ood_mode="clean", need_raw_win=need_rw, raw_wrench_only=raw_w_only)
                r = train_eval_bc(
                    mode=mode,
                    tr_d=tr_d,
                    va_d=va_d,
                    te_clean=te_c,
                    te_drift=te_dr,
                    te_gain=te_gn,
                    w_thresh=w_thresh,
                    raw_contact_dim=cur_raw_dim,
                    device=device,
                    seed=sd,
                    epochs=args.bc_epochs,
                )
                runs.append(r)

            agg = {}
            for k in runs[0].keys():
                vals = [x[k] for x in runs]
                agg[f"{k}_mean"] = round(float(np.mean(vals)), 5)
                agg[f"{k}_std"] = round(float(np.std(vals)), 5)
            bc_results[fk][label] = agg
            print(
                f"  [{fk}] {label:20s} | All={agg['test_all_mse_mean']:.5f}±{agg['test_all_mse_std']:.5f} "
                f"| Con={agg['test_contact_mse_mean']:.5f} | Rev={agg['test_rev_mse_mean']:.5f} "
                f"| DriftOOD={agg['test_drift_ood_mse_mean']:.5f} | GainOOD={agg['test_gain_ood_mse_mean']:.5f}"
            )

    summary = {
        "protocol_version": "v4.1",
        "args": vars(args),
        "meta": cache.meta,
        "unit_audits": unit_audits,
        "pretrain_stats": pretrain_stats,
        "geom_and_mech": geom_and_mech,
        "codex_history_test": codex_test,
        "physical_surprise_test": surprise_test,
        "downstream_lowshot_bc": bc_results,
    }
    out_json = out_dir / "v4_paec_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved V4.1 PAEC results to {out_json}")


if __name__ == "__main__":
    main()
