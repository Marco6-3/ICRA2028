#!/usr/bin/env python3
"""
V4.3 & V4.4 Audited Scaling Suite:
  - Loads RH20T Canonical 6D Contact shards (20 Hz, TCP local frame, SI units)
  - Strict zero-leakage split:
      1) Held-Out Embodiment (entire robot config held out from pretraining)
      2) Held-Out Tasks (20% of unique tasks held out from pretraining)
      3) Pretrain Pool (remaining tasks & configs, stratified by task)
      4) Cross-Dataset Unseen-Lab/Robot Benchmark: RDP (peel_v3, wipe_v1.5, lift_v2)
  - Exp A: Data Scaling Law (N in {100, 300, 1000, MAX}) with fixed PAEC-Tiny (96/2L/d16)
           against B4_NoBG, B2b_MAE_DriftAug, B3b_LatentJEPA_d16_DriftAug
  - Exp B: Capacity & Bottleneck Scaling at N=MAX:
           Tiny (96/2L/16), Small (256/2L/8,16,32,64), Base (512/4L/16) + Small MAE/JEPA
  - Exp C: Multi-pretrain-seed statistical verification (3 pretrain seeds x 3 BC seeds)
  - Strictly FP32 (allow_tf32=False), SHA-256 frozen-weight audit on all transfers.
"""

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_paec import (
    AbstractLatentJEPAVICReg,
    DownstreamActionChunkPolicy,
    MAEContactRecon,
    PAECModel,
    compute_effective_rank,
)


# Canonical SI scaling (zero-centered, preserves physical N/m stiffness across labs/robots)
# Calibrated to match typical 20 Hz manipulation RMS (~3-4 N, ~1 N*m, ~1.5 mm, ~0.005 rad, ~2 mm)
CANON_WRENCH_SCALE = np.array([3.0, 3.0, 4.0, 1.0, 1.0, 1.0], dtype=np.float32)
CANON_ACT_SCALE = np.array([0.0015, 0.0015, 0.0015, 0.005, 0.005, 0.005, 0.002], dtype=np.float32)
CLIP_LIM = 12.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def weight_checksum(model: nn.Module) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CanonicalEpisode:
    __slots__ = (
        "ep_id",
        "cfg",
        "task",
        "len",
        "con",
        "wraw",
        "w_norm",
        "dw_norm",
        "prop",
        "pdelta",
        "act",
        "act_raw",
        "cw",
        "rev_mask",
        "contact_active_frac",
    )

    def __init__(
        self,
        ep_id: str,
        cfg: str,
        task: str,
        wrench_raw: np.ndarray,
        prop_raw: np.ndarray,
        pdelta_raw: np.ndarray,
        anext_raw: np.ndarray,
        prop_mean: Optional[np.ndarray] = None,
        prop_std: Optional[np.ndarray] = None,
    ):
        self.ep_id = ep_id
        self.cfg = cfg
        self.task = task
        self.len = int(len(wrench_raw))
        self.wraw = wrench_raw[:, :6].astype(np.float32)
        self.con = np.clip(self.wraw / CANON_WRENCH_SCALE, -CLIP_LIM, CLIP_LIM).astype(np.float32)
        self.pdelta = np.clip(pdelta_raw / CANON_ACT_SCALE, -CLIP_LIM, CLIP_LIM).astype(np.float32)
        self.act = np.clip(anext_raw / CANON_ACT_SCALE, -CLIP_LIM, CLIP_LIM).astype(np.float32)
        self.act_raw = anext_raw.astype(np.float32)
        if prop_mean is not None and prop_std is not None:
            self.prop = np.clip((prop_raw - prop_mean) / prop_std, -8.0, 8.0).astype(np.float32)
        else:
            # Zero-out absolute global position bias for cross-robot, keep rot6d & grip
            p_clean = prop_raw.astype(np.float32).copy()
            p_clean[:, :3] = p_clean[:, :3] - p_clean[:1, :3]
            self.prop = p_clean

        w_n = np.linalg.norm(self.wraw[:, :3], axis=-1).astype(np.float32)
        dw_n = np.linalg.norm(np.diff(self.wraw, axis=0, prepend=self.wraw[:1]), axis=-1).astype(np.float32)
        self.w_norm = w_n
        self.dw_norm = dw_n
        self.contact_active_frac = float(((w_n > 1.0) | (np.linalg.norm(np.diff(self.wraw[:, :3], axis=0, prepend=self.wraw[:1, :3]), axis=-1) > 0.2)).mean())
        self.cw = np.clip(
            1.0 + 0.5 * (w_n / (np.percentile(w_n, 75) + 1e-4)) + 0.5 * (dw_n / (np.percentile(dw_n, 75) + 1e-4)),
            1.0,
            3.5,
        ).astype(np.float32)

        raw_v = anext_raw[:, :3]
        k = 4
        rev_mask = np.zeros(self.len, dtype=bool)
        if self.len > 2 * k + 1:
            for t in range(k, self.len - k):
                v1 = raw_v[t - k : t].mean(axis=0)
                v2 = raw_v[t : t + k].mean(axis=0)
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-5 and n2 > 1e-5 and float(np.dot(v1, v2) / (n1 * n2)) < -0.25:
                    rev_mask[t] = True
        self.rev_mask = rev_mask


def load_rh20t_shards(shard_dir: Path, min_len: int = 30, min_contact_frac: float = 0.15, max_force_p95: float = 180.0) -> Tuple[List[CanonicalEpisode], Dict[str, Any]]:
    episodes: List[CanonicalEpisode] = []
    stats = {"raw_episodes": 0, "kept_episodes": 0, "rejected_short": 0, "rejected_low_contact": 0, "rejected_force_outlier": 0}
    for p in sorted(shard_dir.glob("rh20t_cfg*.npz")):
        raw = np.load(p, allow_pickle=False)
        ep_starts = raw["ep_starts"]
        ep_ends = raw["ep_ends"]
        wraw = raw["wrench_raw"]
        prop = raw["proprio_state"]
        pdelta = raw["proprio_delta"]
        anext = raw["action_next"]
        ep_names = raw["ep_names"]
        ep_cfgs = raw["ep_cfg"]
        ep_tasks = raw["ep_task"]
        for i, (s, e) in enumerate(zip(ep_starts, ep_ends)):
            stats["raw_episodes"] += 1
            s, e = int(s), int(e)
            if e - s < min_len:
                stats["rejected_short"] += 1
                continue
            ep_w = wraw[s:e]
            fn = np.linalg.norm(ep_w[:, :3], axis=-1)
            if float(np.percentile(fn, 95)) > max_force_p95 or not np.isfinite(ep_w).all():
                stats["rejected_force_outlier"] += 1
                continue
            dfn = np.linalg.norm(np.diff(ep_w[:, :3], axis=0, prepend=ep_w[:1, :3]), axis=-1)
            c_frac = float(((fn > 1.0) | (dfn > 0.2)).mean())
            if c_frac < min_contact_frac:
                stats["rejected_low_contact"] += 1
                continue
            episodes.append(
                CanonicalEpisode(
                    ep_id=str(ep_names[i]),
                    cfg=str(ep_cfgs[i]),
                    task=str(ep_tasks[i]),
                    wrench_raw=ep_w,
                    prop_raw=prop[s:e],
                    pdelta_raw=pdelta[s:e],
                    anext_raw=anext[s:e],
                )
            )
            stats["kept_episodes"] += 1
    return episodes, stats


def load_rdp_canonical(cache_path: Path, task_name: str) -> Dict[str, Any]:
    raw = np.load(cache_path, allow_pickle=False)
    ep_starts = raw["ep_starts"]
    ep_ends = raw["ep_ends"]
    wraw = raw["wrench_raw"]
    prop = raw["proprio_state"]
    pdelta = raw["proprio_delta"]
    anext = raw["action_next"]
    prop_mean = raw["prop_mean"].astype(np.float32)
    prop_std = raw["prop_std"].astype(np.float32)

    eps = []
    for i, (s, e) in enumerate(zip(ep_starts, ep_ends)):
        s, e = int(s), int(e)
        eps.append(
            CanonicalEpisode(
                ep_id=f"rdp_{task_name}_ep{i}",
                cfg="rdp_flexiv",
                task=task_name,
                wrench_raw=wraw[s:e],
                prop_raw=prop[s:e],
                pdelta_raw=pdelta[s:e],
                anext_raw=anext[s:e],
                prop_mean=prop_mean,
                prop_std=prop_std,
            )
        )
    return {
        "episodes": eps,
        "train_eps": raw["train_eps"].tolist(),
        "val_eps": raw["val_eps"].tolist(),
        "test_eps": raw["test_eps"].tolist(),
    }


def make_zero_leakage_splits(episodes: List[CanonicalEpisode], seed: int = 2026) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    by_cfg = defaultdict(list)
    for idx, ep in enumerate(episodes):
        by_cfg[ep.cfg].append(idx)

    # Hold out one rich non-Franka embodiment completely (>= 40 clean episodes)
    cfg_counts = sorted(((len(v), k) for k, v in by_cfg.items()), reverse=True)
    heldout_cfg = None
    for cnt, cname in cfg_counts:
        if cname != "cfg_0005" and cnt >= 40:
            heldout_cfg = cname
            break
    if heldout_cfg is None:
        heldout_cfg = cfg_counts[-1][1]

    heldout_cfg_all = by_cfg[heldout_cfg]
    rem_indices = [i for i, ep in enumerate(episodes) if ep.cfg != heldout_cfg]

    # From remaining configs, hold out 20% of unique tasks completely
    rem_tasks = sorted(set(episodes[i].task for i in rem_indices))
    perm_t = rng.permutation(len(rem_tasks))
    n_ho_tasks = max(2, int(round(0.20 * len(rem_tasks))))
    heldout_task_names = set(rem_tasks[j] for j in perm_t[:n_ho_tasks])

    heldout_task_all = [i for i in rem_indices if episodes[i].task in heldout_task_names]
    train_pool_raw = [i for i in rem_indices if episodes[i].task not in heldout_task_names]

    # Task-stratified interleave of train_pool so prefixes N=100/300/1000 maintain task & cfg diversity
    by_task = defaultdict(list)
    for idx in train_pool_raw:
        by_task[episodes[idx].task].append(idx)
    for t in by_task:
        rng.shuffle(by_task[t])
    train_pool = []
    task_keys = sorted(by_task.keys())
    rng.shuffle(task_keys)
    while True:
        added = False
        for t in task_keys:
            if by_task[t]:
                train_pool.append(by_task[t].pop())
                added = True
        if not added:
            break

    def _split_eval_set(idxs: List[int]) -> Dict[str, List[int]]:
        arr = np.array(idxs, dtype=np.int64)
        rng.shuffle(arr)
        n = len(arr)
        n_tr = min(35, max(10, int(0.35 * n)))
        n_va = max(8, int(0.20 * n))
        return {
            "bc_train": arr[:n_tr].tolist(),
            "val": arr[n_tr : n_tr + n_va].tolist(),
            "test": arr[n_tr + n_va :].tolist(),
        }

    return {
        "heldout_cfg_name": heldout_cfg,
        "heldout_cfg_split": _split_eval_set(heldout_cfg_all),
        "heldout_task_names": sorted(heldout_task_names),
        "heldout_task_split": _split_eval_set(heldout_task_all),
        "train_pool": train_pool,
        "n_train_tasks": len(task_keys),
        "train_cfgs": sorted(set(episodes[i].cfg for i in train_pool)),
    }


def apply_ood(con: np.ndarray, drift_bias: np.ndarray, ood_mode: str) -> np.ndarray:
    if ood_mode == "clean":
        return con
    elif ood_mode == "drift":
        return np.clip(con + drift_bias, -CLIP_LIM, CLIP_LIM).astype(np.float32)
    elif ood_mode == "gain":
        return np.clip(con * 1.6, -CLIP_LIM, CLIP_LIM).astype(np.float32)
    raise ValueError(ood_mode)


def build_model(arch: str, device: torch.device) -> nn.Module:
    # Supported arch specs:
    # PAEC_Tiny_d16, B4_NoBG_Tiny_d16, B2b_MAE_Tiny_d16, B3b_JEPA_Tiny_d16,
    # PAEC_Small_d8, PAEC_Small_d16, PAEC_Small_d32, PAEC_Small_d64, PAEC_Base_d16,
    # B2b_MAE_Small_d16, B3b_JEPA_Small_d16
    if arch == "PAEC_Tiny_d16":
        return PAECModel(6, 6, 7, d_z=16, hidden_dim=96, num_layers=2, use_bg_split=True, drift_aug_std=0.35).to(device)
    if arch == "B4_NoBG_Tiny_d16":
        return PAECModel(6, 6, 7, d_z=16, hidden_dim=96, num_layers=2, use_bg_split=False, drift_aug_std=0.35).to(device)
    if arch == "B2b_MAE_Tiny_d16":
        return MAEContactRecon(6, 7, d_z=16, hidden_dim=96, num_layers=2, drift_aug_std=0.35).to(device)
    if arch == "B3b_JEPA_Tiny_d16":
        return AbstractLatentJEPAVICReg(6, 7, d_z=16, hidden_dim=96, num_layers=2, drift_aug_std=0.35).to(device)
    if arch.startswith("PAEC_Small_d"):
        dz = int(arch.split("_d")[-1])
        return PAECModel(6, 6, 7, d_z=dz, hidden_dim=256, num_layers=2, use_bg_split=True, drift_aug_std=0.35).to(device)
    if arch == "PAEC_Base_d16":
        return PAECModel(6, 6, 7, d_z=16, hidden_dim=512, num_layers=4, use_bg_split=True, drift_aug_std=0.35).to(device)
    if arch == "B2b_MAE_Small_d16":
        return MAEContactRecon(6, 7, d_z=16, hidden_dim=256, num_layers=2, drift_aug_std=0.35).to(device)
    if arch == "B3b_JEPA_Small_d16":
        return AbstractLatentJEPAVICReg(6, 7, d_z=16, hidden_dim=256, num_layers=2, drift_aug_std=0.35).to(device)
    raise ValueError(f"Unknown arch={arch}")


def pretrain_on_episodes(
    name: str,
    model: nn.Module,
    episodes: List[CanonicalEpisode],
    ep_ids: List[int],
    device: torch.device,
    steps: int,
    batch_size: int = 64,
    seq_len: int = 48,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    valid = [e for e in ep_ids if episodes[e].len > seq_len + 2]
    if len(valid) == 0:
        seq_len = min(episodes[e].len for e in ep_ids) - 2
        valid = list(ep_ids)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    model.train()
    t0 = time.time()
    last = {}

    con_b = np.zeros((batch_size, seq_len, 6), dtype=np.float32)
    aex_b = np.zeros((batch_size, seq_len, 7), dtype=np.float32)
    anx_b = np.zeros((batch_size, seq_len, 7), dtype=np.float32)
    cw_b = np.zeros((batch_size, seq_len), dtype=np.float32)

    for step in range(1, steps + 1):
        chosen = rng.choice(valid, size=batch_size, replace=True)
        for i, eid in enumerate(chosen):
            ep = episodes[int(eid)]
            st = int(rng.integers(0, ep.len - seq_len))
            con_b[i] = ep.con[st : st + seq_len]
            aex_b[i] = ep.pdelta[st : st + seq_len]
            anx_b[i] = ep.act[st : st + seq_len]
            cw_b[i] = ep.cw[st : st + seq_len]
        con_t = torch.from_numpy(con_b).to(device)
        aex_t = torch.from_numpy(aex_b).to(device)
        anx_t = torch.from_numpy(anx_b).to(device)
        cw_t = torch.from_numpy(cw_b).to(device)

        opt.zero_grad()
        if isinstance(model, PAECModel):
            loss, stats = model.compute_loss(con_t, aex_t, anx_t, contact_weight=cw_t)
        elif isinstance(model, MAEContactRecon):
            loss, stats = model.compute_loss(con_t, aex_t)
        elif isinstance(model, AbstractLatentJEPAVICReg):
            loss, stats = model.compute_loss(con_t, aex_t, anx_t)
        else:
            raise TypeError(type(model))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        if isinstance(model, AbstractLatentJEPAVICReg):
            model.update_ema()
        last = stats

    elapsed = time.time() - t0
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    last["train_sec"] = round(elapsed, 1)
    last["samples_per_sec"] = round((steps * batch_size * seq_len) / max(elapsed, 1e-3), 1)
    last["n_params"] = count_params(model) if False else sum(p.numel() for p in model.parameters())
    print(f"  [Pretrain] {name:26s} | steps={steps} | sec={elapsed:.1f}s | {last}", flush=True)
    return last


@torch.no_grad()
def extract_ep_features(model: nn.Module, episodes: List[CanonicalEpisode], ep_ids: List[int], drift_bias: np.ndarray, device: torch.device, ood_mode: str = "clean") -> Dict[int, Dict[str, np.ndarray]]:
    out = {}
    for eid in ep_ids:
        ep = episodes[eid]
        con_np = apply_ood(ep.con, drift_bias, ood_mode)
        con_t = torch.from_numpy(con_np).unsqueeze(0).to(device)
        aex_t = torch.from_numpy(ep.pdelta).unsqueeze(0).to(device)
        if isinstance(model, PAECModel):
            z, probe, surp = model.compute_causal_surprise(con_t, aex_t, horizon_idx=0)
            z_np = z[0].cpu().numpy()
            p_np = probe[0].cpu().numpy()
            s_np = surp[0].cpu().numpy()
            out[eid] = {
                "z": z_np,
                "z_probe": np.concatenate([z_np, p_np, s_np[:, None]], axis=-1).astype(np.float32),
            }
        else:
            z_np = model.encode(con_t, aex_t)[0].cpu().numpy()
            out[eid] = {"z": z_np, "z_probe": z_np}
    return out


@torch.no_grad()
def eval_forward_mechanics_canonical(model: PAECModel, episodes: List[CanonicalEpisode], test_ids: List[int], drift_bias: np.ndarray, device: torch.device) -> Dict[str, float]:
    res = {}
    delta = model.horizons[1] if len(model.horizons) > 1 else model.horizons[0]
    h_idx = 1 if len(model.horizons) > 1 else 0
    for ood_mode in ("clean", "drift"):
        trues, preds, revs = [], [], []
        for eid in test_ids:
            ep = episodes[eid]
            if ep.len <= delta + 2:
                continue
            con_in = apply_ood(ep.con, drift_bias, ood_mode)
            con_t = torch.from_numpy(con_in).unsqueeze(0).to(device)
            aex_t = torch.from_numpy(ep.pdelta).unsqueeze(0).to(device)
            anx_t = torch.from_numpy(ep.act).unsqueeze(0).to(device)
            z = model.encode(con_t, aex_t)
            a_agg = PAECModel.aggregate_action(anx_t, delta)
            dy_hat, _, _ = model.predictor(z[:, :-delta, :], a_agg, h_idx)
            dw_true = ep.con[delta:, :6] - ep.con[:-delta, :6]
            dw_hat = dy_hat[0, :, :6].cpu().numpy()
            trues.append(dw_true)
            preds.append(dw_hat)
            revs.append(ep.rev_mask[:-delta])
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


def eval_codex_canonical(feats_te: Dict[int, Dict[str, np.ndarray]], episodes: List[CanonicalEpisode], test_ids: List[int]) -> Optional[float]:
    inst_w, past_v, zs = [], [], []
    for e in test_ids:
        ep = episodes[e]
        if ep.len < 14:
            continue
        for t in range(5, ep.len - 5):
            inst_w.append(ep.con[t, :6])
            past_v.append((ep.act_raw[t - 4 : t, :3] / CANON_ACT_SCALE[:3]).mean(axis=0))
            zs.append(feats_te[e]["z"][t])
    if len(inst_w) < 60:
        return None
    inst_w = np.asarray(inst_w, dtype=np.float32)
    past_v = np.asarray(past_v, dtype=np.float32)
    zs = np.asarray(zs, dtype=np.float32)

    rng = np.random.default_rng(7)
    N = len(inst_w)
    sel = np.sort(rng.choice(N, size=min(N, 1500), replace=False))
    W, P, Z = inst_w[sel], past_v[sel], zs[sel]
    active = np.linalg.norm(P, axis=-1) > 0.20
    W, P, Z = W[active], P[active], Z[active]
    if len(W) < 40:
        return None
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
    if len(di) < 15 or len(si) < 15:
        return None
    if len(di) > 3000:
        idx = rng.choice(len(di), 3000, replace=False)
        di, dj = di[idx], dj[idx]
    if len(si) > 3000:
        idx = rng.choice(len(si), 3000, replace=False)
        si, sj = si[idx], sj[idx]
    return round(float(np.linalg.norm(Z[di] - Z[dj], axis=-1).mean() / (np.linalg.norm(Z[si] - Z[sj], axis=-1).mean() + 1e-8)), 4)


def build_bc_samples_canonical(
    episodes: List[CanonicalEpisode],
    ep_ids: List[int],
    feats_dict: Optional[dict],
    feat_key: str,
    hist_len: int = 4,
    chunk_len: int = 8,
    raw_win_len: int = 12,
) -> Dict[str, np.ndarray]:
    props, feats, acts, wnorms, revs = [], [], [], [], []
    for e in ep_ids:
        ep = episodes[e]
        st = max(hist_len, raw_win_len)
        en = ep.len - chunk_len
        if en <= st:
            continue
        p_full = np.concatenate([ep.prop, ep.pdelta], axis=-1)
        f_arr = feats_dict[e][feat_key] if feats_dict is not None else None
        for t in range(st, en):
            props.append(p_full[t - hist_len + 1 : t + 1].reshape(-1))
            acts.append(ep.act[t : t + chunk_len])
            wnorms.append(ep.w_norm[t])
            revs.append(ep.rev_mask[t])
            if f_arr is not None:
                feats.append(f_arr[t])
    out = {
        "prop": np.asarray(props, dtype=np.float32),
        "act": np.asarray(acts, dtype=np.float32),
        "w_norm": np.asarray(wnorms, dtype=np.float32),
        "rev": np.asarray(revs, dtype=bool),
    }
    if len(feats) > 0:
        out["feat"] = np.asarray(feats, dtype=np.float32)
    return out


def train_eval_bc_fast(
    mode: str,
    tr_d: dict,
    va_d: dict,
    te_clean: dict,
    te_drift: Optional[dict],
    w_thresh: float,
    device: torch.device,
    seed: int = 0,
    epochs: int = 30,
    batch_size: int = 256,
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
    ).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=1.5e-3, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    N = len(tr_d["prop"])
    p_tr = torch.from_numpy(tr_d["prop"]).to(device)
    y_tr = torch.from_numpy(tr_d["act"]).to(device)
    f_tr = torch.from_numpy(tr_d["feat"]).to(device) if mode == "frozen_feat" else None

    # Pre-upload val/test tensors for ultra-fast evaluation
    p_va = torch.from_numpy(va_d["prop"]).to(device)
    y_va = torch.from_numpy(va_d["act"]).to(device)
    f_va = torch.from_numpy(va_d["feat"]).to(device) if mode == "frozen_feat" else None

    def eval_tensors(p_t, y_t, f_t, w_norm, rev_m) -> Tuple[float, float, float]:
        policy.eval()
        with torch.no_grad():
            pred = policy(p_t, feat=f_t)
            errs = ((pred - y_t) ** 2).mean(dim=(1, 2)).cpu().numpy()
        cm = w_norm >= w_thresh
        return (
            float(errs.mean()),
            float(errs[cm].mean()) if cm.sum() > 0 else float(errs.mean()),
            float(errs[rev_m].mean()) if rev_m.sum() > 0 else float(errs.mean()),
        )

    best_val = float("inf")
    best_st = None
    rng = np.random.default_rng(seed)
    bs = min(batch_size, max(32, N // 4))
    for _ in range(epochs):
        policy.train()
        perm = rng.permutation(N)
        for i in range(0, N, bs):
            idx = torch.from_numpy(perm[i : i + bs]).long().to(device)
            opt.zero_grad()
            pred = policy(p_tr[idx], feat=f_tr[idx] if f_tr is not None else None)
            loss = F.mse_loss(pred, y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
            opt.step()
        sched.step()
        v_all, _, _ = eval_tensors(p_va, y_va, f_va, va_d["w_norm"], va_d["rev"])
        if v_all < best_val:
            best_val = v_all
            best_st = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    policy.load_state_dict(best_st)
    p_te = torch.from_numpy(te_clean["prop"]).to(device)
    y_te = torch.from_numpy(te_clean["act"]).to(device)
    f_te = torch.from_numpy(te_clean["feat"]).to(device) if mode == "frozen_feat" else None
    c_all, c_con, c_rev = eval_tensors(p_te, y_te, f_te, te_clean["w_norm"], te_clean["rev"])

    out = {
        "val_mse": round(best_val, 5),
        "test_all_mse": round(c_all, 5),
        "test_contact_mse": round(c_con, 5),
        "test_rev_mse": round(c_rev, 5),
    }
    if te_drift is not None:
        p_dr = torch.from_numpy(te_drift["prop"]).to(device)
        y_dr = torch.from_numpy(te_drift["act"]).to(device)
        f_dr = torch.from_numpy(te_drift["feat"]).to(device) if mode == "frozen_feat" else None
        d_all, _, _ = eval_tensors(p_dr, y_dr, f_dr, te_drift["w_norm"], te_drift["rev"])
        out["test_drift_ood_mse"] = round(d_all, 5)
    return out


def eval_frozen_model_on_domain(
    model: nn.Module,
    episodes: List[CanonicalEpisode],
    tr_pool: List[int],
    val_ids: List[int],
    test_ids: List[int],
    budgets: List[int],
    drift_bias: np.ndarray,
    device: torch.device,
    bc_seeds: List[int],
    bc_epochs: int = 30,
    feat_key: str = "z",
) -> Dict[str, Any]:
    ck_before = weight_checksum(model)
    feats_tr = extract_ep_features(model, episodes, tr_pool, drift_bias, device, "clean")
    feats_va = extract_ep_features(model, episodes, val_ids, drift_bias, device, "clean")
    feats_te = extract_ep_features(model, episodes, test_ids, drift_bias, device, "clean")
    feats_dr = extract_ep_features(model, episodes, test_ids, drift_bias, device, "drift")

    Z_te = np.concatenate([feats_te[e]["z"] for e in test_ids], axis=0)
    geom = {
        "test_eff_rank": round(compute_effective_rank(torch.from_numpy(Z_te)), 2),
        "d_z": int(Z_te.shape[1]),
        "codex_sep_ratio": eval_codex_canonical(feats_te, episodes, test_ids),
    }
    if isinstance(model, PAECModel):
        geom.update(eval_forward_mechanics_canonical(model, episodes, test_ids, drift_bias, device))

    tr_w = np.concatenate([episodes[e].w_norm for e in tr_pool])
    w_thresh = float(np.percentile(tr_w, 65.0))
    va_d = build_bc_samples_canonical(episodes, val_ids, feats_va, feat_key)
    te_c = build_bc_samples_canonical(episodes, test_ids, feats_te, feat_key)
    te_dr = build_bc_samples_canonical(episodes, test_ids, feats_dr, feat_key)

    bc_out = {}
    for nb in budgets:
        n_sub = min(nb, len(tr_pool))
        bk = f"{n_sub}eps"
        runs = []
        for sd in bc_seeds:
            rng_s = np.random.default_rng(1000 + sd)
            sub = sorted(rng_s.choice(tr_pool, size=n_sub, replace=False).tolist())
            tr_d = build_bc_samples_canonical(episodes, sub, feats_tr, feat_key)
            runs.append(train_eval_bc_fast("frozen_feat", tr_d, va_d, te_c, te_dr, w_thresh, device, seed=sd, epochs=bc_epochs))
        agg = {}
        for k in runs[0].keys():
            vals = [x[k] for x in runs]
            agg[f"{k}_mean"] = round(float(np.mean(vals)), 5)
            agg[f"{k}_std"] = round(float(np.std(vals)), 5)
        bc_out[bk] = agg

    ck_after = weight_checksum(model)
    assert ck_before == ck_after, "Frozen weight mutation detected!"
    return {"checksum": ck_before, "strictly_frozen_pass": True, "geom": geom, "bc": bc_out}


def eval_b0_proprio_on_domain(
    episodes: List[CanonicalEpisode],
    tr_pool: List[int],
    val_ids: List[int],
    test_ids: List[int],
    budgets: List[int],
    device: torch.device,
    bc_seeds: List[int],
    bc_epochs: int = 30,
) -> Dict[str, Any]:
    tr_w = np.concatenate([episodes[e].w_norm for e in tr_pool])
    w_thresh = float(np.percentile(tr_w, 65.0))
    va_d = build_bc_samples_canonical(episodes, val_ids, None, "z")
    te_c = build_bc_samples_canonical(episodes, test_ids, None, "z")
    bc_out = {}
    for nb in budgets:
        n_sub = min(nb, len(tr_pool))
        bk = f"{n_sub}eps"
        runs = []
        for sd in bc_seeds:
            rng_s = np.random.default_rng(1000 + sd)
            sub = sorted(rng_s.choice(tr_pool, size=n_sub, replace=False).tolist())
            tr_d = build_bc_samples_canonical(episodes, sub, None, "z")
            runs.append(train_eval_bc_fast("proprio_only", tr_d, va_d, te_c, None, w_thresh, device, seed=sd, epochs=bc_epochs))
        agg = {}
        for k in runs[0].keys():
            vals = [x[k] for x in runs]
            agg[f"{k}_mean"] = round(float(np.mean(vals)), 5)
            agg[f"{k}_std"] = round(float(np.std(vals)), 5)
        bc_out[bk] = agg
    return bc_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rh20t-dir", type=str, required=True)
    parser.add_argument("--rdp-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--bc-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    t_all = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Load RH20T Canonical shards + RDP canonical tasks
    rh_eps, rh_filter_stats = load_rh20t_shards(Path(args.rh20t_dir))
    print(f"Loaded RH20T canonical: {rh_filter_stats}", flush=True)
    splits = make_zero_leakage_splits(rh_eps, seed=2026)
    print(
        f"RH20T splits: heldout_cfg={splits['heldout_cfg_name']} "
        f"(tr={len(splits['heldout_cfg_split']['bc_train'])}, va={len(splits['heldout_cfg_split']['val'])}, te={len(splits['heldout_cfg_split']['test'])}), "
        f"heldout_tasks={len(splits['heldout_task_names'])} "
        f"(tr={len(splits['heldout_task_split']['bc_train'])}, va={len(splits['heldout_task_split']['val'])}, te={len(splits['heldout_task_split']['test'])}), "
        f"train_pool={len(splits['train_pool'])} across {splits['n_train_tasks']} tasks / {splits['train_cfgs']}",
        flush=True,
    )

    rdp_dir = Path(args.rdp_dir)
    rdp_domains = {
        "rdp_peel": load_rdp_canonical(rdp_dir / "peel_full.npz", "peel"),
        "rdp_wipe": load_rdp_canonical(rdp_dir / "wipe_full.npz", "wipe"),
        "rdp_lift": load_rdp_canonical(rdp_dir / "lift_full.npz", "lift"),
    }

    rng_ood = np.random.default_rng(2028)
    drift_bias = (1.5 * rng_ood.choice([-1.0, 1.0], size=6)).astype(np.float32)

    def _rh_budgets(tr_list: List[int]) -> List[int]:
        n_tr = len(tr_list)
        if n_tr < 25:
            return sorted(set([min(5, n_tr), min(15, n_tr)]))
        return sorted(set([10, min(30, n_tr)]))

    rh_task_budgets = _rh_budgets(splits["heldout_task_split"]["bc_train"])
    rh_cfg_budgets = _rh_budgets(splits["heldout_cfg_split"]["bc_train"])

    # 2. Compute B0_ProprioOnly anchors & In-Domain RDP PAEC anchors under identical Canonical SI units
    print("\n=== Computing Anchors (B0_ProprioOnly + In-Domain RDP PAEC_Tiny_d16) ===", flush=True)
    anchors: Dict[str, Any] = {"b0": {}, "indomain_rdp_paec": {}}
    anchors["b0"]["rh20t_heldout_task"] = eval_b0_proprio_on_domain(
        rh_eps, splits["heldout_task_split"]["bc_train"], splits["heldout_task_split"]["val"],
        splits["heldout_task_split"]["test"], rh_task_budgets, device, args.bc_seeds, args.bc_epochs,
    )
    anchors["b0"]["rh20t_heldout_cfg"] = eval_b0_proprio_on_domain(
        rh_eps, splits["heldout_cfg_split"]["bc_train"], splits["heldout_cfg_split"]["val"],
        splits["heldout_cfg_split"]["test"], rh_cfg_budgets, device, args.bc_seeds, args.bc_epochs,
    )
    for rname, rd in rdp_domains.items():
        n_tr = len(rd["train_eps"])
        b_list = [5, 10, max(12, int(round(0.30 * n_tr))), n_tr]
        anchors["b0"][rname] = eval_b0_proprio_on_domain(
            rd["episodes"], rd["train_eps"], rd["val_eps"], rd["test_eps"], b_list, device, args.bc_seeds, args.bc_epochs
        )
        m_id = build_model("PAEC_Tiny_d16", device)
        pstats = pretrain_on_episodes(f"InDomain_{rname}", m_id, rd["episodes"], rd["train_eps"], device, steps=1200, batch_size=32, seed=42)
        res_id = eval_frozen_model_on_domain(
            m_id, rd["episodes"], rd["train_eps"], rd["val_eps"], rd["test_eps"], b_list, drift_bias, device, args.bc_seeds, args.bc_epochs, "z"
        )
        res_id["pretrain_stats"] = pstats
        anchors["indomain_rdp_paec"][rname] = res_id

    results: Dict[str, Any] = {
        "protocol": "v4_3_v4_4_scaling_canonical_si",
        "canon_wrench_scale": CANON_WRENCH_SCALE.tolist(),
        "canon_act_scale": CANON_ACT_SCALE.tolist(),
        "rh20t_filter_stats": rh_filter_stats,
        "splits": {k: v for k, v in splits.items() if k != "train_pool"} | {"train_pool_size": len(splits["train_pool"])},
        "anchors": anchors,
        "exp_a_data_scaling": {},
        "exp_b_capacity_bottleneck": {},
        "exp_c_multiseed": {},
    }

    def _eval_everywhere(model: nn.Module, feat_key: str = "z") -> Dict[str, Any]:
        ev = {
            "rh20t_heldout_task": eval_frozen_model_on_domain(
                model, rh_eps, splits["heldout_task_split"]["bc_train"], splits["heldout_task_split"]["val"],
                splits["heldout_task_split"]["test"], rh_task_budgets, drift_bias, device, args.bc_seeds, args.bc_epochs, feat_key
            ),
            "rh20t_heldout_cfg": eval_frozen_model_on_domain(
                model, rh_eps, splits["heldout_cfg_split"]["bc_train"], splits["heldout_cfg_split"]["val"],
                splits["heldout_cfg_split"]["test"], rh_cfg_budgets, drift_bias, device, args.bc_seeds, args.bc_epochs, feat_key
            ),
        }
        for rname, rd in rdp_domains.items():
            n_tr = len(rd["train_eps"])
            b_list = [5, 10, max(12, int(round(0.30 * n_tr))), n_tr]
            ev[rname] = eval_frozen_model_on_domain(
                model, rd["episodes"], rd["train_eps"], rd["val_eps"], rd["test_eps"], b_list, drift_bias, device, args.bc_seeds, args.bc_epochs, feat_key
            )
        return ev

    # 3. Exp A: Adaptive 4-point log-spaced Data Scaling Law
    n_max = len(splits["train_pool"])
    if n_max < 700:
        raw_scales = [(40, 1200), (120, 2000), (260, 3200), (n_max, 5000)]
    elif n_max < 1200:
        raw_scales = [(50, 1200), (150, 2000), (400, 3200), (n_max, 5000)]
    else:
        raw_scales = [(100, 1500), (300, 2500), (1000, 4000), (n_max, 6000)]
    scale_schedule = []
    for n_ep, stp in raw_scales:
        if n_ep < n_max or len(scale_schedule) == 0:
            scale_schedule.append((min(n_ep, n_max), stp))
        elif scale_schedule[-1][0] != n_max:
            scale_schedule.append((n_max, stp))

    exp_a_archs = ["B2b_MAE_Tiny_d16", "B3b_JEPA_Tiny_d16", "B4_NoBG_Tiny_d16", "PAEC_Tiny_d16"]
    cached_nmax_models: Dict[str, nn.Module] = {}

    print(f"\n=== EXP A: Data Scaling Law Schedule = {scale_schedule} ===", flush=True)
    for n_ep, steps in scale_schedule:
        sk = f"N_{n_ep}"
        results["exp_a_data_scaling"][sk] = {"n_episodes": n_ep, "steps": steps, "models": {}}
        sub_pool = splits["train_pool"][:n_ep]
        for arch in exp_a_archs:
            mdl = build_model(arch, device)
            pstats = pretrain_on_episodes(f"{sk}/{arch}", mdl, rh_eps, sub_pool, device, steps=steps, batch_size=args.batch_size, seed=42)
            ev = _eval_everywhere(mdl, "z")
            ev["pretrain_stats"] = pstats
            if arch == "PAEC_Tiny_d16":
                ev["eval_probe"] = _eval_everywhere(mdl, "z_probe")
            results["exp_a_data_scaling"][sk]["models"][arch] = ev
            if n_ep == n_max:
                cached_nmax_models[arch] = mdl
        with open(out_dir / "v4_scaling_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 4. Exp B: Capacity & Bottleneck Scaling at N=MAX (matched nmax_steps with Exp A)
    nmax_steps = scale_schedule[-1][1]
    print(f"\n=== EXP B: Capacity & Bottleneck Scaling at N={n_max} (steps={nmax_steps}) ===", flush=True)
    exp_b_archs = [
        "PAEC_Small_d8",
        "PAEC_Small_d16",
        "PAEC_Small_d32",
        "PAEC_Small_d64",
        "PAEC_Base_d16",
        "B2b_MAE_Small_d16",
        "B3b_JEPA_Small_d16",
    ]
    # Include PAEC_Tiny_d16 directly from Exp A N=MAX
    results["exp_b_capacity_bottleneck"]["PAEC_Tiny_d16"] = results["exp_a_data_scaling"][f"N_{n_max}"]["models"]["PAEC_Tiny_d16"]
    for arch in exp_b_archs:
        bs_use = 32 if "Base" in arch else args.batch_size
        mdl = build_model(arch, device)
        pstats = pretrain_on_episodes(f"Nmax/{arch}", mdl, rh_eps, splits["train_pool"], device, steps=nmax_steps, batch_size=bs_use, seed=42)
        ev = _eval_everywhere(mdl, "z")
        ev["pretrain_stats"] = pstats
        if arch == "PAEC_Small_d16":
            ev["eval_probe"] = _eval_everywhere(mdl, "z_probe")
        results["exp_b_capacity_bottleneck"][arch] = ev
        with open(out_dir / "v4_scaling_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 5. Exp C: Multi-Pretrain-Seed Verification (seeds 100, 2026 added to seed 42) at N=MAX
    print(f"\n=== EXP C: Multi-Pretrain-Seed Verification at N={n_max} (steps={nmax_steps}) ===", flush=True)
    exp_c_archs = ["B2b_MAE_Tiny_d16", "B3b_JEPA_Tiny_d16", "B4_NoBG_Tiny_d16", "PAEC_Tiny_d16", "PAEC_Small_d16"]
    for arch in exp_c_archs:
        seed_runs = {}
        if arch in results["exp_a_data_scaling"][f"N_{n_max}"]["models"]:
            seed_runs["seed_42"] = results["exp_a_data_scaling"][f"N_{n_max}"]["models"][arch]
        elif arch in results["exp_b_capacity_bottleneck"]:
            seed_runs["seed_42"] = results["exp_b_capacity_bottleneck"][arch]
        for psd in (100, 2026):
            mdl = build_model(arch, device)
            pstats = pretrain_on_episodes(f"Seed{psd}/{arch}", mdl, rh_eps, splits["train_pool"], device, steps=nmax_steps, batch_size=args.batch_size, seed=psd)
            ev = _eval_everywhere(mdl, "z")
            ev["pretrain_stats"] = pstats
            seed_runs[f"seed_{psd}"] = ev
        results["exp_c_multiseed"][arch] = seed_runs
        with open(out_dir / "v4_scaling_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    results["total_wall_min"] = round((time.time() - t_all) / 60.0, 2)
    with open(out_dir / "v4_scaling_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nALL SCALING EXPERIMENTS COMPLETE in {results['total_wall_min']} min!", flush=True)


if __name__ == "__main__":
    main()
