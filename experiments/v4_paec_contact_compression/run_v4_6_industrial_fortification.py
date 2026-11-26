#!/usr/bin/env python3
"""
V4.6 Industrial Scientific Fortification Suite.

Addresses three key T-ASE / T-MECH / RA-L scientific gaps on top of V4.4 & V4.5
without touching any frozen V4.4/V4.5 code or outputs:
  1. Same-capacity B4_NoBG_Small_d32 ablation (902k params, matched to PAEC_Small_d32)
     evaluated on RH20T (Unseen Task, Unseen Config cfg_0004), RDP 3-task cross-dataset
     transfer, and MuJoCo closed-loop under Clean, Static +1.5sigma Drift, and
     Mid-Episode Step +1.5sigma Drift (t >= 35).
  2. Non-strawman Classical Industrial Admittance Control baselines:
       - Admittance_FixedFeed
       - Admittance_TunedShear
       - Admittance_InitTare (air-tared at t=0; cancels static drift, exposes step drift)
  3. Quantitative Zero-Dither Virtual Directional Stiffness Probe:
       - Pure zero-shot 1D virtual z-probe g_phi(z_t, +/-e_z) vs ground-truth platen_k
       - Frozen Ridge probe (trained on 48 train eps, evaluated on 48 held-out eps)
       - Soft-zone defect detection AUROC (platen_k < 1000 N/m) under Clean & Drift.

Strictly FP32 (allow_tf32=False) with SHA-256 weight immutability verification.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
import torch
import torch.nn as nn

from models_paec import (
    AbstractLatentJEPAVICReg,
    MAEContactRecon,
    PAECModel,
)
from mujoco_contact_bench import (
    CompliantWipingEnv,
    build_policy_model_for_key,
    collect_mujoco_demos,
    expert_compliant_policy,
    sha256_model,
    train_closedloop_bc,
)
from run_v4_scaling_suite import (
    CANON_ACT_SCALE,
    CANON_WRENCH_SCALE,
    eval_frozen_model_on_domain,
    load_rdp_canonical,
    load_rh20t_shards,
    make_zero_leakage_splits,
    pretrain_on_episodes,
)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2)))
    return float(np.sum(xm * ym) / denom) if denom > 1e-12 else 0.0


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    def _rankdata(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        # Handle ties cleanly
        sorted_a = a[order]
        i = 0
        n = len(a)
        while i < n:
            j = i + 1
            while j < n and sorted_a[j] == sorted_a[i]:
                j += 1
            if j - i > 1:
                avg = 0.5 * (i + j - 1)
                ranks[order[i:j]] = avg
            i = j
        return ranks

    return _pearson_r(_rankdata(np.asarray(x)), _rankdata(np.asarray(y)))


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    # Exact Mann-Whitney U with tie handling
    order = np.argsort(y_score, kind="mergesort")
    sorted_s = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    i = 0
    n = len(y_score)
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg
        i = j
    u = float(ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0)
    return float(u / (n_pos * n_neg))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def evaluate_closed_loop_with_drift_schedule(
    net: Optional[nn.Module],
    x_mean: Optional[np.ndarray],
    x_std: Optional[np.ndarray],
    mode: str,
    encoder: Optional[nn.Module],
    device: torch.device,
    n_eval: int = 24,
    eval_seed: int = 9100,
    drift_sigma: float = 0.0,
    drift_start_step: int = 0,
    classical_controller: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate either a learned BC policy or a classical admittance controller on
    CompliantWipingEnv with configurable drift onset (t >= drift_start_step).
    """
    env = CompliantWipingEnv()
    rng = np.random.default_rng(eval_seed)
    drift_rng = np.random.default_rng(eval_seed + 555)
    succs, rmses, max_fns, viols, prog_xs = [], [], [], [], []
    soft_succs, med_succs, stiff_succs = [], [], []
    soft_viols, stiff_viols = [], []
    surprise_scores, raw_norm_scores, gt_labels = [], [], []

    if net is not None:
        xm_t = torch.from_numpy(x_mean).to(device)
        xs_t = torch.from_numpy(x_std).to(device)

    for _ in range(n_eval):
        cfg = env.reset(rng)
        kz = float(cfg["platen_k"])
        drift_vec = (
            drift_sigma
            * drift_rng.choice([-1.0, 1.0], size=6).astype(np.float32)
            * CANON_WRENCH_SCALE
        ) if drift_sigma > 0.0 else np.zeros(6, dtype=np.float32)

        p0 = env.get_proprio10d()[:3].copy()
        prev_a_scaled = np.zeros((1, 7), dtype=np.float32)
        w_hist, pa_hist, fz_hist = [], [], []
        w_tare_offset = np.zeros(6, dtype=np.float32)

        for t in range(env.MAX_STEPS):
            active_drift = drift_vec if t >= drift_start_step else np.zeros(6, dtype=np.float32)
            w_obs = env.last_w_avg + active_drift

            if classical_controller is not None:
                if classical_controller == "Admittance_InitTare" and t == 0:
                    w_tare_offset = w_obs - np.array([0.0, 0.0, env.TOOL_TARE_Z, 0.0, 0.0, 0.0], dtype=np.float32)
                w_ctrl = w_obs - w_tare_offset if classical_controller == "Admittance_InitTare" else w_obs

                if classical_controller == "Admittance_FixedFeed":
                    fz_meas = float(env.TOOL_TARE_Z - w_ctrl[2])
                    z_vel = float(env.data.joint("jz").qvel[0])
                    y_pos = float(env.data.site("tcp_site").xpos[1])
                    if t < 8 and fz_meas < 1.0:
                        dz = -0.0010
                    else:
                        dz = float(np.clip(0.00022 * (fz_meas - env.TARGET_FN) - 0.004 * z_vel, -0.0025, 0.0025))
                    dy = float(np.clip(-0.22 * y_pos, -0.005, 0.005))
                    act_canon = np.array([0.0038, dy, dz, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                elif classical_controller in ("Admittance_TunedShear", "Admittance_InitTare"):
                    act_canon = expert_compliant_policy(env, w_ctrl, t)
                else:
                    raise ValueError(f"Unknown classical_controller={classical_controller}")
                z_prev_pred = None
                w_scaled = (w_obs / CANON_WRENCH_SCALE).astype(np.float32)
            else:
                w_scaled = (w_obs / CANON_WRENCH_SCALE).astype(np.float32)
                p_cur = env.get_proprio10d()
                p_rel = p_cur.copy()
                p_rel[:3] = (p_cur[:3] - p0) / 0.15
                prop = np.concatenate([p_rel, prev_a_scaled[0]], axis=-1).astype(np.float32)
                w_hist.append(w_scaled)
                pa_hist.append(prev_a_scaled[0].copy())

                if mode == "B0_ProprioOnly":
                    feat = prop
                    z_cur_t = None
                elif mode == "B1b_RawWrench6D_E2E":
                    feat = np.concatenate([prop, w_scaled], axis=-1)
                    z_cur_t = None
                else:
                    assert encoder is not None
                    wt = torch.from_numpy(np.stack(w_hist, axis=0)).unsqueeze(0).to(device)
                    pt = torch.from_numpy(np.stack(pa_hist, axis=0)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        z_cur_t = encoder.encode(wt, pt)[:, -1, :]
                        z_cur = z_cur_t.squeeze(0).cpu().numpy()
                    feat = np.concatenate([prop, z_cur], axis=-1)

                with torch.no_grad():
                    xt = (torch.from_numpy(feat).to(device) - xm_t) / xs_t
                    a_pred_scaled = net(xt.unsqueeze(0)).squeeze(0).cpu().numpy()
                act_canon = (a_pred_scaled * CANON_ACT_SCALE).astype(np.float32)

                if isinstance(encoder, PAECModel):
                    with torch.no_grad():
                        at_t = torch.from_numpy(a_pred_scaled).unsqueeze(0).to(device)
                        z_prev_pred = encoder.predictor(z_cur_t, at_t, h_idx=0)[0][:, :6].squeeze(0).cpu().numpy()
                else:
                    z_prev_pred = None

                prev_a_scaled = a_pred_scaled[None, :].astype(np.float32)

            w_next_true_raw, _, haz_gt = env.step_action(act_canon)
            next_drift = drift_vec if (t + 1) >= drift_start_step else np.zeros(6, dtype=np.float32)
            w_next_obs_scaled = ((w_next_true_raw + next_drift) / CANON_WRENCH_SCALE).astype(np.float32)
            dw_true = w_next_obs_scaled - w_scaled

            if z_prev_pred is not None and t >= 8:
                surprise_scores.append(float(np.linalg.norm(dw_true - z_prev_pred)))
                raw_norm_scores.append(float(np.linalg.norm(w_next_obs_scaled)))
                gt_labels.append(int(haz_gt))

            fz_hist.append(float(env.TOOL_TARE_Z - w_next_true_raw[2]))

        fz_arr = np.array(fz_hist)
        max_fn = float(fz_arr.max())
        c_frac = float((fz_arr[10:] >= 1.8).mean())
        fn_rmse = float(np.sqrt(np.mean((fz_arr[10:] - env.TARGET_FN) ** 2)))
        final_x = float(env.data.site("tcp_site").xpos[0])
        viol = bool(max_fn > 16.0)
        succ = bool((final_x >= env.X_TARGET) and (c_frac >= 0.75) and (not viol))
        succs.append(float(succ))
        rmses.append(fn_rmse)
        max_fns.append(max_fn)
        viols.append(float(viol))
        prog_xs.append(final_x)
        if kz < 1000.0:
            soft_succs.append(float(succ))
            soft_viols.append(float(viol))
        elif kz <= 1800.0:
            med_succs.append(float(succ))
        else:
            stiff_succs.append(float(succ))
            stiff_viols.append(float(viol))

    out: Dict[str, Any] = {
        "success_rate": round(100.0 * float(np.mean(succs)), 2),
        "fn_rmse_n": round(float(np.mean(rmses)), 4),
        "peak_fn_n": round(float(np.mean(max_fns)), 3),
        "safety_viol_rate": round(100.0 * float(np.mean(viols)), 2),
        "mean_final_x": round(float(np.mean(prog_xs)), 4),
        "soft_succ_rate": round(100.0 * float(np.mean(soft_succs)), 2) if soft_succs else None,
        "med_succ_rate": round(100.0 * float(np.mean(med_succs)), 2) if med_succs else None,
        "stiff_succ_rate": round(100.0 * float(np.mean(stiff_succs)), 2) if stiff_succs else None,
        "stiff_viol_rate": round(100.0 * float(np.mean(stiff_viols)), 2) if stiff_viols else None,
    }
    if len(set(gt_labels)) == 2:
        out["surprise_auroc"] = round(float(_roc_auc(np.array(gt_labels), np.array(surprise_scores))), 4)
        out["raw_norm_auroc"] = round(float(_roc_auc(np.array(gt_labels), np.array(raw_norm_scores))), 4)
    return out


def _agg_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for metric in runs[0].keys():
        vals = [r[metric] for r in runs if r[metric] is not None]
        if not vals:
            continue
        agg[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{metric}_std"] = round(float(np.std(vals)), 4)
    return agg


def _ridge_cv_fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    alphas: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
) -> Tuple[np.ndarray, float]:
    """Train-only standardization + 4-fold CV Ridge regression -> held-out test predictions."""
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = np.maximum(X_tr.std(axis=0, keepdims=True), 1e-5)
    Xtr = (X_tr - mu) / sd
    Xte = (X_te - mu) / sd
    y_mu = float(y_tr.mean())
    yc = y_tr - y_mu

    n = len(Xtr)
    n_folds = 4
    fold_ids = np.arange(n) % n_folds
    best_alpha = alphas[0]
    best_mse = float("inf")
    d = Xtr.shape[1]
    eye_d = np.eye(d, dtype=np.float64)

    for alpha in alphas:
        errs = []
        for f in range(n_folds):
            tr_m = fold_ids != f
            va_m = fold_ids == f
            Xf, yf = Xtr[tr_m], yc[tr_m]
            w = np.linalg.solve(Xf.T @ Xf + alpha * eye_d, Xf.T @ yf)
            pred_va = Xtr[va_m] @ w + y_mu
            errs.append(float(np.mean((pred_va - y_tr[va_m]) ** 2)))
        m_mse = float(np.mean(errs))
        if m_mse < best_mse:
            best_mse = m_mse
            best_alpha = alpha

    w_full = np.linalg.solve(Xtr.T @ Xtr + best_alpha * eye_d, Xtr.T @ yc)
    pred_te = Xte @ w_full + y_mu
    return pred_te.astype(np.float64), float(best_alpha)


@torch.no_grad()
def run_virtual_stiffness_probe_eval(
    encoders: Dict[str, nn.Module],
    device: torch.device,
    n_train: int = 48,
    n_test: int = 48,
) -> Dict[str, Any]:
    """
    Collect N = n_train + n_test expert-controlled episodes across platen_k in [500, 2600] N/m
    and quantitatively evaluate zero-shot virtual z-stiffness probe g_phi(z_t, +/-e_z) and
    train-only Ridge stiffness probes under Clean, Static +1.5sigma Drift, and Step +1.5sigma Drift.
    """
    total_eps = n_train + n_test
    demos = collect_mujoco_demos(total_eps, seed=7777)
    drift_rng = np.random.default_rng(7890)
    per_ep_drift = [
        (1.5 * drift_rng.choice([-1.0, 1.0], size=6).astype(np.float32) * CANON_WRENCH_SCALE)
        for _ in range(total_eps)
    ]

    regimes = {
        "clean": (0.0, 0, 15, 95),
        "static_drift_1p5": (1.0, 0, 15, 95),
        "step_drift_1p5": (1.0, 35, 40, 95),
    }

    k_true_all = np.array([float(d["cfg"]["platen_k"]) for d in demos], dtype=np.float64)
    y_logk_tr = np.log(k_true_all[:n_train])
    y_logk_te = np.log(k_true_all[n_train:])
    k_true_te = k_true_all[n_train:]
    soft_defect_te = (k_true_te < 1000.0).astype(np.int64)

    results: Dict[str, Any] = {
        "n_train": n_train,
        "n_test": n_test,
        "platen_k_min_test": round(float(k_true_te.min()), 2),
        "platen_k_max_test": round(float(k_true_te.max()), 2),
        "n_soft_defect_test_lt1000": int(soft_defect_te.sum()),
        "test_platen_k_true": [round(float(v), 2) for v in k_true_te],
        "regimes": {},
    }

    for rname, (d_mult, d_start, t_lo, t_hi) in regimes.items():
        r_out: Dict[str, Any] = {}
        # 1. Raw Wrench 6D baseline features: mean([w_scaled, dw_scaled]) over steady contact window
        raw_feats = []
        for idx, dm in enumerate(demos):
            T = dm["contact_obs"].shape[0]
            drift_seq = np.zeros((T, 6), dtype=np.float32)
            if d_mult > 0.0:
                drift_seq[d_start:] = d_mult * per_ep_drift[idx][None, :]
            w_scaled = (dm["contact_obs"] + drift_seq) / CANON_WRENCH_SCALE[None, :]
            dw = np.zeros_like(w_scaled)
            dw[1:] = w_scaled[1:] - w_scaled[:-1]
            mask = np.zeros(T, dtype=bool)
            mask[t_lo:t_hi] = True
            mask &= (dm["hazard_gt"] == 0)
            if mask.sum() < 5:
                mask[t_lo:t_hi] = True
            feat_ep = np.concatenate([w_scaled[mask].mean(axis=0), dw[mask].mean(axis=0)], axis=0)
            raw_feats.append(feat_ep)
        raw_arr = np.stack(raw_feats, axis=0)
        pred_logk_raw, alpha_raw = _ridge_cv_fit_predict(raw_arr[:n_train], y_logk_tr, raw_arr[n_train:])
        pred_k_raw = np.exp(pred_logk_raw)
        r_out["Raw_Wrench6D_Stats"] = {
            "ridge_alpha": alpha_raw,
            "r2_logk": round(_r2_score(y_logk_te, pred_logk_raw), 4),
            "r2_k": round(_r2_score(k_true_te, pred_k_raw), 4),
            "pearson_r_k": round(_pearson_r(k_true_te, pred_k_raw), 4),
            "spearman_rho_k": round(_spearman_rho(k_true_te, pred_k_raw), 4),
            "mae_k_nm": round(float(np.mean(np.abs(k_true_te - pred_k_raw))), 2),
            "soft_defect_auroc": round(float(_roc_auc(soft_defect_te, -pred_k_raw)), 4),
        }

        # 2. Frozen Encoders
        for ename, enc in encoders.items():
            enc.eval()
            z_eps, probe18_eps, kz_scalar_eps = [], [], []
            for idx, dm in enumerate(demos):
                T = dm["contact_obs"].shape[0]
                drift_seq = np.zeros((T, 6), dtype=np.float32)
                if d_mult > 0.0:
                    drift_seq[d_start:] = d_mult * per_ep_drift[idx][None, :]
                w_scaled = ((dm["contact_obs"] + drift_seq) / CANON_WRENCH_SCALE[None, :]).astype(np.float32)
                a_scaled = (dm["action_next"] / CANON_ACT_SCALE[None, :]).astype(np.float32)
                prev_a = np.concatenate([np.zeros((1, 7), dtype=np.float32), a_scaled[:-1]], axis=0)

                wt = torch.from_numpy(w_scaled).unsqueeze(0).to(device)
                pt = torch.from_numpy(prev_a).unsqueeze(0).to(device)
                z_t = enc.encode(wt, pt)  # (1, T, d_z)

                mask = np.zeros(T, dtype=bool)
                mask[t_lo:t_hi] = True
                mask &= (dm["hazard_gt"] == 0)
                if mask.sum() < 5:
                    mask[t_lo:t_hi] = True

                z_np = z_t.squeeze(0).cpu().numpy()[mask].mean(axis=0)
                z_eps.append(z_np)

                if isinstance(enc, PAECModel):
                    p18 = enc.compute_virtual_probe(z_t, step_scale=1.0).squeeze(0).cpu().numpy()[mask].mean(axis=0)
                    probe18_eps.append(p18)
                    # +e_z is axis=2 sign=+1 (indices 12,13,14 -> Fz at 14); -e_z is axis=2 sign=-1 (indices 15,16,17 -> Fz at 17)
                    kz_sc = float(0.5 * (p18[14] - p18[17]))
                    kz_scalar_eps.append(kz_sc)

            z_arr = np.stack(z_eps, axis=0)
            pred_logk_z, alpha_z = _ridge_cv_fit_predict(z_arr[:n_train], y_logk_tr, z_arr[n_train:])
            pred_k_z = np.exp(pred_logk_z)
            e_res: Dict[str, Any] = {
                "latent_z_ridge": {
                    "ridge_alpha": alpha_z,
                    "r2_logk": round(_r2_score(y_logk_te, pred_logk_z), 4),
                    "r2_k": round(_r2_score(k_true_te, pred_k_z), 4),
                    "pearson_r_k": round(_pearson_r(k_true_te, pred_k_z), 4),
                    "spearman_rho_k": round(_spearman_rho(k_true_te, pred_k_z), 4),
                    "mae_k_nm": round(float(np.mean(np.abs(k_true_te - pred_k_z))), 2),
                    "soft_defect_auroc": round(float(_roc_auc(soft_defect_te, -pred_k_z)), 4),
                    "test_pred_k": [round(float(v), 2) for v in pred_k_z],
                }
            }

            if len(probe18_eps) > 0:
                kz_arr = np.array(kz_scalar_eps, dtype=np.float64)
                # Determine physical sign purely on train split (zero test leakage)
                tr_r = _pearson_r(kz_arr[:n_train], k_true_all[:n_train])
                sign = 1.0 if tr_r >= 0.0 else -1.0
                kz_te = kz_arr[n_train:]
                p18_arr = np.stack(probe18_eps, axis=0)
                pred_logk_p18, alpha_p18 = _ridge_cv_fit_predict(p18_arr[:n_train], y_logk_tr, p18_arr[n_train:])
                pred_k_p18 = np.exp(pred_logk_p18)

                e_res["probe_kz_zero_shot"] = {
                    "train_sign": sign,
                    "pearson_r_k": round(_pearson_r(sign * kz_te, k_true_te), 4),
                    "pearson_r_logk": round(_pearson_r(sign * kz_te, y_logk_te), 4),
                    "spearman_rho_k": round(_spearman_rho(sign * kz_te, k_true_te), 4),
                    "soft_defect_auroc": round(float(_roc_auc(soft_defect_te, -(sign * kz_te))), 4),
                    "test_scalar": [round(float(v), 5) for v in (sign * kz_te)],
                }
                e_res["probe_18d_ridge"] = {
                    "ridge_alpha": alpha_p18,
                    "r2_logk": round(_r2_score(y_logk_te, pred_logk_p18), 4),
                    "r2_k": round(_r2_score(k_true_te, pred_k_p18), 4),
                    "pearson_r_k": round(_pearson_r(k_true_te, pred_k_p18), 4),
                    "spearman_rho_k": round(_spearman_rho(k_true_te, pred_k_p18), 4),
                    "mae_k_nm": round(float(np.mean(np.abs(k_true_te - pred_k_p18))), 2),
                    "soft_defect_auroc": round(float(_roc_auc(soft_defect_te, -pred_k_p18)), 4),
                    "test_pred_k": [round(float(v), 2) for v in pred_k_p18],
                }
            r_out[ename] = e_res

        results["regimes"][rname] = r_out
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to paec_rh20t_a6000_best.pt")
    parser.add_argument("--baselines-ckpt", type=str, required=True, help="Path to matched_rh20t_baselines.pt")
    parser.add_argument("--rh20t-dir", type=str, required=True)
    parser.add_argument("--rdp-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n-eval", type=int, default=24)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bc-epochs", type=int, default=30)
    args = parser.parse_args()

    t_all = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[V4.6] Device: {device} | strict_fp32=True", flush=True)

    # ------------------------------------------------------------------
    # 1. Load frozen V4.4/V4.5 checkpoints + pretrain B4_NoBG_Small_d32
    # ------------------------------------------------------------------
    encoders: Dict[str, nn.Module] = {}
    pre_hashes: Dict[str, str] = {}

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    for k in ("PAEC_Small_d32", "PAEC_Base_d32"):
        m = build_policy_model_for_key(k, device)
        m.load_state_dict(ckpt["state_dicts"][k])
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        encoders[k] = m
        pre_hashes[k] = sha256_model(m)

    bckpt = torch.load(args.baselines_ckpt, map_location="cpu", weights_only=False)
    for bk in ("B2b_MAE_Small_d16", "B3b_JEPA_Small_d16"):
        m = build_policy_model_for_key(bk, device)
        m.load_state_dict(bckpt[bk])
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        encoders[bk] = m
        pre_hashes[bk] = sha256_model(m)

    rh_eps, rh_filter_stats = load_rh20t_shards(Path(args.rh20t_dir))
    splits = make_zero_leakage_splits(rh_eps, seed=2026)

    b4_ckpt_p = out_dir / "b4_nobg_small_d32.pt"
    b4_model = PAECModel(
        6, 6, 7, d_z=32, hidden_dim=256, num_layers=2, use_bg_split=False, drift_aug_std=0.35
    ).to(device)
    b4_pretrain_stats: Dict[str, Any] = {}
    if b4_ckpt_p.exists():
        b4_save = torch.load(b4_ckpt_p, map_location="cpu", weights_only=False)
        b4_model.load_state_dict(b4_save["state_dict"])
        b4_pretrain_stats = b4_save.get("pretrain_stats", {})
        print(f"[V4.6] Loaded cached B4_NoBG_Small_d32 from {b4_ckpt_p}", flush=True)
    else:
        print(f"[V4.6] Pretraining B4_NoBG_Small_d32 on train_pool N={len(splits['train_pool'])} (bs=128, steps=8000)...", flush=True)
        b4_pretrain_stats = pretrain_on_episodes(
            "V4_6/B4_NoBG_Small_d32",
            b4_model,
            rh_eps,
            splits["train_pool"],
            device,
            steps=8000,
            batch_size=128,
            seq_len=48,
            lr=1.5e-3,
            seed=42,
        )
        torch.save(
            {
                "protocol": "v4_6_b4_nobg_small_d32",
                "n_train_episodes": len(splits["train_pool"]),
                "pretrain_stats": b4_pretrain_stats,
                "state_dict": {kk: vv.detach().cpu() for kk, vv in b4_model.state_dict().items()},
            },
            b4_ckpt_p,
        )
    b4_model.eval()
    for p in b4_model.parameters():
        p.requires_grad = False
    encoders["B4_NoBG_Small_d32"] = b4_model
    pre_hashes["B4_NoBG_Small_d32"] = sha256_model(b4_model)

    # ------------------------------------------------------------------
    # 2. Pillar 1A: Offline V4.4-matched evaluation for B4_NoBG_Small_d32
    # ------------------------------------------------------------------
    print("\n=== [Pillar 1A] Offline RH20T + RDP Evaluation of B4_NoBG_Small_d32 ===", flush=True)
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

    b4_offline = {
        "n_params": int(sum(p.numel() for p in b4_model.parameters())),
        "pretrain_stats": b4_pretrain_stats,
        "rh20t_heldout_task": eval_frozen_model_on_domain(
            b4_model,
            rh_eps,
            splits["heldout_task_split"]["bc_train"],
            splits["heldout_task_split"]["val"],
            splits["heldout_task_split"]["test"],
            rh_task_budgets,
            drift_bias,
            device,
            args.seeds,
            args.bc_epochs,
            "z",
        ),
        "rh20t_heldout_cfg": eval_frozen_model_on_domain(
            b4_model,
            rh_eps,
            splits["heldout_cfg_split"]["bc_train"],
            splits["heldout_cfg_split"]["val"],
            splits["heldout_cfg_split"]["test"],
            rh_cfg_budgets,
            drift_bias,
            device,
            args.seeds,
            args.bc_epochs,
            "z",
        ),
    }
    rdp_5, rdp_10, rdp_full = [], [], []
    rdp_5_dr, rdp_10_dr, rdp_full_dr = [], [], []
    for rname, rd in rdp_domains.items():
        n_tr = len(rd["train_eps"])
        b_list = [5, 10, max(12, int(round(0.30 * n_tr))), n_tr]
        res_rd = eval_frozen_model_on_domain(
            b4_model, rd["episodes"], rd["train_eps"], rd["val_eps"], rd["test_eps"],
            b_list, drift_bias, device, args.seeds, args.bc_epochs, "z",
        )
        b4_offline[rname] = res_rd
        rdp_5.append(res_rd["bc"]["5eps"]["test_all_mse_mean"])
        rdp_10.append(res_rd["bc"]["10eps"]["test_all_mse_mean"])
        rdp_full.append(res_rd["bc"][f"{n_tr}eps"]["test_all_mse_mean"])
        rdp_5_dr.append(res_rd["bc"]["5eps"]["test_drift_ood_mse_mean"])
        rdp_10_dr.append(res_rd["bc"]["10eps"]["test_drift_ood_mse_mean"])
        rdp_full_dr.append(res_rd["bc"][f"{n_tr}eps"]["test_drift_ood_mse_mean"])

    b4_offline["rdp_mean"] = {
        "5eps_clean_mse": round(float(np.mean(rdp_5)), 5),
        "10eps_clean_mse": round(float(np.mean(rdp_10)), 5),
        "100pct_clean_mse": round(float(np.mean(rdp_full)), 5),
        "5eps_drift_mse": round(float(np.mean(rdp_5_dr)), 5),
        "10eps_drift_mse": round(float(np.mean(rdp_10_dr)), 5),
        "100pct_drift_mse": round(float(np.mean(rdp_full_dr)), 5),
    }
    print(
        f"[B4_NoBG_Small_d32 Offline] Task R2={b4_offline['rh20t_heldout_task']['geom']['dw3_clean_r2']:.4f} | "
        f"Cfg R2={b4_offline['rh20t_heldout_cfg']['geom']['dw3_clean_r2']:.4f} | "
        f"RDP 5/10/100%={b4_offline['rdp_mean']['5eps_clean_mse']:.5f}/"
        f"{b4_offline['rdp_mean']['10eps_clean_mse']:.5f}/{b4_offline['rdp_mean']['100pct_clean_mse']:.5f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Pillar 2: Classical Industrial Admittance Control Baselines
    # ------------------------------------------------------------------
    print("\n=== [Pillar 2] Classical Industrial Admittance Baselines ===", flush=True)
    drift_regimes = [
        ("clean", 0.0, 0),
        ("static_drift_1p5", 1.5, 0),
        ("step_drift_1p5", 1.5, 35),
    ]
    classical_res: Dict[str, Any] = {}
    for ctrl_name in ("Admittance_FixedFeed", "Admittance_TunedShear", "Admittance_InitTare"):
        classical_res[ctrl_name] = {}
        for rname, d_sig, d_st in drift_regimes:
            runs_3seed = [
                evaluate_closed_loop_with_drift_schedule(
                    None, None, None, ctrl_name, None, device,
                    n_eval=args.n_eval, eval_seed=es, drift_sigma=d_sig,
                    drift_start_step=d_st, classical_controller=ctrl_name,
                )
                for es in (9100, 9101, 9102)
            ]
            agg = _agg_runs(runs_3seed)
            agg["paired_seed9100"] = runs_3seed[0]
            classical_res[ctrl_name][rname] = agg
            print(
                f"  [{ctrl_name:22s} | {rname:16s}] Succ={agg['success_rate_mean']:5.1f}%±{agg['success_rate_std']:4.1f} | "
                f"RMSE={agg['fn_rmse_n_mean']:.2f}N | Viol={agg['safety_viol_rate_mean']:5.1f}% | "
                f"(paired9100={runs_3seed[0]['success_rate']:5.1f}%)",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 4. Pillar 1B: Closed-Loop MuJoCo Comparison (Clean / Static Drift / Step Drift)
    # ------------------------------------------------------------------
    print("\n=== [Pillar 1B] Closed-Loop MuJoCo Evaluation (Clean / Static / Step Drift) ===", flush=True)
    demos = collect_mujoco_demos(max(args.budgets), seed=2026)
    cl_models = [
        "PAEC_Small_d32",
        "B4_NoBG_Small_d32",
        "B2b_MAE_Small_d16",
        "B3b_JEPA_Small_d16",
    ]
    closedloop_res: Dict[str, Any] = {}
    for mode in cl_models:
        enc = encoders[mode]
        closedloop_res[mode] = {}
        for nb in args.budgets:
            sub_demos = demos[:nb]
            reg_runs: Dict[str, List[Dict[str, Any]]] = {r[0]: [] for r in drift_regimes}
            for sd in args.seeds:
                net, xm, xs = train_closedloop_bc(sub_demos, mode, enc, device, seed=sd, epochs=80)
                for rname, d_sig, d_st in drift_regimes:
                    ev = evaluate_closed_loop_with_drift_schedule(
                        net, xm, xs, mode, enc, device,
                        n_eval=args.n_eval, eval_seed=9100,
                        drift_sigma=d_sig, drift_start_step=d_st,
                    )
                    reg_runs[rname].append(ev)
            closedloop_res[mode][f"{nb}demos"] = {
                rname: _agg_runs(reg_runs[rname]) for rname, _, _ in drift_regimes
            }
            c_m = closedloop_res[mode][f"{nb}demos"]["clean"]["success_rate_mean"]
            s_m = closedloop_res[mode][f"{nb}demos"]["static_drift_1p5"]["success_rate_mean"]
            j_m = closedloop_res[mode][f"{nb}demos"]["step_drift_1p5"]["success_rate_mean"]
            print(
                f"  [{mode:20s} | {nb:2d}d] Clean={c_m:5.1f}% | StaticDrift={s_m:5.1f}% | StepDrift(t>=35)={j_m:5.1f}%",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 5. Pillar 3: Quantitative Zero-Dither Virtual Stiffness Probe
    # ------------------------------------------------------------------
    print("\n=== [Pillar 3] Quantitative Virtual Stiffness Probe (N=48 train / 48 test) ===", flush=True)
    probe_res = run_virtual_stiffness_probe_eval(encoders, device, n_train=48, n_test=48)
    for rname in ("clean", "static_drift_1p5", "step_drift_1p5"):
        r_d = probe_res["regimes"][rname]
        p_zs = r_d["PAEC_Small_d32"]["probe_kz_zero_shot"]
        p_18 = r_d["PAEC_Small_d32"]["probe_18d_ridge"]
        raw_s = r_d["Raw_Wrench6D_Stats"]
        print(
            f"  [{rname:16s}] PAEC-Small ZeroShot 1D r={p_zs['pearson_r_k']:+.4f} rho={p_zs['spearman_rho_k']:+.4f} "
            f"SoftAUROC={p_zs['soft_defect_auroc']:.4f} | "
            f"Probe18D R2_logk={p_18['r2_logk']:.4f} R2_k={p_18['r2_k']:.4f} MAE={p_18['mae_k_nm']:.1f} | "
            f"Raw6D R2_k={raw_s['r2_k']:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 6. SHA-256 Frozen Weight Audit & Save
    # ------------------------------------------------------------------
    for k, h_before in pre_hashes.items():
        if sha256_model(encoders[k]) != h_before:
            raise RuntimeError(f"Frozen weight hash mismatch detected for {k}!")

    out_json = {
        "protocol": "v4_6_industrial_fortification",
        "strict_fp32": True,
        "strictly_frozen_pass": True,
        "pre_hashes": pre_hashes,
        "rh20t_filter_stats": rh_filter_stats,
        "splits_summary": {
            "train_pool_size": len(splits["train_pool"]),
            "heldout_cfg_name": splits["heldout_cfg_name"],
            "n_heldout_tasks": len(splits["heldout_task_names"]),
        },
        "pillar1_b4_nobg_small_d32_offline": b4_offline,
        "pillar1_closedloop_ablation": closedloop_res,
        "pillar2_classical_admittance": classical_res,
        "pillar3_virtual_stiffness_probe": probe_res,
        "total_wall_sec": round(time.time() - t_all, 1),
    }
    out_p = out_dir / "v4_6_results.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"\n[V4.6] COMPLETE in {out_json['total_wall_sec']}s -> {out_p} (strictly_frozen_pass=True)", flush=True)


if __name__ == "__main__":
    main()
