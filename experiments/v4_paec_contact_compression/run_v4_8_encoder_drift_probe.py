#!/usr/bin/env python3
"""
V4.8 Script 1: Encoder-Side Drift Fix Pretraining & Virtual Stiffness Probe Re-Evaluation.

Strictly FP32 (allow_tf32=False), zero-leakage RH20T splits (seed=2026), SHA-256 frozen audit.
Pretrains two encoder-side drift-invariant PAEC variants on RH20T train_pool (N=6,289):
  1. PAEC_Small_d32_PureDiff  (use_abs_wrench=False, causal wrench delta + prev_a only)
  2. PAEC_Small_d32_WideDrift (drift_aug_std=1.5, step_drift_prob=0.35)

Evaluates:
  - Offline RH20T Heldout-Task / Heldout-Cfg & Cross-Dataset RDP 3-task transfer
  - V4.6-matched Zero-Dither Virtual Stiffness Probe (N=48 train / N=48 held-out test, seed=7777)
    across 4 regimes: clean, thermal_drift_0p5, static_drift_1p5, step_drift_1p5_t35
    comparing Raw_Wrench6D_Stats, PAEC_Small_d32, PAEC_Small_d32_InitTare,
    B4_NoBG_Small_d32, PAEC_Small_d32_PureDiff, PAEC_Small_d32_WideDrift.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from act_policy import step_encode_recurrent
from models_paec import PAECModel
from mujoco_contact_bench import (
    CompliantWipingEnv,
    build_policy_model_for_key,
    collect_mujoco_demos,
    sha256_model,
)
from run_v4_6_industrial_fortification import (
    _pearson_r,
    _r2_score,
    _ridge_cv_fit_predict,
    _roc_auc,
    _spearman_rho,
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


@torch.no_grad()
def verify_recurrent_equivalence_all(models: Dict[str, nn.Module], device: torch.device):
    w_seq = torch.randn(2, 24, 6, device=device)
    pa_seq = torch.randn(2, 24, 7, device=device)
    diffs: Dict[str, float] = {}
    for name, m in models.items():
        m.eval()
        z_full = m.encode(w_seq, pa_seq)
        h = None
        z_steps = []
        for t in range(24):
            w_prev = None if t == 0 else w_seq[:, t - 1, :]
            z_t, h = step_encode_recurrent(m, w_seq[:, t, :], w_prev, pa_seq[:, t, :], h)
            z_steps.append(z_t)
        z_rec = torch.stack(z_steps, dim=1)
        diff = float((z_full - z_rec).abs().max().item())
        assert diff < 1e-5, f"O(1) recurrent audit failed for {name}: max_diff={diff}"
        diffs[name] = diff
        print(f"  [Audit] O(1) recurrent {name:28s}: max_abs_diff={diff:.2e} (< 1e-5 PASS)", flush=True)
    return diffs


def pretrain_or_load(
    name: str,
    model: PAECModel,
    ckpt_path: Path,
    rh_eps: List[Dict[str, Any]],
    train_pool: List[int],
    device: torch.device,
    steps: int = 8000,
) -> Dict[str, Any]:
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        stats = ck.get("pretrain_stats", {})
        print(f"[V4.8] Loaded cached {name} from {ckpt_path}", flush=True)
    else:
        print(f"[V4.8] Pretraining {name} on train_pool N={len(train_pool)} (bs=128, steps={steps})...", flush=True)
        stats = pretrain_on_episodes(
            f"V4_8/{name}",
            model,
            rh_eps,
            train_pool,
            device,
            steps=steps,
            batch_size=128,
            seq_len=48,
            lr=1.5e-3,
            seed=42,
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol": f"v4_8_{name.lower()}",
                "n_train_episodes": len(train_pool),
                "pretrain_stats": stats,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            },
            ckpt_path,
        )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return stats


@torch.no_grad()
def run_v4_8_stiffness_probe(
    encoders: Dict[str, nn.Module],
    device: torch.device,
    n_train: int = 48,
    n_test: int = 48,
) -> Dict[str, Any]:
    """
    Exact V4.6-matched probe dataset (seed=7777, drift_rng=7890) across 4 regimes:
      clean (0.0), thermal_drift_0p5 (+0.5sigma), static_drift_1p5 (+1.5sigma), step_drift_1p5_t35 (t>=35).
    Also tests PAEC_Small_d32_InitTare (frozen V4.4 PAEC_Small_d32 with t=0 free-air tare).
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
        "thermal_drift_0p5": (1.0 / 3.0, 0, 15, 95),
        "static_drift_1p5": (1.0, 0, 15, 95),
        "step_drift_1p5_t35": (1.0, 35, 40, 95),
    }

    k_true_all = np.array([float(d["cfg"]["platen_k"]) for d in demos], dtype=np.float64)
    y_logk_tr = np.log(k_true_all[:n_train])
    y_logk_te = np.log(k_true_all[n_train:])
    k_true_te = k_true_all[n_train:]
    soft_defect_te = (k_true_te < 1000.0).astype(np.int64)
    nominal_air_tare = np.array([0.0, 0.0, CompliantWipingEnv.TOOL_TARE_Z, 0.0, 0.0, 0.0], dtype=np.float32)

    results: Dict[str, Any] = {
        "n_train": n_train,
        "n_test": n_test,
        "platen_k_min_test": round(float(k_true_te.min()), 2),
        "platen_k_max_test": round(float(k_true_te.max()), 2),
        "n_soft_defect_test_lt1000": int(soft_defect_te.sum()),
        "regimes": {},
    }

    eval_entries = [
        ("PAEC_Small_d32", encoders["PAEC_Small_d32"], False),
        ("PAEC_Small_d32_InitTare", encoders["PAEC_Small_d32"], True),
        ("B4_NoBG_Small_d32", encoders["B4_NoBG_Small_d32"], False),
        ("PAEC_Small_d32_PureDiff", encoders["PAEC_Small_d32_PureDiff"], False),
        ("PAEC_Small_d32_WideDrift", encoders["PAEC_Small_d32_WideDrift"], False),
    ]

    for rname, (d_mult, d_start, t_lo, t_hi) in regimes.items():
        r_out: Dict[str, Any] = {}
        # 1. Raw Wrench 6D Stats
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
            raw_feats.append(np.concatenate([w_scaled[mask].mean(axis=0), dw[mask].mean(axis=0)], axis=0))
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

        # 2. Encoders
        for ename, enc, use_init_tare in eval_entries:
            enc.eval()
            z_eps, probe18_eps, kz_scalar_eps = [], [], []
            for idx, dm in enumerate(demos):
                T = dm["contact_obs"].shape[0]
                drift_seq = np.zeros((T, 6), dtype=np.float32)
                if d_mult > 0.0:
                    drift_seq[d_start:] = d_mult * per_ep_drift[idx][None, :]
                w_obs = dm["contact_obs"] + drift_seq
                if use_init_tare:
                    w_tare = w_obs[0] - nominal_air_tare
                    w_obs = w_obs - w_tare[None, :]
                w_scaled = (w_obs / CANON_WRENCH_SCALE[None, :]).astype(np.float32)
                a_scaled = (dm["action_next"] / CANON_ACT_SCALE[None, :]).astype(np.float32)
                prev_a = np.concatenate([np.zeros((1, 7), dtype=np.float32), a_scaled[:-1]], axis=0)

                wt = torch.from_numpy(w_scaled).unsqueeze(0).to(device)
                pt = torch.from_numpy(prev_a).unsqueeze(0).to(device)
                z_t = enc.encode(wt, pt)

                mask = np.zeros(T, dtype=bool)
                mask[t_lo:t_hi] = True
                mask &= (dm["hazard_gt"] == 0)
                if mask.sum() < 5:
                    mask[t_lo:t_hi] = True

                z_eps.append(z_t.squeeze(0).cpu().numpy()[mask].mean(axis=0))
                p18 = enc.compute_virtual_probe(z_t, step_scale=1.0).squeeze(0).cpu().numpy()[mask].mean(axis=0)
                probe18_eps.append(p18)
                kz_scalar_eps.append(float(0.5 * (p18[14] - p18[17])))

            z_arr = np.stack(z_eps, axis=0)
            pred_logk_z, alpha_z = _ridge_cv_fit_predict(z_arr[:n_train], y_logk_tr, z_arr[n_train:])
            pred_k_z = np.exp(pred_logk_z)

            kz_arr = np.array(kz_scalar_eps, dtype=np.float64)
            sign = 1.0 if _pearson_r(kz_arr[:n_train], k_true_all[:n_train]) >= 0.0 else -1.0
            kz_te = kz_arr[n_train:]
            p18_arr = np.stack(probe18_eps, axis=0)
            pred_logk_p18, alpha_p18 = _ridge_cv_fit_predict(p18_arr[:n_train], y_logk_tr, p18_arr[n_train:])
            pred_k_p18 = np.exp(pred_logk_p18)

            r_out[ename] = {
                "latent_z_ridge": {
                    "ridge_alpha": alpha_z,
                    "r2_logk": round(_r2_score(y_logk_te, pred_logk_z), 4),
                    "r2_k": round(_r2_score(k_true_te, pred_k_z), 4),
                    "pearson_r_k": round(_pearson_r(k_true_te, pred_k_z), 4),
                    "spearman_rho_k": round(_spearman_rho(k_true_te, pred_k_z), 4),
                    "mae_k_nm": round(float(np.mean(np.abs(k_true_te - pred_k_z))), 2),
                    "soft_defect_auroc": round(float(_roc_auc(soft_defect_te, -pred_k_z)), 4),
                },
                "probe_kz_zero_shot": {
                    "train_sign": sign,
                    "pearson_r_k": round(_pearson_r(sign * kz_te, k_true_te), 4),
                    "spearman_rho_k": round(_spearman_rho(sign * kz_te, k_true_te), 4),
                    "soft_defect_auroc": round(float(_roc_auc(soft_defect_te, -(sign * kz_te))), 4),
                },
                "probe_18d_ridge": {
                    "ridge_alpha": alpha_p18,
                    "r2_logk": round(_r2_score(y_logk_te, pred_logk_p18), 4),
                    "r2_k": round(_r2_score(k_true_te, pred_k_p18), 4),
                    "pearson_r_k": round(_pearson_r(k_true_te, pred_k_p18), 4),
                    "spearman_rho_k": round(_spearman_rho(k_true_te, pred_k_p18), 4),
                    "mae_k_nm": round(float(np.mean(np.abs(k_true_te - pred_k_p18))), 2),
                    "soft_defect_auroc": round(float(_roc_auc(soft_defect_te, -pred_k_p18)), 4),
                },
            }

        results["regimes"][rname] = r_out
    return results


def eval_offline_rh20t_and_rdp(
    name: str,
    model: PAECModel,
    rh_eps: List[Dict[str, Any]],
    splits: Dict[str, Any],
    rdp_domains: Dict[str, Any],
    device: torch.device,
    seeds: List[int],
    bc_epochs: int = 30,
) -> Dict[str, Any]:
    rng_ood = np.random.default_rng(2028)
    drift_bias = (1.5 * rng_ood.choice([-1.0, 1.0], size=6)).astype(np.float32)

    def _rh_budgets(tr_list: List[int]) -> List[int]:
        n_tr = len(tr_list)
        if n_tr < 25:
            return sorted(set([min(5, n_tr), min(15, n_tr)]))
        return sorted(set([10, min(30, n_tr)]))

    rh_task_budgets = _rh_budgets(splits["heldout_task_split"]["bc_train"])
    rh_cfg_budgets = _rh_budgets(splits["heldout_cfg_split"]["bc_train"])

    out: Dict[str, Any] = {
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "rh20t_heldout_task": eval_frozen_model_on_domain(
            model,
            rh_eps,
            splits["heldout_task_split"]["bc_train"],
            splits["heldout_task_split"]["val"],
            splits["heldout_task_split"]["test"],
            rh_task_budgets,
            drift_bias,
            device,
            seeds,
            bc_epochs,
            "z",
        ),
        "rh20t_heldout_cfg": eval_frozen_model_on_domain(
            model,
            rh_eps,
            splits["heldout_cfg_split"]["bc_train"],
            splits["heldout_cfg_split"]["val"],
            splits["heldout_cfg_split"]["test"],
            rh_cfg_budgets,
            drift_bias,
            device,
            seeds,
            bc_epochs,
            "z",
        ),
    }

    rdp_5, rdp_10, rdp_full = [], [], []
    rdp_5_dr, rdp_10_dr, rdp_full_dr = [], [], []
    for rname, rd in rdp_domains.items():
        n_tr = len(rd["train_eps"])
        b_list = [5, 10, max(12, int(round(0.30 * n_tr))), n_tr]
        res_rd = eval_frozen_model_on_domain(
            model, rd["episodes"], rd["train_eps"], rd["val_eps"], rd["test_eps"],
            b_list, drift_bias, device, seeds, bc_epochs, "z",
        )
        out[rname] = res_rd
        rdp_5.append(res_rd["bc"]["5eps"]["test_all_mse_mean"])
        rdp_10.append(res_rd["bc"]["10eps"]["test_all_mse_mean"])
        rdp_full.append(res_rd["bc"][f"{n_tr}eps"]["test_all_mse_mean"])
        rdp_5_dr.append(res_rd["bc"]["5eps"]["test_drift_ood_mse_mean"])
        rdp_10_dr.append(res_rd["bc"]["10eps"]["test_drift_ood_mse_mean"])
        rdp_full_dr.append(res_rd["bc"][f"{n_tr}eps"]["test_drift_ood_mse_mean"])

    out["rdp_mean"] = {
        "5eps_clean_mse": round(float(np.mean(rdp_5)), 5),
        "10eps_clean_mse": round(float(np.mean(rdp_10)), 5),
        "100pct_clean_mse": round(float(np.mean(rdp_full)), 5),
        "5eps_drift_mse": round(float(np.mean(rdp_5_dr)), 5),
        "10eps_drift_mse": round(float(np.mean(rdp_10_dr)), 5),
        "100pct_drift_mse": round(float(np.mean(rdp_full_dr)), 5),
    }
    print(
        f"[{name:24s} Offline] Task R2={out['rh20t_heldout_task']['geom']['dw3_clean_r2']:.4f} "
        f"(Drift={out['rh20t_heldout_task']['geom']['dw3_drift_r2']:.4f}) | "
        f"Cfg R2={out['rh20t_heldout_cfg']['geom']['dw3_clean_r2']:.4f} | "
        f"RDP 10-shot Clean={out['rdp_mean']['10eps_clean_mse']:.5f} Drift={out['rdp_mean']['10eps_drift_mse']:.5f}",
        flush=True,
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--b4-ckpt", type=str, required=True)
    parser.add_argument("--rh20t-dir", type=str, required=True)
    parser.add_argument("--rdp-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--pretrain-steps", type=int, default=8000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--skip-offline", action="store_true")
    args = parser.parse_args()

    t_all = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[V4.8 Probe] Device={device} | strict_fp32=True", flush=True)

    encoders: Dict[str, nn.Module] = {}
    pre_hashes: Dict[str, str] = {}

    # Load frozen V4.4 PAEC_Small_d32 & V4.6 B4_NoBG_Small_d32
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m_paec = build_policy_model_for_key("PAEC_Small_d32", device)
    m_paec.load_state_dict(ckpt["state_dicts"]["PAEC_Small_d32"])
    m_paec.eval()
    for p in m_paec.parameters():
        p.requires_grad = False
    encoders["PAEC_Small_d32"] = m_paec
    pre_hashes["PAEC_Small_d32"] = sha256_model(m_paec)

    b4_ck = torch.load(args.b4_ckpt, map_location="cpu", weights_only=False)
    m_b4 = PAECModel(6, 6, 7, d_z=32, hidden_dim=256, num_layers=2, use_bg_split=False, drift_aug_std=0.35).to(device)
    m_b4.load_state_dict(b4_ck["state_dict"])
    m_b4.eval()
    for p in m_b4.parameters():
        p.requires_grad = False
    encoders["B4_NoBG_Small_d32"] = m_b4
    pre_hashes["B4_NoBG_Small_d32"] = sha256_model(m_b4)

    # Load RH20T and pretrain/load PureDiff & WideDrift
    rh_eps, rh_filter_stats = load_rh20t_shards(Path(args.rh20t_dir))
    splits = make_zero_leakage_splits(rh_eps, seed=2026)

    m_purediff = PAECModel(
        6, 6, 7, d_z=32, hidden_dim=256, num_layers=2,
        use_bg_split=True, use_abs_wrench=False, drift_aug_std=0.0,
    ).to(device)
    stats_purediff = pretrain_or_load(
        "PAEC_Small_d32_PureDiff",
        m_purediff,
        out_dir / "paec_small_d32_purediff.pt",
        rh_eps,
        splits["train_pool"],
        device,
        steps=args.pretrain_steps,
    )
    encoders["PAEC_Small_d32_PureDiff"] = m_purediff
    pre_hashes["PAEC_Small_d32_PureDiff"] = sha256_model(m_purediff)

    m_widedrift = PAECModel(
        6, 6, 7, d_z=32, hidden_dim=256, num_layers=2,
        use_bg_split=True, use_abs_wrench=True, drift_aug_std=1.5, step_drift_prob=0.35,
    ).to(device)
    stats_widedrift = pretrain_or_load(
        "PAEC_Small_d32_WideDrift",
        m_widedrift,
        out_dir / "paec_small_d32_widedrift.pt",
        rh_eps,
        splits["train_pool"],
        device,
        steps=args.pretrain_steps,
    )
    encoders["PAEC_Small_d32_WideDrift"] = m_widedrift
    pre_hashes["PAEC_Small_d32_WideDrift"] = sha256_model(m_widedrift)

    print("\n=== O(1) Recurrent Equivalence Audit ===", flush=True)
    rec_diffs = verify_recurrent_equivalence_all(encoders, device)

    offline_res: Dict[str, Any] = {}
    if not args.skip_offline:
        print("\n=== Offline RH20T + Cross-Dataset RDP Evaluation ===", flush=True)
        rdp_dir = Path(args.rdp_dir)
        rdp_domains = {
            "rdp_peel": load_rdp_canonical(rdp_dir / "peel_full.npz", "peel"),
            "rdp_wipe": load_rdp_canonical(rdp_dir / "wipe_full.npz", "wipe"),
            "rdp_lift": load_rdp_canonical(rdp_dir / "lift_full.npz", "lift"),
        }
        offline_res["PAEC_Small_d32_PureDiff"] = {
            "pretrain_stats": stats_purediff,
            **eval_offline_rh20t_and_rdp(
                "PAEC_Small_d32_PureDiff", m_purediff, rh_eps, splits, rdp_domains, device, args.seeds, args.bc_epochs
            ),
        }
        offline_res["PAEC_Small_d32_WideDrift"] = {
            "pretrain_stats": stats_widedrift,
            **eval_offline_rh20t_and_rdp(
                "PAEC_Small_d32_WideDrift", m_widedrift, rh_eps, splits, rdp_domains, device, args.seeds, args.bc_epochs
            ),
        }

    print("\n=== Virtual Directional Stiffness Probe (Clean / +0.5σ / Static +1.5σ / Step +1.5σ) ===", flush=True)
    probe_res = run_v4_8_stiffness_probe(encoders, device, n_train=48, n_test=48)
    for rname in ("clean", "thermal_drift_0p5", "static_drift_1p5", "step_drift_1p5_t35"):
        print(f"\n  [{rname}]", flush=True)
        for ename in ("PAEC_Small_d32", "PAEC_Small_d32_InitTare", "B4_NoBG_Small_d32", "PAEC_Small_d32_PureDiff", "PAEC_Small_d32_WideDrift"):
            ed = probe_res["regimes"][rname][ename]
            zs = ed["probe_kz_zero_shot"]
            p18 = ed["probe_18d_ridge"]
            print(
                f"    {ename:25s}: ZS1D r={zs['pearson_r_k']:+.4f} AUROC={zs['soft_defect_auroc']:.4f} | "
                f"Probe18D R2_k={p18['r2_k']:+.4f} R2_logk={p18['r2_logk']:+.4f} AUROC={p18['soft_defect_auroc']:.4f}",
                flush=True,
            )

    for k, h_before in pre_hashes.items():
        if sha256_model(encoders[k]) != h_before:
            raise RuntimeError(f"Frozen weight hash mismatch for {k}!")

    out_json = {
        "protocol": "v4_8_encoder_drift_probe",
        "strict_fp32": True,
        "strictly_frozen_pass": True,
        "pre_hashes": pre_hashes,
        "recurrent_max_abs_diff": rec_diffs,
        "rh20t_filter_stats": rh_filter_stats,
        "offline_rh20t_rdp": offline_res,
        "virtual_stiffness_probe": probe_res,
        "total_wall_sec": round(time.time() - t_all, 1),
    }
    out_p = out_dir / "v4_8_encoder_drift_probe.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"\n[V4.8 Probe] COMPLETE in {out_json['total_wall_sec']}s -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
