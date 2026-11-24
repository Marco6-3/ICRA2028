#!/usr/bin/env python3
"""
V4 Route 1: Zero-Shot Cross-Task Frozen Contact Representation Transfer (peel <-> wipe).

Pretrains (or loads frozen checkpoints of) representation models on SOURCE task TRAIN episodes ONLY,
completely freezes all encoder & predictor parameters, and evaluates zero-shot transfer onto an
UNSEEN TARGET task under TWO strict normalization protocols:

  1. src_norm (Strict Physical Zero-Shot):
       Target raw physical contact/motion is normalized using SOURCE task train stats ONLY.
       The frozen encoder never touches any Target task statistics (tests true physical zero-drift
       and cross-task scale/mean shift robustness).
  2. tgt_norm (Target-Train Standardized Zero-Shot):
       Target contact/motion is standardized with Target TRAIN-only stats before entering the
       100% frozen Source encoder (zero gradient updates).

Evaluates on UNSEEN TARGET task:
  - Channel & frozen-weight integrity audits
  - Zero-shot forward 6D wrench mechanics fidelity (dw3 R^2 on Clean, Reversal, Drift OOD)
    converted to Target-std units for direct apples-to-apples comparison with in-domain results
  - Zero-shot representation effective rank & Codex matched-wrench / divergent-history separation
  - Zero-shot frozen-encoder low-shot Action-Chunk BC (10% / 30% / 100% x 3 seeds) on
    Clean, Contact-Active, Reversal, Drift OOD (+1.5 sigma), and Gain OOD (1.6x).
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models_paec import (
    AbstractLatentJEPAVICReg,
    MAEContactRecon,
    PAECModel,
    compute_effective_rank,
)
from run_v4_paec_benchmark import (
    RDPEpisodeCache,
    build_bc_samples,
    eval_codex_wrench_matching,
    pretrain_model,
    set_seed,
    train_eval_bc,
)


def weight_checksum(model: nn.Module) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def load_raw_episodes(cache_path: str, contact_subset: str = "all") -> List[Dict[str, np.ndarray]]:
    raw = np.load(cache_path, allow_pickle=False)
    meta = json.loads(str(raw["meta_json"]))
    wdim = int(meta["wrench_dim"])
    con = raw["contact_obs"].astype(np.float32)
    if contact_subset == "wrench6d":
        con = con[:, :wdim]
    pdelta = raw["proprio_delta"].astype(np.float32)
    anext = raw["action_next"].astype(np.float32)
    ep_starts = raw["ep_starts"]
    ep_ends = raw["ep_ends"]
    eps = []
    for s, e in zip(ep_starts, ep_ends):
        s, e = int(s), int(e)
        eps.append(
            {
                "con_raw": con[s:e],
                "pdelta_raw": pdelta[s:e],
                "anext_raw": anext[s:e],
            }
        )
    return eps


def normalize_target_episode(
    raw_ep: Dict[str, np.ndarray],
    src_cache: RDPEpisodeCache,
    tgt_cache: RDPEpisodeCache,
    norm_mode: str,
    ood_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply identical physical OOD perturbation in physical units, then normalize
    using either Source stats (src_norm) or Target train stats (tgt_norm).
    Uses V4.1 zero-centered act_center / act_scale so physical delta_q == 0 <=> a_norm == 0.
    """
    con_phys = raw_ep["con_raw"]
    if ood_mode == "clean":
        con_p = con_phys
    elif ood_mode == "drift":
        con_p = con_phys + tgt_cache.drift_bias * tgt_cache.con_std
    elif ood_mode == "gain":
        con_p = tgt_cache.con_mean + 1.6 * (con_phys - tgt_cache.con_mean)
    else:
        raise ValueError(ood_mode)

    if norm_mode == "src_norm":
        c_m, c_s = src_cache.con_mean, src_cache.con_std
        a_m, a_s = src_cache.act_center, src_cache.act_scale
    elif norm_mode == "tgt_norm":
        c_m, c_s = tgt_cache.con_mean, tgt_cache.con_std
        a_m, a_s = tgt_cache.act_center, tgt_cache.act_scale
    else:
        raise ValueError(norm_mode)

    con_in = np.clip((con_p - c_m) / c_s, -8.0, 8.0).astype(np.float32)
    aex_in = np.clip((raw_ep["pdelta_raw"] - a_m) / a_s, -8.0, 8.0).astype(np.float32)
    anx_in = np.clip((raw_ep["anext_raw"] - a_m) / a_s, -8.0, 8.0).astype(np.float32)
    return con_in, aex_in, anx_in


@torch.no_grad()
def extract_transfer_features(
    model: nn.Module,
    raw_eps: List[Dict[str, np.ndarray]],
    src_cache: RDPEpisodeCache,
    tgt_cache: RDPEpisodeCache,
    ep_ids: List[int],
    device: torch.device,
    norm_mode: str,
    ood_mode: str = "clean",
) -> dict:
    out = {}
    for eid in ep_ids:
        con_np, aex_np, _ = normalize_target_episode(raw_eps[eid], src_cache, tgt_cache, norm_mode, ood_mode)
        con_t = torch.from_numpy(con_np).unsqueeze(0).to(device)
        aex_t = torch.from_numpy(aex_np).unsqueeze(0).to(device)
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
                "surprise": np.zeros(len(con_np), dtype=np.float32),
            }
    return out


@torch.no_grad()
def eval_zeroshot_forward_mechanics(
    model: PAECModel,
    raw_eps: List[Dict[str, np.ndarray]],
    src_cache: RDPEpisodeCache,
    tgt_cache: RDPEpisodeCache,
    device: torch.device,
    norm_mode: str,
) -> Dict[str, float]:
    """
    Zero-shot 3-step 6D wrench delta prediction on UNSEEN target TEST episodes.
    Predictions are expressed in Target-std units so MSE & R^2 are strictly comparable
    to in-domain target benchmarks.
    """
    wdim = tgt_cache.wrench_dim
    # Scale factor converting predictor output units -> Target-std units
    scale_to_tgt = (
        (src_cache.con_std[:wdim] / tgt_cache.con_std[:wdim]).astype(np.float32)
        if norm_mode == "src_norm"
        else np.ones(wdim, dtype=np.float32)
    )
    res = {}
    for ood_mode in ["clean", "drift"]:
        trues, preds, revs = [], [], []
        for eid in tgt_cache.test_eps:
            ep = tgt_cache.episodes[eid]
            L = ep["len"]
            delta = model.horizons[1] if len(model.horizons) > 1 else model.horizons[0]
            h_idx = 1 if len(model.horizons) > 1 else 0
            if L <= delta + 2:
                continue
            con_in, aex_in, anx_in = normalize_target_episode(raw_eps[eid], src_cache, tgt_cache, norm_mode, ood_mode)
            con_t = torch.from_numpy(con_in).unsqueeze(0).to(device)
            aex_t = torch.from_numpy(aex_in).unsqueeze(0).to(device)
            anx_t = torch.from_numpy(anx_in).unsqueeze(0).to(device)
            z = model.encode(con_t, aex_t)
            a_agg = PAECModel.aggregate_action(anx_t, delta)
            dy_hat, _, _ = model.predictor(z[:, :-delta, :], a_agg, h_idx)

            dw_hat_tgt = dy_hat[0, :, :wdim].cpu().numpy() * scale_to_tgt
            # True physical wrench delta in Target-std units (drift-invariant)
            dw_true_tgt = ep["con"][delta:, :wdim] - ep["con"][:-delta, :wdim]
            trues.append(dw_true_tgt)
            preds.append(dw_hat_tgt)
            revs.append(ep["rev_mask"][:-delta])

        Y = np.concatenate(trues, axis=0)
        P = np.concatenate(preds, axis=0)
        R = np.concatenate(revs, axis=0)
        mse = float(np.mean((Y - P) ** 2))
        r2 = float(1.0 - np.sum((Y - P) ** 2) / (np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2) + 1e-8))
        res[f"dw3_{ood_mode}_mse"] = round(mse, 4)
        res[f"dw3_{ood_mode}_r2"] = round(r2, 4)
        if ood_mode == "clean" and R.sum() >= 15:
            r2_rev = float(
                1.0 - np.sum((Y[R] - P[R]) ** 2) / (np.sum((Y[R] - Y[R].mean(axis=0, keepdims=True)) ** 2) + 1e-8)
            )
            res["dw3_rev_r2"] = round(r2_rev, 4)
    return res


def build_transfer_models(
    contact_dim: int,
    wrench_dim: int,
    act_dim: int,
    device: torch.device,
    d_z: int = 16,
    hidden_dim: int = 96,
    num_layers: int = 2,
) -> Dict[str, nn.Module]:
    return {
        "B2_MAE": MAEContactRecon(contact_dim, act_dim, d_z=d_z, hidden_dim=hidden_dim, num_layers=num_layers, drift_aug_std=0.0).to(device),
        "B2b_MAE_DriftAug": MAEContactRecon(contact_dim, act_dim, d_z=d_z, hidden_dim=hidden_dim, num_layers=num_layers, drift_aug_std=0.35).to(device),
        "B3_LatentJEPA_d16": AbstractLatentJEPAVICReg(contact_dim, act_dim, d_z=d_z, hidden_dim=hidden_dim, num_layers=num_layers, drift_aug_std=0.0).to(device),
        "B3b_LatentJEPA_d16_DriftAug": AbstractLatentJEPAVICReg(contact_dim, act_dim, d_z=d_z, hidden_dim=hidden_dim, num_layers=num_layers, drift_aug_std=0.35).to(device),
        "B3_LatentJEPA_d64": AbstractLatentJEPAVICReg(contact_dim, act_dim, d_z=64, hidden_dim=hidden_dim, num_layers=num_layers, drift_aug_std=0.0).to(device),
        "B4_NoBG": PAECModel(contact_dim, wrench_dim, act_dim, d_z=d_z, hidden_dim=hidden_dim, num_layers=num_layers, use_bg_split=False, drift_aug_std=0.35).to(device),
        "Abl_NoDriftAug": PAECModel(contact_dim, wrench_dim, act_dim, d_z=d_z, hidden_dim=hidden_dim, num_layers=num_layers, use_bg_split=True, drift_aug_std=0.0).to(device),
        "Ours_PAEC": PAECModel(contact_dim, wrench_dim, act_dim, d_z=d_z, hidden_dim=hidden_dim, num_layers=num_layers, use_bg_split=True, drift_aug_std=0.35).to(device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-cache", type=str, required=True)
    parser.add_argument("--tgt-cache", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--src-ckpt", type=str, default="", help="Optional path to pretrained_models.pt from source task")
    parser.add_argument("--save-src-ckpt", type=str, default="", help="Optional path to save freshly pretrained source ckpt")
    parser.add_argument("--pretrain-steps", type=int, default=1200)
    parser.add_argument("--pretrain-seed", type=int, default=42)
    parser.add_argument("--bc-epochs", type=int, default=35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--norm-modes", type=str, nargs="+", default=["src_norm", "tgt_norm"])
    parser.add_argument("--act-norm-mode", type=str, default="rms", choices=["rms", "si_fixed", "zscore_legacy"])
    parser.add_argument("--con-norm-mode", type=str, default="zscore", choices=["zscore", "si_fixed"])
    parser.add_argument("--contact-subset", type=str, default="all", choices=["all", "wrench6d"])
    parser.add_argument("--d-z", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()

    t_start = time.time()
    set_seed(args.pretrain_seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    src_cache = RDPEpisodeCache(
        args.src_cache,
        act_norm_mode=args.act_norm_mode,
        con_norm_mode=args.con_norm_mode,
        contact_subset=args.contact_subset,
    )
    tgt_cache = RDPEpisodeCache(
        args.tgt_cache,
        act_norm_mode=args.act_norm_mode,
        con_norm_mode=args.con_norm_mode,
        contact_subset=args.contact_subset,
    )
    tgt_raw_eps = load_raw_episodes(args.tgt_cache, contact_subset=args.contact_subset)

    # Strict cross-task sensor compatibility audit
    assert src_cache.contact_dim == tgt_cache.contact_dim, f"contact_dim mismatch: {src_cache.contact_dim} vs {tgt_cache.contact_dim}"
    assert src_cache.wrench_dim == tgt_cache.wrench_dim, f"wrench_dim mismatch"
    assert src_cache.act_dim == tgt_cache.act_dim, f"act_dim mismatch"
    if args.contact_subset == "all":
        assert src_cache.meta["arm"] == tgt_cache.meta["arm"], f"arm mismatch"
        assert src_cache.meta["tactile_keys"] == tgt_cache.meta["tactile_keys"], f"tactile_keys mismatch"
    print(
        f"Cross-task channel audit PASS (V4.1, subset={args.contact_subset}): "
        f"{src_cache.meta['source']} -> {tgt_cache.meta['source']} (contact_dim={src_cache.contact_dim})"
    )

    models = build_transfer_models(
        src_cache.contact_dim,
        src_cache.wrench_dim,
        src_cache.act_dim,
        device,
        d_z=args.d_z,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    pretrain_stats = {}
    ckpt_loaded_from = None
    if args.src_ckpt and Path(args.src_ckpt).exists():
        print(f"\n=== [1] Loading Frozen Source Pretrained Checkpoints from {args.src_ckpt} ===")
        ckpt = torch.load(args.src_ckpt, map_location=device, weights_only=False)
        for name, mdl in models.items():
            assert name in ckpt["state_dicts"], f"Missing {name} in checkpoint"
            mdl.load_state_dict(ckpt["state_dicts"][name])
            mdl.eval()
        ckpt_loaded_from = str(args.src_ckpt)
    else:
        print("\n=== [1] Pretraining Matched Models on SOURCE Task TRAIN Episodes ONLY ===")
        for name, mdl in models.items():
            pretrain_stats[name] = pretrain_model(name, mdl, src_cache, device, steps=args.pretrain_steps, seed=args.pretrain_seed)
        if args.save_src_ckpt:
            sp = Path(args.save_src_ckpt)
            sp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "protocol_version": "v4.1",
                    "args": vars(args),
                    "meta": src_cache.meta,
                    "con_mean": src_cache.con_mean,
                    "con_std": src_cache.con_std,
                    "act_center": src_cache.act_center,
                    "act_scale": src_cache.act_scale,
                    "act_mean": src_cache.act_mean,
                    "act_std": src_cache.act_std,
                    "act_rms": src_cache.act_rms,
                    "state_dicts": {k: v.state_dict() for k, v in models.items()},
                },
                sp,
            )
            print(f"Saved source pretrained checkpoints to {sp}")

    # Strictly freeze all source models and snapshot checksums
    checksums_before = {}
    for name, mdl in models.items():
        mdl.eval()
        for p in mdl.parameters():
            p.requires_grad = False
        checksums_before[name] = weight_checksum(mdl)

    # Target task contact threshold & episode budgets
    tr_w = np.concatenate([tgt_cache.episodes[e]["w_norm"] for e in tgt_cache.train_eps])
    w_thresh = float(np.percentile(tr_w, 65.0))
    raw_in_dim = tgt_cache.contact_dim * 2 + tgt_cache.act_dim
    n_tr_tot = len(tgt_cache.train_eps)
    fracs = [0.20, 0.50, 1.00] if n_tr_tot < 25 else [0.10, 0.30, 1.00]

    # Run Target B0_ProprioOnly anchor once (independent of norm_mode)
    print("\n=== [2] Target Anchor: B0_ProprioOnly ===")
    b0_results = {}
    for frac in fracs:
        fk = f"{int(round(frac * 100))}pct"
        n_sub = max(2, min(n_tr_tot, int(round(n_tr_tot * frac))))
        va_d = build_bc_samples(tgt_cache, tgt_cache.val_eps, None, "z", ood_mode="clean", need_raw_win=False)
        te_c = build_bc_samples(tgt_cache, tgt_cache.test_eps, None, "z", ood_mode="clean", need_raw_win=False)
        te_dr = build_bc_samples(tgt_cache, tgt_cache.test_eps, None, "z", ood_mode="drift", need_raw_win=False)
        te_gn = build_bc_samples(tgt_cache, tgt_cache.test_eps, None, "z", ood_mode="gain", need_raw_win=False)
        runs = []
        for sd in args.seeds:
            rng_s = np.random.default_rng(1000 + sd)
            sub_eps = sorted(rng_s.choice(tgt_cache.train_eps, size=n_sub, replace=False).tolist())
            tr_d = build_bc_samples(tgt_cache, sub_eps, None, "z", ood_mode="clean", need_raw_win=False)
            runs.append(
                train_eval_bc(
                    mode="proprio_only",
                    tr_d=tr_d,
                    va_d=va_d,
                    te_clean=te_c,
                    te_drift=te_dr,
                    te_gain=te_gn,
                    w_thresh=w_thresh,
                    raw_contact_dim=raw_in_dim,
                    device=device,
                    seed=sd,
                    epochs=args.bc_epochs,
                )
            )
        agg = {}
        for k in runs[0].keys():
            vals = [x[k] for x in runs]
            agg[f"{k}_mean"] = round(float(np.mean(vals)), 5)
            agg[f"{k}_std"] = round(float(np.std(vals)), 5)
        b0_results[fk] = agg
        print(
            f"  [{fk}] B0_ProprioOnly   | All={agg['test_all_mse_mean']:.5f}±{agg['test_all_mse_std']:.5f} "
            f"| Con={agg['test_contact_mse_mean']:.5f} | Rev={agg['test_rev_mse_mean']:.5f}"
        )

    protocols_out = {}
    for norm_mode in args.norm_modes:
        print(f"\n======================================================================")
        print(f"=== Zero-Shot Transfer Evaluation Protocol: {norm_mode} ===")
        print(f"======================================================================")
        feats_tr, feats_va, feats_te, feats_drift, feats_gain = {}, {}, {}, {}, {}
        geom_and_mech = {}
        for name, mdl in models.items():
            feats_tr[name] = extract_transfer_features(mdl, tgt_raw_eps, src_cache, tgt_cache, tgt_cache.train_eps, device, norm_mode, "clean")
            feats_va[name] = extract_transfer_features(mdl, tgt_raw_eps, src_cache, tgt_cache, tgt_cache.val_eps, device, norm_mode, "clean")
            feats_te[name] = extract_transfer_features(mdl, tgt_raw_eps, src_cache, tgt_cache, tgt_cache.test_eps, device, norm_mode, "clean")
            feats_drift[name] = extract_transfer_features(mdl, tgt_raw_eps, src_cache, tgt_cache, tgt_cache.test_eps, device, norm_mode, "drift")
            feats_gain[name] = extract_transfer_features(mdl, tgt_raw_eps, src_cache, tgt_cache, tgt_cache.test_eps, device, norm_mode, "gain")

            Z_te = np.concatenate([feats_te[name][e]["z"] for e in tgt_cache.test_eps], axis=0)
            info = {"test_eff_rank": round(compute_effective_rank(torch.from_numpy(Z_te)), 2), "d_z": int(Z_te.shape[1])}
            if isinstance(mdl, PAECModel):
                info.update(eval_zeroshot_forward_mechanics(mdl, tgt_raw_eps, src_cache, tgt_cache, device, norm_mode))
            geom_and_mech[name] = info
            print(f"  [{norm_mode} | {name:16s}] {info}")

        codex_test = eval_codex_wrench_matching(feats_te, tgt_cache)
        print(f"  [{norm_mode}] Codex history separation: {json.dumps(codex_test)}")

        bc_specs = [
            ("Xfer_B2_MAE", "B2_MAE", "z"),
            ("Xfer_B2b_MAE_Drift", "B2b_MAE_DriftAug", "z"),
            ("Xfer_B3_JEPA_d16", "B3_LatentJEPA_d16", "z"),
            ("Xfer_B3b_JEPA_d16_Dr", "B3b_LatentJEPA_d16_DriftAug", "z"),
            ("Xfer_B3_JEPA_d64", "B3_LatentJEPA_d64", "z"),
            ("Xfer_B4_NoBG", "B4_NoBG", "z"),
            ("Xfer_Abl_NoDriftAug", "Abl_NoDriftAug", "z"),
            ("Xfer_PAEC_z", "Ours_PAEC", "z"),
            ("Xfer_PAEC_Probe", "Ours_PAEC", "z_probe"),
        ]

        bc_results = {}
        for frac in fracs:
            fk = f"{int(round(frac * 100))}pct"
            n_sub = max(2, min(n_tr_tot, int(round(n_tr_tot * frac))))
            bc_results[fk] = {"B0_ProprioOnly": b0_results[fk]}
            print(f"\n--- [{norm_mode}] Budget: {fk} ({n_sub}/{n_tr_tot} target train episodes) ---")
            for label, mkey, fkey in bc_specs:
                va_d = build_bc_samples(tgt_cache, tgt_cache.val_eps, feats_va[mkey], fkey, ood_mode="clean", need_raw_win=False)
                te_c = build_bc_samples(tgt_cache, tgt_cache.test_eps, feats_te[mkey], fkey, ood_mode="clean", need_raw_win=False)
                te_dr = build_bc_samples(tgt_cache, tgt_cache.test_eps, feats_drift[mkey], fkey, ood_mode="drift", need_raw_win=False)
                te_gn = build_bc_samples(tgt_cache, tgt_cache.test_eps, feats_gain[mkey], fkey, ood_mode="gain", need_raw_win=False)

                runs = []
                for sd in args.seeds:
                    rng_s = np.random.default_rng(1000 + sd)
                    sub_eps = sorted(rng_s.choice(tgt_cache.train_eps, size=n_sub, replace=False).tolist())
                    tr_d = build_bc_samples(tgt_cache, sub_eps, feats_tr[mkey], fkey, ood_mode="clean", need_raw_win=False)
                    runs.append(
                        train_eval_bc(
                            mode="frozen_feat",
                            tr_d=tr_d,
                            va_d=va_d,
                            te_clean=te_c,
                            te_drift=te_dr,
                            te_gain=te_gn,
                            w_thresh=w_thresh,
                            raw_contact_dim=raw_in_dim,
                            device=device,
                            seed=sd,
                            epochs=args.bc_epochs,
                        )
                    )
                agg = {}
                for k in runs[0].keys():
                    vals = [x[k] for x in runs]
                    agg[f"{k}_mean"] = round(float(np.mean(vals)), 5)
                    agg[f"{k}_std"] = round(float(np.std(vals)), 5)
                bc_results[fk][label] = agg
                print(
                    f"  [{norm_mode}|{fk}] {label:19s} | All={agg['test_all_mse_mean']:.5f}±{agg['test_all_mse_std']:.5f} "
                    f"| Con={agg['test_contact_mse_mean']:.5f} | Rev={agg['test_rev_mse_mean']:.5f} "
                    f"| DriftOOD={agg['test_drift_ood_mse_mean']:.5f} | GainOOD={agg['test_gain_ood_mse_mean']:.5f}"
                )

        protocols_out[norm_mode] = {
            "geom_and_mech": geom_and_mech,
            "codex_history_test": codex_test,
            "downstream_lowshot_bc": bc_results,
        }

    # Verify frozen encoder weights were strictly untouched
    checksums_after = {name: weight_checksum(mdl) for name, mdl in models.items()}
    frozen_audit_pass = (checksums_before == checksums_after)
    assert frozen_audit_pass, f"Frozen audit failed! {checksums_before} vs {checksums_after}"

    summary = {
        "protocol_version": "v4.1",
        "args": vars(args),
        "src_meta": src_cache.meta,
        "tgt_meta": tgt_cache.meta,
        "ckpt_loaded_from": ckpt_loaded_from,
        "pretrain_stats": pretrain_stats,
        "frozen_audit": {
            "checksums": checksums_before,
            "strictly_frozen_pass": frozen_audit_pass,
        },
        "protocols": protocols_out,
        "total_sec": round(time.time() - t_start, 1),
    }
    out_json = out_dir / "v4_cross_task_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved cross-task zero-shot transfer results to {out_json} (total {summary['total_sec']}s)")


if __name__ == "__main__":
    main()
