#!/usr/bin/env python3
"""
V4.7 Industrial 3-Task ACT + Visual-Tactile Defect-Inspection Suite.

Evaluates 5 matched ACT variants (with frozen ImageNet ResNet18 spatial features
and O(1) recurrent GRU tactile encoding) + Classical Industrial Baseline across 3
MuJoCo industrial defect-inspection tasks:
  1. battery_scan   (EV Battery Cell: normal / soft swelling / weld burr, limit 11.5 N)
  2. auto_connector (Automotive Plug Insertion: normal / internal burr jam, limit 15.0 N)
  3. semi_grasp     (Semiconductor Fragile Carrier Micro-Force Transfer, limit 14.0 N)

Strictly FP32 (allow_tf32=False), SHA-256 frozen-weight audit, Clean & +1.5sigma Drift OOD,
online within-episode defect detection AUROC, and batch=1 GPU/CPU P50/P95/P99 latency benchmark.
"""

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from act_policy import (
    ACTPolicy,
    FrozenResNet18Extractor,
    TemporalEnsembler,
    step_encode_recurrent,
)
from envs_industrial_defect import make_env
from models_paec import PAECModel
from mujoco_contact_bench import build_policy_model_for_key, sha256_model
from run_v4_scaling_suite import CANON_ACT_SCALE, CANON_WRENCH_SCALE


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score, kind="mergesort")
    sorted_s = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    i = 0
    n = len(y_score)
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    u = float(ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0)
    return float(u / (n_pos * n_neg))


@torch.no_grad()
def verify_recurrent_equivalence(paec: PAECModel, device: torch.device):
    """Assert O(1) step_encode_recurrent matches full sequence encode to < 1e-5."""
    w_seq = torch.randn(2, 24, 6, device=device)
    pa_seq = torch.randn(2, 24, 7, device=device)
    z_full = paec.encode(w_seq, pa_seq)
    h = None
    z_steps = []
    for t in range(24):
        w_prev = None if t == 0 else w_seq[:, t - 1, :]
        z_t, h = step_encode_recurrent(paec, w_seq[:, t, :], w_prev, pa_seq[:, t, :], h)
        z_steps.append(z_t)
    z_rec = torch.stack(z_steps, dim=1)
    max_diff = float((z_full - z_rec).abs().max().item())
    assert max_diff < 1e-5, f"Recurrent GRU equivalence failed: max_diff={max_diff}"
    print(f"[Audit] O(1) recurrent GRU max_abs_diff={max_diff:.2e} (< 1e-5 PASS)", flush=True)


@torch.no_grad()
def collect_task_demos(
    task_name: str,
    n_demos: int,
    vis_extractor: FrozenResNet18Extractor,
    device: torch.device,
    seed: int = 2027,
) -> List[Dict[str, Any]]:
    env = make_env(task_name, rgb_w=64, rgb_h=64)
    rng = np.random.default_rng(seed)
    rng_vis = np.random.default_rng(seed + 404)
    demos = []
    try:
        for eid in range(n_demos):
            cfg = env.reset(rng)
            coarse_vis_xy = env.get_coarse_vis_xy(rng_vis)
            w_cur = env.last_w_avg.copy()
            imgs, ws, ps, acts, fzs = [], [], [], [], []
            extra: Dict[str, Any] = {"depths": [], "finger_forces": []}
            for t in range(env.MAX_STEPS):
                imgs.append(env.render_rgb(64, 64))
                p_cur = env.get_proprio10d()
                act = env.expert_policy(w_cur, t)
                ws.append(w_cur)
                ps.append(p_cur)
                acts.append(act)
                w_cur, p_next, sig = env.step_action(act)
                fzs.append(float(w_cur[2]))
                if task_name == "auto_connector":
                    extra["depths"].append(max(0.0, env.SOCKET_TOP_Z - float(p_next[2])))
                elif task_name == "semi_grasp":
                    extra["finger_forces"].append(float(sig))
            ev = env.evaluate_episode(fzs, extra)
            img_np = np.stack(imgs, axis=0)
            img_t = torch.from_numpy(img_np).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255.0
            vis_feat = vis_extractor(img_t).cpu().numpy().astype(np.float32)
            demos.append(
                {
                    "ep_id": eid,
                    "cfg": cfg,
                    "coarse_vis_xy": coarse_vis_xy,
                    "vis_feat": vis_feat,
                    "contact_obs": np.stack(ws, axis=0).astype(np.float32),
                    "proprio_state": np.stack(ps, axis=0).astype(np.float32),
                    "action_next": np.stack(acts, axis=0).astype(np.float32),
                    "eval": ev,
                }
            )
    finally:
        env._close_renderer()
    return demos


def prepare_act_dataset(
    demos: List[Dict[str, Any]],
    mode: str,
    encoder: Optional[nn.Module],
    device: torch.device,
    chunk_len: int = 8,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    vfeats_l, props_l, feats_l, chunks_l = [], [], [], []

    for dm in demos:
        T = dm["contact_obs"].shape[0]
        w_scaled = (dm["contact_obs"] / CANON_WRENCH_SCALE[None, :]).astype(np.float32)
        p_raw = dm["proprio_state"]
        p_rel = p_raw.copy()
        p_rel[:, :3] = (p_raw[:, :3] - p_raw[0:1, :3]) / 0.12
        a_scaled = (dm["action_next"] / CANON_ACT_SCALE[None, :]).astype(np.float32)
        prev_a = np.concatenate([np.zeros((1, 7), dtype=np.float32), a_scaled[:-1]], axis=0)
        vis_xy_rel = np.repeat(((dm["coarse_vis_xy"] - p_raw[0, :2]) / 0.12)[None, :], T, axis=0).astype(np.float32)
        prop = np.concatenate([p_rel, prev_a, vis_xy_rel], axis=-1).astype(np.float32)  # 19D

        if mode == "ACT_RGB_RawFT":
            c_seq = w_scaled
        elif encoder is not None:
            encoder.eval()
            with torch.no_grad():
                wt = torch.from_numpy(w_scaled).unsqueeze(0).to(device)
                pt = torch.from_numpy(prev_a).unsqueeze(0).to(device)
                c_seq = encoder.encode(wt, pt).squeeze(0).cpu().numpy().astype(np.float32)
        else:
            c_seq = None

        pad = np.repeat(a_scaled[-1:], chunk_len, axis=0)
        a_pad = np.concatenate([a_scaled, pad], axis=0)

        for t in range(T):
            vfeats_l.append(dm["vis_feat"][t])
            props_l.append(prop[t])
            chunks_l.append(a_pad[t : t + chunk_len])
            if c_seq is not None:
                feats_l.append(c_seq[t])

    vfeats_np = np.stack(vfeats_l, axis=0).astype(np.float32)
    props_np = np.stack(props_l, axis=0).astype(np.float32)
    chunks_np = np.stack(chunks_l, axis=0).astype(np.float32)

    p_mean = props_np.mean(axis=0).astype(np.float32)
    p_std = np.maximum(props_np.std(axis=0), 1e-3).astype(np.float32)
    props_np = ((props_np - p_mean) / p_std).astype(np.float32)

    data: Dict[str, np.ndarray] = {"vis": vfeats_np, "prop": props_np, "chunk": chunks_np}
    c_mean, c_std = None, None
    if feats_l:
        feats_np = np.stack(feats_l, axis=0).astype(np.float32)
        c_mean = feats_np.mean(axis=0).astype(np.float32)
        rms = float(np.sqrt(np.mean(feats_np.var(axis=0))))
        c_std = np.full(feats_np.shape[1], max(rms, 0.25), dtype=np.float32)
        data["contact"] = ((feats_np - c_mean) / c_std).astype(np.float32)
    return data, p_mean, p_std, c_mean, c_std


def train_act_policy(
    demos: List[Dict[str, Any]],
    mode: str,
    encoder: Optional[nn.Module],
    device: torch.device,
    seed: int = 0,
    num_steps: int = 1200,
    batch_size: int = 128,
    lr: float = 8e-4,
) -> Tuple[ACTPolicy, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data, p_mean, p_std, c_mean, c_std = prepare_act_dataset(demos, mode, encoder, device, chunk_len=8)
    c_dim = data["contact"].shape[1] if "contact" in data else 0

    policy = ACTPolicy(
        prop_dim=data["prop"].shape[1],
        contact_dim=c_dim,
        act_dim=7,
        chunk_len=8,
        d_model=96,
        nhead=4,
        num_enc_layers=1,
        num_dec_layers=2,
        d_style=16,
    ).to(device)

    vis_t = torch.from_numpy(data["vis"]).to(device)
    prop_t = torch.from_numpy(data["prop"]).to(device)
    chunk_t = torch.from_numpy(data["chunk"]).to(device)
    con_t = torch.from_numpy(data["contact"]).to(device) if c_dim > 0 else None

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps, eta_min=lr * 0.1)

    N = len(vis_t)
    bs = min(batch_size, N)
    policy.train()
    for _ in range(num_steps):
        idx = torch.randint(0, N, (bs,), device=device)
        cf = con_t[idx] if con_t is not None else None
        loss, _ = policy.compute_loss(vis_t[idx], prop_t[idx], chunk_t[idx], contact_feat=cf, kl_weight=0.2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
        opt.step()
        sched.step()
    policy.eval()
    return policy, p_mean, p_std, c_mean, c_std


def evaluate_act_or_classical(
    task_name: str,
    policy: Optional[ACTPolicy],
    vis_extractor: Optional[FrozenResNet18Extractor],
    p_mean: Optional[np.ndarray],
    p_std: Optional[np.ndarray],
    c_mean: Optional[np.ndarray],
    c_std: Optional[np.ndarray],
    mode: str,
    encoder: Optional[nn.Module],
    device: torch.device,
    n_eval: int = 16,
    eval_seed: int = 9200,
    drift_sigma: float = 0.0,
) -> Dict[str, Any]:
    env = make_env(task_name, rgb_w=64, rgb_h=64)
    rng = np.random.default_rng(eval_seed)
    rng_vis = np.random.default_rng(eval_seed + 404)
    drift_rng = np.random.default_rng(eval_seed + 777)

    succs, viols, peaks = [], [], []
    ep_labels, ep_types = [], []
    ep_rel_surp, ep_stiff_std, ep_raw_norm_max, ep_raw_rate_max = [], [], [], []

    if policy is not None:
        assert vis_extractor is not None
        pm_t = torch.from_numpy(p_mean).to(device)
        ps_t = torch.from_numpy(p_std).to(device)
        cm_t = torch.from_numpy(c_mean).to(device) if c_mean is not None else None
        cs_t = torch.from_numpy(c_std).to(device) if c_std is not None else None
        ensembler = TemporalEnsembler(chunk_len=8, act_dim=7, k_exp=0.25)

    try:
        for _ in range(n_eval):
            cfg = env.reset(rng)
            coarse_vis_xy = env.get_coarse_vis_xy(rng_vis)
            drift_vec = (
                drift_sigma * drift_rng.choice([-1.0, 1.0], size=6).astype(np.float32) * CANON_WRENCH_SCALE
            ) if drift_sigma > 0.0 else np.zeros(6, dtype=np.float32)

            p0 = env.get_proprio10d()[:3].copy()
            vis_xy_rel = ((coarse_vis_xy - p0[:2]) / 0.12).astype(np.float32)
            prev_a_scaled = np.zeros((1, 7), dtype=np.float32)
            fz_hist = []
            extra: Dict[str, Any] = {"depths": [], "finger_forces": []}
            surp_steps, kz_steps, rnorm_steps, rrate_steps = [], [], [], []

            h_gru: Optional[torch.Tensor] = None
            w_prev_t: Optional[torch.Tensor] = None
            v_t: Optional[torch.Tensor] = None

            if policy is not None:
                ensembler.reset()

            for t in range(env.MAX_STEPS):
                w_obs = env.last_w_avg + drift_vec
                w_scaled = (w_obs / CANON_WRENCH_SCALE).astype(np.float32)

                if mode == "Classical_Baseline":
                    act_canon = env.classical_policy(w_obs, t)
                    z_pred_dw = None
                else:
                    assert policy is not None and vis_extractor is not None
                    if t % 2 == 0 or v_t is None:
                        rgb_u8 = env.render_rgb(64, 64)
                        img_t = torch.from_numpy(rgb_u8).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
                        with torch.no_grad():
                            v_t = vis_extractor(img_t)

                    p_cur = env.get_proprio10d()
                    p_rel = p_cur.copy()
                    p_rel[:3] = (p_cur[:3] - p0) / 0.12
                    prop = np.concatenate([p_rel, prev_a_scaled[0], vis_xy_rel], axis=-1).astype(np.float32)
                    prop_t = ((torch.from_numpy(prop).to(device) - pm_t) / ps_t).unsqueeze(0)

                    if mode == "ACT_RGB_Proprio":
                        cf_t = None
                        z_cur_t = None
                    elif mode == "ACT_RGB_RawFT":
                        assert cm_t is not None and cs_t is not None
                        cf_t = ((torch.from_numpy(w_scaled).to(device) - cm_t) / cs_t).unsqueeze(0)
                        z_cur_t = None
                    else:
                        assert encoder is not None and cm_t is not None and cs_t is not None
                        w_cur_t = torch.from_numpy(w_scaled).unsqueeze(0).to(device)
                        pa_cur_t = torch.from_numpy(prev_a_scaled).to(device)
                        z_cur_t, h_gru = step_encode_recurrent(encoder, w_cur_t, w_prev_t, pa_cur_t, h_gru)
                        w_prev_t = w_cur_t
                        cf_t = ((z_cur_t.squeeze(0) - cm_t) / cs_t).unsqueeze(0)
                        if isinstance(encoder, PAECModel):
                            with torch.no_grad():
                                p18 = encoder.compute_virtual_probe(z_cur_t, step_scale=1.0).squeeze(0).cpu().numpy()
                                kz_steps.append(float(0.5 * (p18[14] - p18[17])))

                    with torch.no_grad():
                        chunk_t, _, _ = policy(v_t, prop_t, contact_feat=cf_t, act_chunk=None)
                        chunk_np = chunk_t.squeeze(0).cpu().numpy()
                    a_exec_scaled = ensembler.step(chunk_np)
                    act_canon = (a_exec_scaled * CANON_ACT_SCALE).astype(np.float32)

                    if isinstance(encoder, PAECModel) and z_cur_t is not None:
                        with torch.no_grad():
                            at_t = torch.from_numpy(a_exec_scaled).unsqueeze(0).to(device)
                            z_pred_dw = encoder.predictor(z_cur_t, at_t, h_idx=0)[0][:, :6].squeeze(0).cpu().numpy()
                    else:
                        z_pred_dw = None
                    prev_a_scaled = a_exec_scaled[None, :].astype(np.float32)

                w_next_raw, p_next, sig = env.step_action(act_canon)
                w_next_scaled = ((w_next_raw + drift_vec) / CANON_WRENCH_SCALE).astype(np.float32)
                dw_true = w_next_scaled - w_scaled

                if t >= 10:
                    rnorm_steps.append(float(np.linalg.norm(w_next_scaled)))
                    rrate_steps.append(float(np.linalg.norm(dw_true)))
                    if z_pred_dw is not None:
                        surp_steps.append(float(np.linalg.norm(dw_true - z_pred_dw)))

                fz_hist.append(float(w_next_raw[2]))
                if task_name == "auto_connector":
                    extra["depths"].append(max(0.0, env.SOCKET_TOP_Z - float(p_next[2])))
                elif task_name == "semi_grasp":
                    extra["finger_forces"].append(float(sig))

            ev = env.evaluate_episode(fz_hist, extra)
            succs.append(float(ev["success"]))
            viols.append(float(ev["safety_viol"]))
            peaks.append(float(ev["peak_fn"]))

            ep_labels.append(int(cfg["is_defect"]))
            ep_types.append(str(cfg["defect_type"]))
            ep_raw_norm_max.append(float(np.max(rnorm_steps)) if rnorm_steps else 0.0)
            ep_raw_rate_max.append(float(np.max(rrate_steps)) if rrate_steps else 0.0)
            if surp_steps:
                s_arr = np.asarray(surp_steps, dtype=np.float64)
                ep_rel_surp.append(float(np.max(s_arr) - np.median(s_arr)))
            if kz_steps:
                kz_arr = np.asarray(kz_steps, dtype=np.float64)
                ep_stiff_std.append(float(np.std(kz_arr)))
    finally:
        env._close_renderer()

    out: Dict[str, Any] = {
        "success_rate": round(100.0 * float(np.mean(succs)), 2),
        "safety_viol_rate": round(100.0 * float(np.mean(viols)), 2),
        "peak_fn_n": round(float(np.mean(peaks)), 3),
    }
    y = np.asarray(ep_labels, dtype=np.int64)
    if len(set(y.tolist())) == 2:
        out["raw_norm_max_auroc"] = round(float(_roc_auc(y, np.asarray(ep_raw_norm_max))), 4)
        out["raw_rate_max_auroc"] = round(float(_roc_auc(y, np.asarray(ep_raw_rate_max))), 4)
        if ep_rel_surp:
            out["paec_rel_surprise_auroc"] = round(float(_roc_auc(y, np.asarray(ep_rel_surp))), 4)
        if ep_stiff_std:
            out["paec_stiff_contrast_auroc"] = round(float(_roc_auc(y, np.asarray(ep_stiff_std))), 4)
            if task_name == "battery_scan":
                idx_sn = [i for i, dt in enumerate(ep_types) if dt in ("normal", "swelling")]
                if len(idx_sn) >= 4 and len(set(y[idx_sn].tolist())) == 2:
                    out["paec_stiff_swelling_auroc"] = round(
                        float(_roc_auc(y[idx_sn], np.asarray(ep_stiff_std)[idx_sn])), 4
                    )
                    out["raw_norm_swelling_auroc"] = round(
                        float(_roc_auc(y[idx_sn], np.asarray(ep_raw_norm_max)[idx_sn])), 4
                    )
    return out


def _agg_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for k in runs[0].keys():
        vals = [r[k] for r in runs if r[k] is not None]
        if vals:
            agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{k}_std"] = round(float(np.std(vals)), 4)
    return agg


@torch.no_grad()
def benchmark_edge_latency(
    paec: PAECModel,
    vis_extractor: FrozenResNet18Extractor,
    device_gpu: torch.device,
    n_iters: int = 500,
) -> Dict[str, Any]:
    """Measure full end-to-end batch=1 (FrozenResNet18 + O(1) PAEC + ACTPolicy) latency."""
    act_paec = ACTPolicy(prop_dim=19, contact_dim=32, act_dim=7, chunk_len=8, d_model=96, nhead=4, num_enc_layers=1, num_dec_layers=2).eval()
    act_raw = ACTPolicy(prop_dim=19, contact_dim=6, act_dim=7, chunk_len=8, d_model=96, nhead=4, num_enc_layers=1, num_dec_layers=2).eval()
    res: Dict[str, Any] = {
        "paec_params": int(sum(p.numel() for p in paec.parameters())),
        "resnet18_layer3_params": int(sum(p.numel() for p in vis_extractor.parameters())),
        "act_paec_params": int(sum(p.numel() for p in act_paec.parameters())),
        "act_raw_params": int(sum(p.numel() for p in act_raw.parameters())),
    }

    for dev_name, dev in [("gpu", device_gpu), ("cpu", torch.device("cpu"))]:
        if dev_name == "gpu" and dev.type != "cuda":
            continue
        p_m = copy.deepcopy(paec).to(dev).eval()
        v_m = copy.deepcopy(vis_extractor).to(dev).eval()
        a_m = copy.deepcopy(act_paec).to(dev).eval()
        w_t = torch.randn(1, 6, device=dev)
        w_prev = torch.randn(1, 6, device=dev)
        pa_t = torch.randn(1, 7, device=dev)
        h = torch.zeros(p_m.num_layers, 1, p_m.hidden_dim, device=dev)
        img_in = torch.rand(1, 3, 64, 64, device=dev)
        prop_in = torch.randn(1, 19, device=dev)

        for _ in range(30):
            vf = v_m(img_in)
            z, h = step_encode_recurrent(p_m, w_t, w_prev, pa_t, h)
            _ = a_m(vf, prop_in, contact_feat=z)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            vf = v_m(img_in)
            z, h = step_encode_recurrent(p_m, w_t, w_prev, pa_t, h)
            _ = a_m(vf, prop_in, contact_feat=z)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.asarray(times_ms, dtype=np.float64)
        res[dev_name] = {
            "mean_ms": round(float(arr.mean()), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "max_hz": round(1000.0 / max(float(np.percentile(arr, 95)), 1e-4), 1),
        }
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--baselines-ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--quick-test", action="store_true", help="Fast 1-seed smoke benchmark")
    parser.add_argument("--budgets", type=int, nargs="+", default=[15])
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=12)
    args = parser.parse_args()

    t_all = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    budgets = [10] if args.quick_test else list(args.budgets)
    seeds = [0] if args.quick_test else [0, 1, 2]
    n_eval = 8 if args.quick_test else args.n_eval
    num_steps = args.num_steps
    tasks = ["battery_scan", "auto_connector", "semi_grasp"]
    regimes = [("clean", 0.0), ("thermal_drift_0p5", 0.5), ("extreme_drift_1p5", 1.5)]

    print(
        f"[V4.7] Device={device} | quick={args.quick_test} | budgets={budgets} | seeds={seeds} | "
        f"n_eval={n_eval} | num_steps={num_steps} | regimes={[r[0] for r in regimes]}",
        flush=True,
    )

    vis_extractor = FrozenResNet18Extractor(device)

    encoders: Dict[str, Optional[nn.Module]] = {
        "ACT_RGB_Proprio": None,
        "ACT_RGB_RawFT": None,
    }
    pre_hashes: Dict[str, str] = {"FrozenResNet18": sha256_model(vis_extractor)}
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m_paec = build_policy_model_for_key("PAEC_Small_d32", device)
    m_paec.load_state_dict(ckpt["state_dicts"]["PAEC_Small_d32"])
    m_paec.eval()
    for p in m_paec.parameters():
        p.requires_grad = False
    verify_recurrent_equivalence(m_paec, device)
    encoders["ACT_RGB_PAEC"] = m_paec
    pre_hashes["ACT_RGB_PAEC"] = sha256_model(m_paec)

    bckpt = torch.load(args.baselines_ckpt, map_location="cpu", weights_only=False)
    for mode_k, bk in [("ACT_RGB_MAE", "B2b_MAE_Small_d16"), ("ACT_RGB_JEPA", "B3b_JEPA_Small_d16")]:
        bm = build_policy_model_for_key(bk, device)
        bm.load_state_dict(bckpt[bk])
        bm.eval()
        for p in bm.parameters():
            p.requires_grad = False
        encoders[mode_k] = bm
        pre_hashes[mode_k] = sha256_model(bm)

    results: Dict[str, Any] = {
        "protocol": "v4_7_act_3task_resnet18_industrial_defect",
        "strict_fp32": True,
        "strictly_frozen_pass": True,
        "quick_test": bool(args.quick_test),
        "pre_hashes": pre_hashes,
        "tasks": {},
    }

    for tname in tasks:
        print(f"\n==================== TASK: {tname} ====================", flush=True)
        demos = collect_task_demos(tname, max(budgets), vis_extractor, device, seed=2027)
        exp_succ = 100.0 * float(np.mean([d["eval"]["success"] for d in demos]))
        exp_viol = 100.0 * float(np.mean([d["eval"]["safety_viol"] for d in demos]))
        print(f"  [Expert Demos N={len(demos)}] Succ={exp_succ:.1f}% | Viol={exp_viol:.1f}%", flush=True)

        t_res: Dict[str, Any] = {
            "expert_demos": {"n": len(demos), "success_rate": round(exp_succ, 2), "safety_viol_rate": round(exp_viol, 2)},
            "classical_baseline": {},
            "act_methods": {},
        }

        t_res["classical_baseline"] = {}
        for rname, d_sig in regimes:
            cls_runs = [
                evaluate_act_or_classical(
                    tname, None, None, None, None, None, None, "Classical_Baseline",
                    None, device, n_eval=n_eval, eval_seed=es, drift_sigma=d_sig
                )
                for es in (9200, 9201, 9202)
            ]
            t_res["classical_baseline"][rname] = _agg_runs(cls_runs)
        print(
            f"  [Classical_Baseline ] Clean={t_res['classical_baseline']['clean']['success_rate_mean']:5.1f}% "
            f"(Viol={t_res['classical_baseline']['clean']['safety_viol_rate_mean']:4.1f}%) | "
            f"0.5σ={t_res['classical_baseline']['thermal_drift_0p5']['success_rate_mean']:5.1f}% | "
            f"1.5σ={t_res['classical_baseline']['extreme_drift_1p5']['success_rate_mean']:5.1f}%",
            flush=True,
        )

        for mode in ("ACT_RGB_Proprio", "ACT_RGB_RawFT", "ACT_RGB_MAE", "ACT_RGB_JEPA", "ACT_RGB_PAEC"):
            enc = encoders[mode]
            t_res["act_methods"][mode] = {}
            for nb in budgets:
                sub_demos = demos[:nb]
                reg_runs: Dict[str, List[Dict[str, Any]]] = {r[0]: [] for r in regimes}
                for sd in seeds:
                    pol, pm, ps, cm, cs = train_act_policy(sub_demos, mode, enc, device, seed=sd, num_steps=num_steps, batch_size=128, lr=8e-4)
                    for rname, d_sig in regimes:
                        reg_runs[rname].append(
                            evaluate_act_or_classical(
                                tname, pol, vis_extractor, pm, ps, cm, cs, mode, enc,
                                device, n_eval=n_eval, eval_seed=9200, drift_sigma=d_sig
                            )
                        )
                t_res["act_methods"][mode][f"{nb}demos"] = {
                    rname: _agg_runs(reg_runs[rname]) for rname, _ in regimes
                }
                cm_res = t_res["act_methods"][mode][f"{nb}demos"]["clean"]
                t_res_05 = t_res["act_methods"][mode][f"{nb}demos"]["thermal_drift_0p5"]
                d_res_15 = t_res["act_methods"][mode][f"{nb}demos"]["extreme_drift_1p5"]
                print(
                    f"  [{mode:16s} | {nb:2d}d] Clean={cm_res['success_rate_mean']:5.1f}% (Viol={cm_res['safety_viol_rate_mean']:4.1f}%) | "
                    f"0.5σ={t_res_05['success_rate_mean']:5.1f}% (Viol={t_res_05['safety_viol_rate_mean']:4.1f}%) | "
                    f"1.5σ={d_res_15['success_rate_mean']:5.1f}% (Viol={d_res_15['safety_viol_rate_mean']:4.1f}%)",
                    flush=True,
                )

        results["tasks"][tname] = t_res
        with open(out_dir / "v4_7_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Measuring Edge Latency (batch=1, ResNet18 + O(1) PAEC + ACT) ===", flush=True)
    results["edge_latency"] = benchmark_edge_latency(m_paec, vis_extractor, device, n_iters=200 if args.quick_test else 500)
    print("  Latency:", results["edge_latency"], flush=True)

    # Verify frozen SHA-256 hashes (ResNet18 + tactile encoders)
    assert sha256_model(vis_extractor) == pre_hashes["FrozenResNet18"], "FrozenResNet18 hash mismatch!"
    for k, h_before in pre_hashes.items():
        if k == "FrozenResNet18":
            continue
        if sha256_model(encoders[k]) != h_before:
            raise RuntimeError(f"Frozen weight mutation detected for {k}!")

    results["total_wall_sec"] = round(time.time() - t_all, 1)
    with open(out_dir / "v4_7_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[V4.7] COMPLETE in {results['total_wall_sec']}s -> {out_dir / 'v4_7_results.json'}", flush=True)


if __name__ == "__main__":
    main()
