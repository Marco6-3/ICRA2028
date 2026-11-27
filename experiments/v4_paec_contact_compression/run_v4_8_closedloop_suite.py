#!/usr/bin/env python3
"""
V4.8 Script 2: Failure Anatomy, Missing Fair Baselines (NoBG & RawHistGRU),
Chamfered Task 2 Root-Cause Isolation, and Task 3 Hierarchical FSM + Contact Primitive.

Strictly FP32 (allow_tf32=False), SHA-256 frozen audit, NO downstream drift augmentation,
15 demos x 3 seeds (0, 1, 2) x n_eval=24 paired episodes (72 rollouts per cell).
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
import torch.nn.functional as F

from act_policy import (
    ACTPolicy,
    FrozenResNet18Extractor,
    TemporalEnsembler,
    step_encode_recurrent,
)
from envs_industrial_defect import make_env
from envs_v4_8_diagnostic import AutoConnectorChamferEnv
from models_paec import CausalContactBackbone, PAECModel, build_causal_contact_input
from mujoco_contact_bench import build_policy_model_for_key, sha256_model
from run_v4_7_act_defect_suite import (
    _agg_runs,
    _roc_auc,
    collect_task_demos,
    prepare_act_dataset,
)
from run_v4_scaling_suite import CANON_ACT_SCALE, CANON_WRENCH_SCALE


@torch.no_grad()
def verify_recurrent_equivalence(model: nn.Module, name: str, device: torch.device) -> float:
    model.eval()
    w_seq = torch.randn(2, 24, 6, device=device)
    pa_seq = torch.randn(2, 24, 7, device=device)
    if isinstance(model, CausalContactBackbone):
        x_seq = build_causal_contact_input(w_seq, pa_seq, use_act_exec=True, use_causal_diff=True, use_abs_wrench=True)
        z_full = model(x_seq)
    else:
        z_full = model.encode(w_seq, pa_seq)
    h = None
    z_steps = []
    for t in range(24):
        w_prev = None if t == 0 else w_seq[:, t - 1, :]
        z_t, h = step_encode_recurrent(model, w_seq[:, t, :], w_prev, pa_seq[:, t, :], h)
        z_steps.append(z_t)
    z_rec = torch.stack(z_steps, dim=1)
    max_diff = float((z_full - z_rec).abs().max().item())
    assert max_diff < 1e-5, f"O(1) recurrent audit failed for {name}: {max_diff}"
    print(f"  [Audit] O(1) recurrent {name:26s}: max_abs_diff={max_diff:.2e} (< 1e-5 PASS)", flush=True)
    return max_diff


@torch.no_grad()
def collect_chamfer_demos(
    n_demos: int,
    vis_extractor: FrozenResNet18Extractor,
    device: torch.device,
    seed: int = 2027,
) -> List[Dict[str, Any]]:
    env = AutoConnectorChamferEnv(rgb_w=64, rgb_h=64)
    rng = np.random.default_rng(seed)
    rng_vis = np.random.default_rng(seed + 404)
    demos = []
    try:
        for eid in range(n_demos):
            cfg = env.reset(rng)
            coarse_vis_xy = env.get_coarse_vis_xy(rng_vis)
            w_cur = env.last_w_avg.copy()
            imgs, ws, ps, acts, fzs = [], [], [], [], []
            extra: Dict[str, Any] = {"depths": []}
            for t in range(env.MAX_STEPS):
                imgs.append(env.render_rgb(64, 64))
                p_cur = env.get_proprio10d()
                act = env.expert_policy_no_oracle(w_cur, t, coarse_vis_xy)
                ws.append(w_cur)
                ps.append(p_cur)
                acts.append(act)
                w_cur, p_next, _ = env.step_action(act)
                fzs.append(float(w_cur[2]))
                extra["depths"].append(max(0.0, env.SOCKET_TOP_Z - float(p_next[2])))
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


def train_act_policy_v4_8(
    demos: List[Dict[str, Any]],
    mode: str,
    encoder: Optional[nn.Module],
    device: torch.device,
    seed: int = 0,
    num_steps: int = 1000,
    batch_size: int = 128,
    lr: float = 8e-4,
) -> Tuple[ACTPolicy, Optional[CausalContactBackbone], np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if mode != "ACT_RGB_RawHistGRU":
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
        return policy, None, p_mean, p_std, c_mean, c_std

    # End-to-end trainable CausalContactBackbone(GRU) + ACTPolicy (RawHistGRU)
    chunk_len = 8
    vfeats_l, props_l, chunks_l = [], [], []
    hist_seqs = []
    for dm in demos:
        T = dm["contact_obs"].shape[0]
        w_scaled = (dm["contact_obs"] / CANON_WRENCH_SCALE[None, :]).astype(np.float32)
        p_raw = dm["proprio_state"]
        p_rel = p_raw.copy()
        p_rel[:, :3] = (p_raw[:, :3] - p_raw[0:1, :3]) / 0.12
        a_scaled = (dm["action_next"] / CANON_ACT_SCALE[None, :]).astype(np.float32)
        prev_a = np.concatenate([np.zeros((1, 7), dtype=np.float32), a_scaled[:-1]], axis=0)
        vis_xy_rel = np.repeat(((dm["coarse_vis_xy"] - p_raw[0, :2]) / 0.12)[None, :], T, axis=0).astype(np.float32)
        prop = np.concatenate([p_rel, prev_a, vis_xy_rel], axis=-1).astype(np.float32)

        wt = torch.from_numpy(w_scaled).unsqueeze(0)
        pt = torch.from_numpy(prev_a).unsqueeze(0)
        hist_seqs.append(build_causal_contact_input(wt, pt, use_act_exec=True, use_causal_diff=True, use_abs_wrench=True).squeeze(0))

        pad = np.repeat(a_scaled[-1:], chunk_len, axis=0)
        a_pad = np.concatenate([a_scaled, pad], axis=0)
        for t in range(T):
            vfeats_l.append(dm["vis_feat"][t])
            props_l.append(prop[t])
            chunks_l.append(a_pad[t : t + chunk_len])

    vfeats_np = np.stack(vfeats_l, axis=0).astype(np.float32)
    props_np = np.stack(props_l, axis=0).astype(np.float32)
    chunks_np = np.stack(chunks_l, axis=0).astype(np.float32)
    p_mean = props_np.mean(axis=0).astype(np.float32)
    p_std = np.maximum(props_np.std(axis=0), 1e-3).astype(np.float32)
    props_np = ((props_np - p_mean) / p_std).astype(np.float32)

    vis_t = torch.from_numpy(vfeats_np).to(device)
    prop_t = torch.from_numpy(props_np).to(device)
    chunk_t = torch.from_numpy(chunks_np).to(device)
    hist_all_t = torch.stack(hist_seqs, dim=0).to(device)  # (N_ep, T, 19)

    gru_enc = CausalContactBackbone(in_dim=19, d_z=32, hidden_dim=128, num_layers=2, backbone_type="gru").to(device)
    policy = ACTPolicy(
        prop_dim=props_np.shape[1],
        contact_dim=32,
        act_dim=7,
        chunk_len=8,
        d_model=96,
        nhead=4,
        num_enc_layers=1,
        num_dec_layers=2,
        d_style=16,
    ).to(device)

    all_params = list(policy.parameters()) + list(gru_enc.parameters())
    opt = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps, eta_min=lr * 0.1)

    N = len(vis_t)
    bs = min(batch_size, N)
    policy.train()
    gru_enc.train()
    for _ in range(num_steps):
        idx = torch.randint(0, N, (bs,), device=device)
        z_all = gru_enc(hist_all_t).reshape(N, 32)
        loss, _ = policy.compute_loss(vis_t[idx], prop_t[idx], chunk_t[idx], contact_feat=z_all[idx], kl_weight=0.2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 2.0)
        opt.step()
        sched.step()
    policy.eval()
    gru_enc.eval()
    return policy, gru_enc, p_mean, p_std, None, None


@torch.no_grad()
def evaluate_v4_8_episode_suite(
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
    n_eval: int = 24,
    eval_seed: int = 9200,
    drift_sigma: float = 0.0,
) -> Dict[str, Any]:
    env = AutoConnectorChamferEnv(rgb_w=64, rgb_h=64) if task_name == "auto_connector_chamfer" else make_env(task_name, rgb_w=64, rgb_h=64)
    rng = np.random.default_rng(eval_seed)
    rng_vis = np.random.default_rng(eval_seed + 404)
    drift_rng = np.random.default_rng(eval_seed + 777)

    succs, viols, peaks = [], [], []
    norm_succs, jam_succs = [], []
    contact_est, sock_entry, max_depths_all, max_depths_fail, prem_yield_fail = [], [], [], [], []
    ep_labels, ep_types = [], []
    ep_rel_surp, ep_stiff_std, ep_raw_norm_max, ep_raw_rate_max = [], [], [], []

    if policy is not None:
        assert vis_extractor is not None and p_mean is not None and p_std is not None
        pm_t = torch.from_numpy(p_mean).to(device)
        ps_t = torch.from_numpy(p_std).to(device)
        cm_t = torch.from_numpy(c_mean).to(device) if c_mean is not None else None
        cs_t = torch.from_numpy(c_std).to(device) if c_std is not None else None
        ensembler = TemporalEnsembler(chunk_len=8, act_dim=7, k_exp=0.25)

    is_conn = task_name in ("auto_connector", "auto_connector_chamfer")
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

                if mode == "Classical_Guided":
                    act_canon = env.classical_guided_policy(w_obs, t, coarse_vis_xy)
                    z_pred_dw = None
                elif mode == "Classical_Baseline":
                    act_canon = env.classical_policy(w_obs, t)
                    z_pred_dw = None
                else:
                    assert policy is not None and vis_extractor is not None
                    if t % 2 == 0 or v_t is None:
                        rgb_u8 = env.render_rgb(64, 64)
                        img_t = torch.from_numpy(rgb_u8).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
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
                    elif mode == "ACT_RGB_RawHistGRU":
                        assert encoder is not None
                        w_cur_t = torch.from_numpy(w_scaled).unsqueeze(0).to(device)
                        pa_cur_t = torch.from_numpy(prev_a_scaled).to(device)
                        z_cur_t, h_gru = step_encode_recurrent(encoder, w_cur_t, w_prev_t, pa_cur_t, h_gru)
                        w_prev_t = w_cur_t
                        cf_t = z_cur_t
                    else:
                        assert encoder is not None and cm_t is not None and cs_t is not None
                        w_cur_t = torch.from_numpy(w_scaled).unsqueeze(0).to(device)
                        pa_cur_t = torch.from_numpy(prev_a_scaled).to(device)
                        z_cur_t, h_gru = step_encode_recurrent(encoder, w_cur_t, w_prev_t, pa_cur_t, h_gru)
                        w_prev_t = w_cur_t
                        cf_t = ((z_cur_t.squeeze(0) - cm_t) / cs_t).unsqueeze(0)
                        if isinstance(encoder, PAECModel):
                            p18 = encoder.compute_virtual_probe(z_cur_t, step_scale=1.0).squeeze(0).cpu().numpy()
                            kz_steps.append(float(0.5 * (p18[14] - p18[17])))

                    chunk_t, _, _ = policy(v_t, prop_t, contact_feat=cf_t, act_chunk=None)
                    a_exec_scaled = ensembler.step(chunk_t.squeeze(0).cpu().numpy())
                    act_canon = (a_exec_scaled * CANON_ACT_SCALE).astype(np.float32)

                    if isinstance(encoder, PAECModel) and z_cur_t is not None:
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
                if is_conn:
                    extra["depths"].append(max(0.0, env.SOCKET_TOP_Z - float(p_next[2])))
                elif task_name == "semi_grasp":
                    extra["finger_forces"].append(float(sig))

            ev = env.evaluate_episode(fz_hist, extra)
            succs.append(float(ev["success"]))
            viols.append(float(ev["safety_viol"]))
            peaks.append(float(ev["peak_fn"]))

            if is_conn:
                fz_true = env.TOOL_TARE_Z - np.asarray(fz_hist, dtype=np.float64)
                max_fn_true = float(fz_true.max())
                max_d_mm = float(ev["max_depth_mm"])
                contact_est.append(float(max_fn_true >= 1.5))
                sock_entry.append(float(max_d_mm >= 7.0))
                max_depths_all.append(max_d_mm)
                if cfg["is_defect"] == 0:
                    norm_succs.append(float(ev["success"]))
                else:
                    jam_succs.append(float(ev["success"]))
                if not ev["success"]:
                    max_depths_fail.append(max_d_mm)
                    prem_yield_fail.append(float((max_fn_true < 6.0) and (max_d_mm < 12.0)))

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
    if is_conn:
        out["normal_succ_rate"] = round(100.0 * float(np.mean(norm_succs)), 2) if norm_succs else None
        out["jam_succ_rate"] = round(100.0 * float(np.mean(jam_succs)), 2) if jam_succs else None
        out["contact_established_rate"] = round(100.0 * float(np.mean(contact_est)), 2)
        out["socket_entry_rate"] = round(100.0 * float(np.mean(sock_entry)), 2)
        out["max_depth_mm_all"] = round(float(np.mean(max_depths_all)), 2)
        out["max_depth_mm_failed"] = round(float(np.mean(max_depths_fail)), 2) if max_depths_fail else None
        out["premature_yield_rate_of_failed"] = round(100.0 * float(np.mean(prem_yield_fail)), 2) if prem_yield_fail else None

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


# ======================================================================
# Part 4: Task 3 Hierarchical FSM + Contact Primitive (No Oracle Leakage)
# ======================================================================
class ContactPrimitiveHead(nn.Module):
    """Lightweight 2-layer MLP predicting chuck touchdown logit & slip-grip delta."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_semigrasp_primitive(
    demos: List[Dict[str, Any]],
    mode: str,
    encoder: Optional[nn.Module],
    device: torch.device,
    seed: int = 0,
    num_steps: int = 800,
) -> Tuple[ContactPrimitiveHead, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    feats_l, touch_l, slip_l = [], [], []
    CHUCK_X = 0.085
    GRIPPER_TARE_Z = 3.924

    for dm in demos:
        T = dm["contact_obs"].shape[0]
        w_scaled = (dm["contact_obs"] / CANON_WRENCH_SCALE[None, :]).astype(np.float32)
        p_raw = dm["proprio_state"]
        p_rel = p_raw.copy()
        p_rel[:, :3] = (p_raw[:, :3] - p_raw[0:1, :3]) / 0.12
        a_scaled = (dm["action_next"] / CANON_ACT_SCALE[None, :]).astype(np.float32)
        prev_a = np.concatenate([np.zeros((1, 7), dtype=np.float32), a_scaled[:-1]], axis=0)

        if mode == "FSM_RawFT_Primitive":
            c_seq = w_scaled
        else:
            assert encoder is not None
            encoder.eval()
            with torch.no_grad():
                wt = torch.from_numpy(w_scaled).unsqueeze(0).to(device)
                pt = torch.from_numpy(prev_a).unsqueeze(0).to(device)
                c_seq = encoder.encode(wt, pt).squeeze(0).cpu().numpy().astype(np.float32)

        for t in range(T):
            fz_rel_true = float(GRIPPER_TARE_Z - dm["contact_obs"][t, 2])
            tcp_x = float(p_raw[t, 0])
            g_cur = float(p_raw[t, 9])
            is_touch = 1.0 if (tcp_x >= CHUCK_X - 0.015 and fz_rel_true < -1.2) else 0.0
            is_slip_tighten = 1.0 if (tcp_x < 0.060 and float(p_raw[t, 2]) > 0.05 and dm["action_next"][t, 6] < -0.0001 and g_cur < 0.016) else 0.0
            feat = np.concatenate([p_rel[t], prev_a[t], c_seq[t]], axis=-1).astype(np.float32)
            feats_l.append(feat)
            touch_l.append(is_touch)
            slip_l.append(is_slip_tighten)

    X = np.stack(feats_l, axis=0).astype(np.float32)
    y_touch = np.asarray(touch_l, dtype=np.float32)
    y_slip = np.asarray(slip_l, dtype=np.float32)

    xm = X.mean(axis=0).astype(np.float32)
    xs = np.maximum(X.std(axis=0), 1e-3).astype(np.float32)
    Xn = (X - xm) / xs

    X_t = torch.from_numpy(Xn).to(device)
    yt_t = torch.from_numpy(y_touch).to(device)
    ys_t = torch.from_numpy(y_slip).to(device)

    pos_w = torch.tensor([(len(y_touch) - y_touch.sum()) / max(y_touch.sum(), 1.0)], device=device).clamp(1.0, 20.0)
    head = ContactPrimitiveHead(X.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    N = len(X_t)
    bs = min(128, N)
    head.train()
    for _ in range(num_steps):
        idx = torch.randint(0, N, (bs,), device=device)
        out = head(X_t[idx])
        loss_t = F.binary_cross_entropy_with_logits(out[:, 0], yt_t[idx], pos_weight=pos_w)
        loss_s = F.binary_cross_entropy_with_logits(out[:, 1], ys_t[idx])
        loss = loss_t + 0.5 * loss_s
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    return head, xm, xs


@torch.no_grad()
def evaluate_semigrasp_fsm(
    mode: str,
    head: Optional[ContactPrimitiveHead],
    xm: Optional[np.ndarray],
    xs: Optional[np.ndarray],
    encoder: Optional[nn.Module],
    device: torch.device,
    n_eval: int = 24,
    eval_seed: int = 9200,
    drift_sigma: float = 0.0,
) -> Dict[str, Any]:
    env = make_env("semi_grasp", rgb_w=64, rgb_h=64)
    rng = np.random.default_rng(eval_seed)
    rng_vis = np.random.default_rng(eval_seed + 404)
    drift_rng = np.random.default_rng(eval_seed + 777)

    succs, viols, peaks = [], [], []
    if head is not None:
        assert xm is not None and xs is not None
        xm_t = torch.from_numpy(xm).to(device)
        xs_t = torch.from_numpy(xs).to(device)

    try:
        for _ in range(n_eval):
            env.reset(rng)
            coarse_vis_xy = env.get_coarse_vis_xy(rng_vis)
            drift_vec = (
                drift_sigma * drift_rng.choice([-1.0, 1.0], size=6).astype(np.float32) * CANON_WRENCH_SCALE
            ) if drift_sigma > 0.0 else np.zeros(6, dtype=np.float32)

            p0 = env.get_proprio10d()[:3].copy()
            prev_a_scaled = np.zeros((1, 7), dtype=np.float32)
            fz_hist = []
            extra: Dict[str, Any] = {"finger_forces": []}
            h_gru: Optional[torch.Tensor] = None
            w_prev_t: Optional[torch.Tensor] = None
            placed = False

            for t in range(env.MAX_STEPS):
                w_obs = env.last_w_avg + drift_vec
                w_scaled = (w_obs / CANON_WRENCH_SCALE).astype(np.float32)
                p_cur = env.get_proprio10d()
                tcp_x, tcp_y = float(p_cur[0]), float(p_cur[1])
                jz = float(env.data.joint("jz").qpos[0])
                g_cur = float(env.ctrl[3])

                touch_detected = False
                slip_detected = False
                if mode == "FSM_Classical_ForceThresh":
                    fz_rel_obs = float(env.GRIPPER_TARE_Z - w_obs[2])
                    touch_detected = (tcp_x >= env.CHUCK_X - 0.015) and (fz_rel_obs < -1.2)
                    slip_detected = abs(float(w_obs[4])) > 0.028
                else:
                    assert head is not None
                    p_rel = p_cur.copy()
                    p_rel[:3] = (p_cur[:3] - p0) / 0.12
                    if mode == "FSM_RawFT_Primitive":
                        cf_np = w_scaled
                    else:
                        assert encoder is not None
                        w_cur_t = torch.from_numpy(w_scaled).unsqueeze(0).to(device)
                        pa_cur_t = torch.from_numpy(prev_a_scaled).to(device)
                        z_cur_t, h_gru = step_encode_recurrent(encoder, w_cur_t, w_prev_t, pa_cur_t, h_gru)
                        w_prev_t = w_cur_t
                        cf_np = z_cur_t.squeeze(0).cpu().numpy()
                    feat_np = np.concatenate([p_rel, prev_a_scaled[0], cf_np], axis=-1).astype(np.float32)
                    feat_t = ((torch.from_numpy(feat_np).to(device) - xm_t) / xs_t).unsqueeze(0)
                    logits = head(feat_t).squeeze(0)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    touch_detected = (tcp_x >= env.CHUCK_X - 0.015) and (probs[0] > 0.5)
                    slip_detected = probs[1] > 0.5

                if touch_detected:
                    placed = True

                # 4-stage industrial FSM using ONLY coarse_vis_xy (no oracle x0 or z_chuck)
                if placed or (tcp_x >= env.CHUCK_X - 0.015 and g_cur > 0.024):
                    dx, dy = 0.0, 0.0
                    dg = 0.0032 if g_cur < 0.035 else 0.0
                    dz = 0.0025 if g_cur >= 0.028 else 0.0
                elif g_cur > 0.0158 and tcp_x < 0.045:
                    ex_vis = float(coarse_vis_xy[0] - tcp_x)
                    ey_vis = float(coarse_vis_xy[1] - tcp_y)
                    if jz <= -0.068 and abs(ex_vis) < 0.006:
                        dx, dy, dz = 0.0, 0.0, 0.0
                        dg = float(np.clip(0.0154 - g_cur, -0.0024, 0.0))
                    else:
                        dx = float(np.clip(0.55 * ex_vis, -0.0045, 0.0045))
                        dy = float(np.clip(0.50 * ey_vis, -0.0030, 0.0030))
                        dz = float(np.clip(0.45 * (-0.072 - jz), -0.0040, 0.0040)) if abs(ex_vis) < 0.008 else 0.0
                        dg = 0.0
                elif tcp_x < 0.055 and jz < -0.006:
                    dx, dy = 0.0, 0.0
                    dz = float(np.clip(0.45 * (0.004 - jz), -0.0040, 0.0040))
                    dg = -0.0004 if (slip_detected and g_cur > 0.0148) else 0.0
                elif tcp_x < env.CHUCK_X - 0.004:
                    dx = float(np.clip(0.45 * (env.CHUCK_X - tcp_x), -0.0045, 0.0045))
                    dy = float(np.clip(-0.40 * tcp_y, -0.0020, 0.0020))
                    dz = float(np.clip(0.35 * (0.004 - jz), -0.0020, 0.0020))
                    dg = 0.0
                else:
                    dx = float(np.clip(0.40 * (env.CHUCK_X - tcp_x), -0.0020, 0.0020))
                    dy = 0.0
                    dz = -0.0030
                    dg = 0.0

                act_canon = np.array([dx, dy, dz, 0.0, 0.0, 0.0, dg], dtype=np.float32)
                prev_a_scaled = (act_canon / CANON_ACT_SCALE).astype(np.float32)[None, :]
                w_next_raw, _, sig = env.step_action(act_canon)
                fz_hist.append(float(w_next_raw[2]))
                extra["finger_forces"].append(float(sig))

            ev = env.evaluate_episode(fz_hist, extra)
            succs.append(float(ev["success"]))
            viols.append(float(ev["safety_viol"]))
            peaks.append(float(ev["peak_fn"]))
    finally:
        env._close_renderer()

    return {
        "success_rate": round(100.0 * float(np.mean(succs)), 2),
        "safety_viol_rate": round(100.0 * float(np.mean(viols)), 2),
        "peak_fn_n": round(float(np.mean(peaks)), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--baselines-ckpt", type=str, required=True)
    parser.add_argument("--b4-ckpt", type=str, required=True)
    parser.add_argument("--purediff-ckpt", type=str, required=True)
    parser.add_argument("--widedrift-ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--n-demos", type=int, default=15)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=24)
    parser.add_argument("--parts", type=str, nargs="+", default=["part1", "part2", "part3", "part4"])
    parser.add_argument("--out-json", type=str, default="v4_8_closedloop_results.json")
    args = parser.parse_args()

    t_all = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = out_dir / args.out_json

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = [0] if args.quick_test else [0, 1, 2]
    n_eval = 4 if args.quick_test else args.n_eval
    num_steps = 100 if args.quick_test else args.num_steps
    n_demos = 10 if args.quick_test else args.n_demos
    regimes = [("clean", 0.0), ("thermal_drift_0p5", 0.5), ("extreme_drift_1p5", 1.5)]

    print(
        f"[V4.8 ClosedLoop] Device={device} | quick={args.quick_test} | n_demos={n_demos} | "
        f"seeds={seeds} | n_eval={n_eval} | num_steps={num_steps}",
        flush=True,
    )

    vis_extractor = FrozenResNet18Extractor(device)
    pre_hashes: Dict[str, str] = {"FrozenResNet18": sha256_model(vis_extractor)}

    encoders: Dict[str, Optional[nn.Module]] = {
        "ACT_RGB_Proprio": None,
        "ACT_RGB_RawFT": None,
        "ACT_RGB_RawHistGRU": None,
    }

    # Load frozen encoders
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m_paec = build_policy_model_for_key("PAEC_Small_d32", device)
    m_paec.load_state_dict(ckpt["state_dicts"]["PAEC_Small_d32"])

    bckpt = torch.load(args.baselines_ckpt, map_location="cpu", weights_only=False)
    m_mae = build_policy_model_for_key("B2b_MAE_Small_d16", device)
    m_mae.load_state_dict(bckpt["B2b_MAE_Small_d16"])
    m_jepa = build_policy_model_for_key("B3b_JEPA_Small_d16", device)
    m_jepa.load_state_dict(bckpt["B3b_JEPA_Small_d16"])

    b4_ck = torch.load(args.b4_ckpt, map_location="cpu", weights_only=False)
    m_nobg = PAECModel(6, 6, 7, d_z=32, hidden_dim=256, num_layers=2, use_bg_split=False, drift_aug_std=0.35).to(device)
    m_nobg.load_state_dict(b4_ck["state_dict"])

    pd_ck = torch.load(args.purediff_ckpt, map_location="cpu", weights_only=False)
    m_purediff = PAECModel(6, 6, 7, d_z=32, hidden_dim=256, num_layers=2, use_bg_split=True, use_abs_wrench=False, drift_aug_std=0.0).to(device)
    m_purediff.load_state_dict(pd_ck["state_dict"])

    wd_ck = torch.load(args.widedrift_ckpt, map_location="cpu", weights_only=False)
    m_widedrift = PAECModel(6, 6, 7, d_z=32, hidden_dim=256, num_layers=2, use_bg_split=True, use_abs_wrench=True, drift_aug_std=1.5, step_drift_prob=0.35).to(device)
    m_widedrift.load_state_dict(wd_ck["state_dict"])

    frozen_map = {
        "ACT_RGB_MAE": m_mae,
        "ACT_RGB_JEPA": m_jepa,
        "ACT_RGB_NoBG": m_nobg,
        "ACT_RGB_PAEC": m_paec,
        "ACT_RGB_PAEC_PureDiff": m_purediff,
        "ACT_RGB_PAEC_WideDrift": m_widedrift,
    }
    for k, m in frozen_map.items():
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        encoders[k] = m
        pre_hashes[k] = sha256_model(m)
        verify_recurrent_equivalence(m, k, device)

    # Also verify RawHistGRU O(1) recurrent equivalence
    _test_gru = CausalContactBackbone(in_dim=19, d_z=32, hidden_dim=128, num_layers=2, backbone_type="gru").to(device)
    verify_recurrent_equivalence(_test_gru, "ACT_RGB_RawHistGRU", device)
    del _test_gru

    results: Dict[str, Any] = {
        "protocol": "v4_8_diagnostic_ablation_closedloop",
        "strict_fp32": True,
        "strictly_frozen_pass": True,
        "quick_test": bool(args.quick_test),
        "n_demos": n_demos,
        "seeds": seeds,
        "n_eval": n_eval,
        "pre_hashes": pre_hashes,
        "parts": {},
    }

    def run_act_suite_for_task(task_name: str, demos: List[Dict[str, Any]], methods: List[str], include_classical_guided: bool = False) -> Dict[str, Any]:
        exp_succ = 100.0 * float(np.mean([d["eval"]["success"] for d in demos]))
        exp_viol = 100.0 * float(np.mean([d["eval"]["safety_viol"] for d in demos]))
        print(f"\n==================== {task_name} (N_demos={len(demos)}, Expert Succ={exp_succ:.1f}%) ====================", flush=True)
        t_out: Dict[str, Any] = {
            "expert_demos": {"n": len(demos), "success_rate": round(exp_succ, 2), "safety_viol_rate": round(exp_viol, 2)},
            "act_methods": {},
        }
        if include_classical_guided:
            t_out["classical_guided"] = {}
            for rname, d_sig in regimes:
                runs = [
                    evaluate_v4_8_episode_suite(
                        task_name, None, None, None, None, None, None, "Classical_Guided",
                        None, device, n_eval=n_eval, eval_seed=es, drift_sigma=d_sig,
                    )
                    for es in (9200, 9201, 9202)
                ]
                t_out["classical_guided"][rname] = _agg_runs(runs)
            cg_c = t_out["classical_guided"]["clean"]
            cg_05 = t_out["classical_guided"]["thermal_drift_0p5"]
            cg_15 = t_out["classical_guided"]["extreme_drift_1p5"]
            print(
                f"  [Classical_Guided       ] Clean={cg_c['success_rate_mean']:5.1f}% (Viol={cg_c['safety_viol_rate_mean']:4.1f}%) | "
                f"0.5σ={cg_05['success_rate_mean']:5.1f}% | 1.5σ={cg_15['success_rate_mean']:5.1f}%",
                flush=True,
            )

        for mode in methods:
            enc = encoders[mode]
            reg_runs: Dict[str, List[Dict[str, Any]]] = {r[0]: [] for r in regimes}
            for sd in seeds:
                pol, gru_e, pm, ps, cm, cs = train_act_policy_v4_8(
                    demos, mode, enc, device, seed=sd, num_steps=num_steps, batch_size=128, lr=8e-4
                )
                use_enc = gru_e if mode == "ACT_RGB_RawHistGRU" else enc
                if mode == "ACT_RGB_Proprio":
                    ev_c = evaluate_v4_8_episode_suite(
                        task_name, pol, vis_extractor, pm, ps, cm, cs, mode, use_enc,
                        device, n_eval=n_eval, eval_seed=9200, drift_sigma=0.0,
                    )
                    for rname, _ in regimes:
                        reg_runs[rname].append(copy.deepcopy(ev_c))
                else:
                    for rname, d_sig in regimes:
                        reg_runs[rname].append(
                            evaluate_v4_8_episode_suite(
                                task_name, pol, vis_extractor, pm, ps, cm, cs, mode, use_enc,
                                device, n_eval=n_eval, eval_seed=9200, drift_sigma=d_sig,
                            )
                        )
            t_out["act_methods"][mode] = {rname: _agg_runs(reg_runs[rname]) for rname, _ in regimes}
            c_m = t_out["act_methods"][mode]["clean"]
            d05 = t_out["act_methods"][mode]["thermal_drift_0p5"]
            d15 = t_out["act_methods"][mode]["extreme_drift_1p5"]
            extra_s = ""
            if "premature_yield_rate_of_failed_mean" in d05:
                extra_s = f" | 0.5σ_PremYieldFail={d05.get('premature_yield_rate_of_failed_mean', 0.0):4.1f}% Depth={d05['max_depth_mm_all_mean']:4.1f}mm"
            print(
                f"  [{mode:23s}] Clean={c_m['success_rate_mean']:5.1f}% (V={c_m['safety_viol_rate_mean']:4.1f}%) | "
                f"0.5σ={d05['success_rate_mean']:5.1f}% (V={d05['safety_viol_rate_mean']:4.1f}%) | "
                f"1.5σ={d15['success_rate_mean']:5.1f}% (V={d15['safety_viol_rate_mean']:4.1f}%){extra_s}",
                flush=True,
            )
        return t_out

    # Part 1: Original auto_connector failure anatomy + complete 9-method fair comparison
    if "part1" in args.parts:
        demos_ac = collect_task_demos("auto_connector", n_demos, vis_extractor, device, seed=2027)
        results["parts"]["part1_auto_connector_anatomy"] = run_act_suite_for_task(
            "auto_connector",
            demos_ac,
            [
                "ACT_RGB_Proprio",
                "ACT_RGB_RawFT",
                "ACT_RGB_RawHistGRU",
                "ACT_RGB_MAE",
                "ACT_RGB_JEPA",
                "ACT_RGB_NoBG",
                "ACT_RGB_PAEC",
                "ACT_RGB_PAEC_PureDiff",
                "ACT_RGB_PAEC_WideDrift",
            ],
        )
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Part 2: Original battery_scan complete fair comparison
    if "part2" in args.parts:
        demos_bs = collect_task_demos("battery_scan", n_demos, vis_extractor, device, seed=2027)
        results["parts"]["part2_battery_scan_fair"] = run_act_suite_for_task(
            "battery_scan",
            demos_bs,
            [
                "ACT_RGB_Proprio",
                "ACT_RGB_RawFT",
                "ACT_RGB_RawHistGRU",
                "ACT_RGB_MAE",
                "ACT_RGB_JEPA",
                "ACT_RGB_NoBG",
                "ACT_RGB_PAEC",
                "ACT_RGB_PAEC_PureDiff",
                "ACT_RGB_PAEC_WideDrift",
            ],
        )
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Part 3: Chamfered Task 2 (auto_connector_chamfer, no-oracle recovery demos)
    if "part3" in args.parts:
        demos_acc = collect_chamfer_demos(n_demos, vis_extractor, device, seed=2027)
        results["parts"]["part3_auto_connector_chamfer"] = run_act_suite_for_task(
            "auto_connector_chamfer",
            demos_acc,
            [
                "ACT_RGB_Proprio",
                "ACT_RGB_RawFT",
                "ACT_RGB_RawHistGRU",
                "ACT_RGB_NoBG",
                "ACT_RGB_PAEC",
                "ACT_RGB_PAEC_PureDiff",
                "ACT_RGB_PAEC_WideDrift",
            ],
            include_classical_guided=True,
        )
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Part 4: Task 3 Hierarchical FSM + Contact Primitive (semi_grasp)
    if "part4" in args.parts:
        print(f"\n==================== part4_semi_grasp_fsm (N_demos={n_demos}) ====================", flush=True)
        demos_sg = collect_task_demos("semi_grasp", n_demos, vis_extractor, device, seed=2027)
        sg_res: Dict[str, Any] = {
            "expert_demos": {
                "n": len(demos_sg),
                "success_rate": round(100.0 * float(np.mean([d["eval"]["success"] for d in demos_sg])), 2),
            },
            "v4_7_pure_e2e_act_paec_ref": {"clean_succ": 13.89, "thermal_0p5_succ": 5.56, "extreme_1p5_succ": 0.0},
            "fsm_methods": {},
        }
        fsm_specs: List[Tuple[str, Optional[nn.Module]]] = [
            ("FSM_Classical_ForceThresh", None),
            ("FSM_RawFT_Primitive", None),
            ("FSM_NoBG_Primitive", m_nobg),
            ("FSM_PAEC_Primitive", m_paec),
            ("FSM_PAEC_PureDiff_Primitive", m_purediff),
            ("FSM_PAEC_WideDrift_Primitive", m_widedrift),
        ]
        for fmode, fenc in fsm_specs:
            reg_runs = {r[0]: [] for r in regimes}
            if fmode == "FSM_Classical_ForceThresh":
                for rname, d_sig in regimes:
                    for es in (9200, 9201, 9202):
                        reg_runs[rname].append(
                            evaluate_semigrasp_fsm(
                                fmode, None, None, None, None, device, n_eval=n_eval, eval_seed=es, drift_sigma=d_sig
                            )
                        )
            else:
                for sd in seeds:
                    head, xm, xs = train_semigrasp_primitive(demos_sg, fmode, fenc, device, seed=sd, num_steps=800)
                    for rname, d_sig in regimes:
                        reg_runs[rname].append(
                            evaluate_semigrasp_fsm(
                                fmode, head, xm, xs, fenc, device, n_eval=n_eval, eval_seed=9200, drift_sigma=d_sig
                            )
                        )
            sg_res["fsm_methods"][fmode] = {rname: _agg_runs(reg_runs[rname]) for rname, _ in regimes}
            c_m = sg_res["fsm_methods"][fmode]["clean"]
            d05 = sg_res["fsm_methods"][fmode]["thermal_drift_0p5"]
            d15 = sg_res["fsm_methods"][fmode]["extreme_drift_1p5"]
            print(
                f"  [{fmode:28s}] Clean={c_m['success_rate_mean']:5.1f}% (V={c_m['safety_viol_rate_mean']:4.1f}%) | "
                f"0.5σ={d05['success_rate_mean']:5.1f}% (V={d05['safety_viol_rate_mean']:4.1f}%) | "
                f"1.5σ={d15['success_rate_mean']:5.1f}% (V={d15['safety_viol_rate_mean']:4.1f}%)",
                flush=True,
            )
        results["parts"]["part4_semi_grasp_fsm"] = sg_res

    # Final SHA-256 frozen audit
    assert sha256_model(vis_extractor) == pre_hashes["FrozenResNet18"], "FrozenResNet18 hash mismatch!"
    for k, m in frozen_map.items():
        if sha256_model(m) != pre_hashes[k]:
            raise RuntimeError(f"Frozen weight mutation detected for {k}!")

    results["total_wall_sec"] = round(time.time() - t_all, 1)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[V4.8 ClosedLoop] COMPLETE in {results['total_wall_sec']}s -> {out_json_path}", flush=True)


if __name__ == "__main__":
    main()
