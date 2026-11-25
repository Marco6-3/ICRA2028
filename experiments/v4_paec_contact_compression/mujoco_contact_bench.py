#!/usr/bin/env python3
"""
V4.5 Closed-Loop Contact-Rich MuJoCo Benchmark (Robotics-Plugin Standards).

Tests whether a frozen Physics-Anchored Action-Effect Contact Compression (PAEC)
encoder pretrained ONLY on heterogeneous real-robot RH20T 6D F/T trajectories can
zero-shot transfer to a live closed-loop MuJoCo compliant wiping & snag-recovery
task with few-shot (5/10/20 demos) downstream BC policies.

Strictly follows MuJoCo 3/4.x robotics-plugin best practices:
  - MjSpec procedural modeling, SI units, mjINT_IMPLICITFAST, mjCONE_ELLIPTIC
  - Declarative mjSENS_FORCE + mjSENS_TORQUE wrist sensors (fixed sensordata layout)
  - Named access (data.body/geom/joint/sensor/actuator) + .copy() history logging
  - Independent physical hazard ground-truth labels (snag patch / lateral impulse)
  - Zero-shot SHA-256 frozen weight audit + Clean & +1.5sigma Drift OOD evaluation
  - Offscreen mujoco.Renderer rollout inspection strips
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from models_paec import (
    AbstractLatentJEPAVICReg,
    MAEContactRecon,
    PAECModel,
)


def _rot6d_from_rotmat(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1).astype(np.float32)
from run_v4_scaling_suite import (
    CANON_ACT_SCALE,
    CANON_WRENCH_SCALE,
    load_rh20t_shards,
    make_zero_leakage_splits,
    pretrain_on_episodes,
)


class CompliantWipingEnv:
    """
    Physically calibrated MuJoCo 4-DoF compliant wiping & snag-adaptation environment.

    Hidden per-episode physical variations (invisible to proprioception):
      - platen_z0: surface elevation offset in [-0.018, +0.018] m
      - platen_k:  surface spring stiffness in [450, 2800] N/m
      - platen_tilt: surface pitch tilt in [-0.07, +0.07] rad
      - snag_x: high-friction rubber strip center in [-0.07, +0.07] m (mu in [1.45, 2.25])
      - impulse_step / impulse_fy: unseen lateral/eccentric disturbance pulse
    """

    DT_SIM = 0.002
    SUBSTEPS = 25  # 20 Hz control loop (50 ms/step)
    MAX_STEPS = 105
    X_START = -0.15
    X_TARGET = 0.14
    TARGET_FN = 5.5
    TOOL_TARE_Z = 2.73  # Gravity tare below wrist_ft_site (0.28 kg tool mass)

    def __init__(self, render_w: int = 480, render_h: int = 360):
        self.render_w = render_w
        self.render_h = render_h
        self._build_spec(
            platen_z0=0.0,
            platen_k=1400.0,
            platen_tilt=0.0,
            snag_x=0.0,
            snag_mu=1.8,
            base_mu=0.45,
        )

    def _build_spec(
        self,
        platen_z0: float,
        platen_k: float,
        platen_tilt: float,
        snag_x: float,
        snag_mu: float,
        base_mu: float,
    ):
        spec = mujoco.MjSpec()
        spec.option.timestep = self.DT_SIM
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10.0

        spec.default.geom.solref = [0.025, 1.5]
        spec.default.joint.armature = 0.01
        spec.default.joint.damping = [0.35, 0.35, 0.35]

        # Studio lighting & overhead / oblique cameras for rollout visual inspection
        spec.worldbody.add_light(
            pos=[0.10, -0.35, 0.90],
            dir=[-0.10, 0.35, -1.0],
            diffuse=[0.95, 0.95, 0.95],
            ambient=[0.35, 0.35, 0.35],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
        spec.worldbody.add_camera(
            name="side_cam",
            pos=[0.0, -0.52, 0.14],
            xyaxes=[1, 0, 0, 0, 0.22, 0.975],
            fovy=45,
        )
        spec.worldbody.add_camera(
            name="oblique_cam",
            pos=[0.30, -0.38, 0.24],
            xyaxes=[0.78, 0.62, 0, -0.28, 0.36, 0.89],
            fovy=45,
        )

        # Base table (visual only)
        spec.worldbody.add_geom(
            name="table_visual",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.35, 0.22, 0.015],
            pos=[0, 0, -0.06],
            contype=0,
            conaffinity=0,
            mass=0.0,
            rgba=[0.18, 0.20, 0.24, 1.0],
        )

        # Compliant tilted work platen (spring-damped Z compliance)
        platen = spec.worldbody.add_body(
            name="platen",
            pos=[0.0, 0.0, platen_z0],
            euler=[0.0, platen_tilt, 0.0],
        )
        platen.add_joint(
            name="pz",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, 0, 1],
            stiffness=platen_k,
            damping=max(65.0, 0.08 * platen_k),
            springref=0.0,
            range=[-0.045, 0.045],
        )
        platen.add_geom(
            name="platen_base",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.26, 0.12, 0.015],
            pos=[0.0, 0.0, -0.015],
            mass=0.8,
            condim=4,
            friction=[base_mu, 0.015, 0.0005],
            rgba=[0.42, 0.48, 0.56, 1.0],
        )
        # High-friction rubber snag tape (~0.4 mm thin flush layer, pure Coulomb shear spike)
        platen.add_geom(
            name="snag_strip",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.025, 0.115, 0.0002],
            pos=[snag_x, 0.0, 0.0002],
            mass=0.05,
            condim=4,
            friction=[snag_mu, 0.03, 0.001],
            rgba=[0.85, 0.30, 0.18, 1.0],
        )

        # Articulated 4-DoF gantry + pitch compliance head (jx, jy, jz, jp)
        bx = spec.worldbody.add_body(name="arm_x", pos=[0.0, 0.0, 0.18], mass=0.5, inertia=[1e-3, 1e-3, 1e-3])
        bx.add_joint(name="jx", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[-0.28, 0.28], damping=3.0)
        by = bx.add_body(name="arm_y", pos=[0, 0, 0], mass=0.5, inertia=[1e-3, 1e-3, 1e-3])
        by.add_joint(name="jy", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], range=[-0.15, 0.15], damping=3.0)
        bz = by.add_body(name="arm_z", pos=[0, 0, 0], mass=0.5, inertia=[1e-3, 1e-3, 1e-3])
        bz.add_joint(name="jz", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.25, 0.10], damping=4.0)

        bp = bz.add_body(name="wrist_pitch", pos=[0, 0, 0])
        bp.add_joint(name="jp", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-0.55, 0.55], damping=0.6)
        bp.add_geom(
            name="wrist_mount",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.024, 0.018],
            mass=0.3,
            contype=0,
            conaffinity=0,
            rgba=[0.22, 0.25, 0.30, 1.0],
        )

        # Tool body with wrist 6D F/T sensor site and compliant wiper capsule
        tool = bp.add_body(name="tool", pos=[0, 0, -0.038])
        tool.add_site(name="wrist_ft_site", pos=[0, 0, 0.018], size=[0.01])
        tool.add_site(name="tcp_site", pos=[0, 0, -0.055], size=[0.008], rgba=[0.1, 0.9, 0.3, 0.8])
        tool.add_geom(
            name="wiper_stem",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.012, 0.022],
            pos=[0, 0, -0.015],
            mass=0.15,
            contype=0,
            conaffinity=0,
            rgba=[0.70, 0.72, 0.76, 1.0],
        )
        tool.add_geom(
            name="wiper_tip",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.015, 0.018],
            pos=[0, 0, -0.042],
            mass=0.15,
            condim=4,
            friction=[0.65, 0.02, 0.001],
            rgba=[0.08, 0.48, 0.82, 1.0],
        )

        # Position servos (explicit trntype=mjTRN_JOINT per spec_editing rules)
        for jn, kp, dr, rng, flim in [
            ("jx", 450.0, 1.0, [-0.28, 0.28], 40.0),
            ("jy", 450.0, 1.0, [-0.15, 0.15], 40.0),
            ("jz", 110.0, 1.3, [-0.25, 0.10], 40.0),
            ("jp", 28.0, 1.0, [-0.55, 0.55], 8.0),
        ]:
            act = spec.add_actuator(name=f"act_{jn}", target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_position(kp=kp, dampratio=dr)
            act.ctrllimited = True
            act.ctrlrange = rng
            act.forcelimited = True
            act.forcerange = [-flim, flim]

        spec.add_sensor(name="wrist_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="wrist_ft_site")
        spec.add_sensor(name="wrist_torque", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="wrist_ft_site")

        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

    def reset(self, rng: np.random.Generator) -> Dict[str, Any]:
        z0 = float(rng.uniform(-0.016, 0.016))
        kz = float(np.exp(rng.uniform(np.log(500.0), np.log(2600.0))))
        tilt = float(rng.uniform(-0.06, 0.06))
        snag_x = float(rng.uniform(-0.065, 0.065))
        snag_mu = float(rng.uniform(1.45, 2.25))
        base_mu = float(rng.uniform(0.35, 0.55))
        imp_step = int(rng.integers(45, 78))
        imp_fy = float(rng.choice([-1.0, 1.0]) * rng.uniform(3.2, 5.0))

        self.ep_cfg = {
            "platen_z0": z0,
            "platen_k": kz,
            "platen_tilt": tilt,
            "snag_x": snag_x,
            "snag_mu": snag_mu,
            "base_mu": base_mu,
            "imp_step": imp_step,
            "imp_fy": imp_fy,
        }
        self._build_spec(z0, kz, tilt, snag_x, snag_mu, base_mu)
        mujoco.mj_resetData(self.model, self.data)

        # Start gently hovering ~2 mm above the left edge of the platen
        init_x = self.X_START
        surface_left = z0 - init_x * float(np.sin(tilt))
        init_jz = float(np.clip(surface_left - 0.065, -0.20, 0.05))
        self.data.joint("jx").qpos[0] = init_x
        self.data.joint("jy").qpos[0] = 0.0
        self.data.joint("jz").qpos[0] = init_jz
        self.data.joint("jp").qpos[0] = 0.0
        self.ctrl = np.array([init_x, 0.0, init_jz, 0.0], dtype=np.float64)
        self.data.ctrl[:] = self.ctrl
        mujoco.mj_forward(self.model, self.data)
        self.step_idx = 0
        self.tool_body_id = self.model.body("tool").id
        self.last_w_avg = self.get_wrench_tcp()
        return self.ep_cfg

    def get_wrench_tcp(self) -> np.ndarray:
        """
        Return 6D wrench [Fx, Fy, Fz, Tx, Ty, Tz] in TCP local frame (N, N*m).
        Note: MuJoCo mjSENS_FORCE returns force in site frame; normal platen contact
        exerts upward reaction on tip (so Fz_raw < 0 under downward pressing).
        """
        f = self.data.sensor("wrist_force").data.copy()
        t = self.data.sensor("wrist_torque").data.copy()
        return np.concatenate([f, t]).astype(np.float32)

    def get_proprio10d(self) -> np.ndarray:
        pos = self.data.site("tcp_site").xpos.copy().astype(np.float32)
        xmat = self.data.site("tcp_site").xmat.copy().reshape(1, 3, 3)
        r6 = _rot6d_from_rotmat(xmat)[0]
        return np.concatenate([pos, r6, np.zeros(1, dtype=np.float32)]).astype(np.float32)

    def step_action(self, act7_canon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Execute a canonical 7D TCP delta action [dp(3 m), dr(3 rad), dg(1 m)] for 1 20Hz step."""
        dx = float(np.clip(act7_canon[0], -0.010, 0.010))
        dy = float(np.clip(act7_canon[1], -0.008, 0.008))
        dz = float(np.clip(act7_canon[2], -0.010, 0.010))
        dp = float(np.clip(act7_canon[4], -0.04, 0.04))  # pitch around local +Y

        self.ctrl[0] = float(np.clip(self.ctrl[0] + dx, -0.26, 0.26))
        self.ctrl[1] = float(np.clip(self.ctrl[1] + dy, -0.10, 0.10))
        self.ctrl[2] = float(np.clip(self.ctrl[2] + dz, -0.23, 0.06))
        self.ctrl[3] = float(np.clip(self.ctrl[3] + dp, -0.45, 0.45))

        imp_active = int(self.ep_cfg["imp_step"] <= self.step_idx < self.ep_cfg["imp_step"] + 4)
        self.data.xfrc_applied[:] = 0.0
        if imp_active:
            self.data.xfrc_applied[self.tool_body_id, 1] = self.ep_cfg["imp_fy"]
            self.data.xfrc_applied[self.tool_body_id, 3] = -0.12 * self.ep_cfg["imp_fy"]

        self.data.ctrl[:] = self.ctrl
        w_accum = np.zeros(6, dtype=np.float64)
        for _ in range(self.SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            w_accum += self.get_wrench_tcp()
        self.data.xfrc_applied[:] = 0.0
        self.last_w_avg = (w_accum / float(self.SUBSTEPS)).astype(np.float32)

        self.step_idx += 1
        tcp_x = float(self.data.site("tcp_site").xpos[0])
        in_snag = int(abs(tcp_x - self.ep_cfg["snag_x"]) <= 0.025)
        hazard_gt = int(in_snag or imp_active)
        return self.last_w_avg.copy(), self.get_proprio10d(), hazard_gt


def expert_compliant_policy(env: CompliantWipingEnv, w_tcp: np.ndarray, step_idx: int) -> np.ndarray:
    """
    Force-feedback expert controller for demonstration generation.
    Uses gravity-tared 6D wrist wrench (Fz normal regulation + Fx snag yield + Fy centering).
    """
    fz_norm = float(env.TOOL_TARE_Z - w_tcp[2])  # 0 N in air, positive under downward pressing
    fx_shear = float(w_tcp[0])
    fy_lat = float(w_tcp[1])
    y_pos = float(env.data.site("tcp_site").xpos[1])
    z_vel = float(env.data.joint("jz").qvel[0])

    # Advance along +x smoothly, yielding velocity slightly on high-friction snag
    vx = 0.0040 if abs(fx_shear) < 5.5 else 0.0028
    # Smoothly damped normal force admittance toward TARGET_FN (5.5 N) with shear relief
    target_fn = max(3.2, env.TARGET_FN - max(0.0, abs(fx_shear) - 4.5) * 0.25)
    if step_idx < 8 and fz_norm < 1.0:
        dz = -0.0010
    else:
        dz = 0.00018 * (fz_norm - target_fn) - 0.004 * z_vel
        dz = float(np.clip(dz, -0.0020, 0.0020))

    # Lateral centering & disturbance rejection from Fy + y_pos
    dy = float(np.clip(-0.22 * y_pos + 0.00035 * fy_lat, -0.005, 0.005))
    # Gentle pitch adaptation to platen tilt / shear torque
    ty = float(w_tcp[4])
    dp = float(np.clip(-0.003 * np.tanh(ty), -0.012, 0.012))
    return np.array([vx, dy, dz, 0.0, dp, 0.0, 0.0], dtype=np.float32)


def collect_mujoco_demos(n_demos: int, seed: int = 2026) -> List[Dict[str, Any]]:
    env = CompliantWipingEnv()
    rng = np.random.default_rng(seed)
    demos = []
    for eid in range(n_demos):
        cfg = env.reset(rng)
        w_list, p_list, a_list, haz_list = [], [], [], []
        w_cur = env.last_w_avg.copy()
        for t in range(env.MAX_STEPS):
            p_cur = env.get_proprio10d()
            act = expert_compliant_policy(env, w_cur, t)
            w_list.append(w_cur)
            p_list.append(p_cur)
            a_list.append(act)
            w_cur, _, haz = env.step_action(act)
            haz_list.append(haz)
        w_arr = np.stack(w_list, axis=0)
        fz = env.TOOL_TARE_Z - w_arr[:, 2]
        final_x = float(env.data.site("tcp_site").xpos[0])
        demos.append(
            {
                "ep_id": eid,
                "cfg": cfg,
                "contact_obs": w_arr.astype(np.float32),
                "proprio_state": np.stack(p_list, axis=0).astype(np.float32),
                "action_next": np.stack(a_list, axis=0).astype(np.float32),
                "hazard_gt": np.array(haz_list, dtype=np.int64),
                "final_x": final_x,
                "max_fn": float(fz.max()),
                "mean_fn": float(fz[8:].mean()),
                "c_frac": float((fz[10:] >= 1.8).mean()),
                "fz_head": fz[:10].round(2).tolist(),
                "fz_tail": fz[-6:].round(2).tolist(),
                "fn_rmse": float(np.sqrt(np.mean((fz[10:] - env.TARGET_FN) ** 2))),
                "success": bool(final_x >= env.X_TARGET and float(fz.max()) <= 16.0 and float((fz[10:] >= 1.8).mean()) >= 0.75),
            }
        )
    return demos


class ClosedLoopPolicyNet(nn.Module):
    def __init__(self, in_dim: int, act_dim: int = 7, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sha256_model(model: nn.Module) -> str:
    h = hashlib.sha256()
    for _, p in sorted(model.named_parameters()):
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def build_policy_model_for_key(key: str, device: torch.device) -> nn.Module:
    if key == "PAEC_Tiny_d16":
        return PAECModel(6, 7, d_z=16, hidden_dim=96, num_layers=2).to(device)
    if key == "PAEC_Small_d32":
        return PAECModel(6, 7, d_z=32, hidden_dim=256, num_layers=2).to(device)
    if key == "PAEC_Base_d32":
        return PAECModel(6, 7, d_z=32, hidden_dim=512, num_layers=4).to(device)
    if key == "B2b_MAE_Small_d16":
        return MAEContactRecon(6, 7, d_z=16, hidden_dim=256, num_layers=2).to(device)
    if key == "B3b_JEPA_Small_d16":
        return AbstractLatentJEPAVICReg(6, 7, d_z=16, hidden_dim=256, num_layers=2).to(device)
    raise ValueError(f"Unknown key: {key}")


def encode_demo_features(
    demo: Dict[str, Any],
    mode: str,
    encoder: Optional[nn.Module],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-step policy inputs (scaled in Canonical SI) and scaled targets."""
    w_scaled = demo["contact_obs"] / CANON_WRENCH_SCALE[None, :]
    p_raw = demo["proprio_state"]  # [pos(3), rot6d(6), grip(1)]
    # Relative position to episode start + orientation + prev action
    p_rel = p_raw.copy()
    p_rel[:, :3] = (p_raw[:, :3] - p_raw[0:1, :3]) / 0.15
    a_scaled = demo["action_next"] / CANON_ACT_SCALE[None, :]
    prev_a = np.concatenate([np.zeros((1, 7), dtype=np.float32), a_scaled[:-1]], axis=0)
    prop = np.concatenate([p_rel, prev_a], axis=-1).astype(np.float32)  # 17D

    if mode == "B0_ProprioOnly":
        feats = prop
    elif mode == "B1b_RawWrench6D_E2E":
        feats = np.concatenate([prop, w_scaled], axis=-1)
    else:
        assert encoder is not None
        encoder.eval()
        with torch.no_grad():
            wt = torch.from_numpy(w_scaled).unsqueeze(0).to(device)
            pt = torch.from_numpy(prev_a).unsqueeze(0).to(device)
            z = encoder.encode(wt, pt).squeeze(0).cpu().numpy()
        feats = np.concatenate([prop, z], axis=-1)
    return feats.astype(np.float32), a_scaled.astype(np.float32)


def train_closedloop_bc(
    demos: List[Dict[str, Any]],
    mode: str,
    encoder: Optional[nn.Module],
    device: torch.device,
    seed: int = 0,
    epochs: int = 80,
) -> Tuple[ClosedLoopPolicyNet, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    xs, ys = [], []
    for dm in demos:
        x_ep, y_ep = encode_demo_features(dm, mode, encoder, device)
        xs.append(x_ep)
        ys.append(y_ep)
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)

    x_mean = X.mean(axis=0).astype(np.float32)
    x_std = np.maximum(X.std(axis=0), 1e-3).astype(np.float32)
    Xn = torch.from_numpy((X - x_mean) / x_std).to(device)
    Yn = torch.from_numpy(Y).to(device)

    net = ClosedLoopPolicyNet(in_dim=X.shape[1], act_dim=7, hidden=128).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Xn), device=device)
        for i in range(0, len(Xn), 128):
            idx = perm[i : i + 128]
            loss = F.mse_loss(net(Xn[idx]), Yn[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net, x_mean, x_std


def evaluate_closed_loop(
    net: ClosedLoopPolicyNet,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    mode: str,
    encoder: Optional[nn.Module],
    device: torch.device,
    n_eval: int = 24,
    eval_seed: int = 9001,
    drift_sigma: float = 0.0,
    capture_frames_ep: Optional[int] = None,
) -> Dict[str, Any]:
    env = CompliantWipingEnv()
    rng = np.random.default_rng(eval_seed)
    drift_rng = np.random.default_rng(eval_seed + 555)
    succs, rmses, max_fns, viols, prog_xs = [], [], [], [], []
    surprise_scores, raw_rate_scores, raw_norm_scores, gt_labels = [], [], [], []
    captured_imgs: List[np.ndarray] = []

    xm_t = torch.from_numpy(x_mean).to(device)
    xs_t = torch.from_numpy(x_std).to(device)

    for eid in range(n_eval):
        env.reset(rng)
        drift_vec = (drift_sigma * drift_rng.choice([-1.0, 1.0], size=6).astype(np.float32) * CANON_WRENCH_SCALE) if drift_sigma > 0 else np.zeros(6, dtype=np.float32)
        p0 = env.get_proprio10d()[:3].copy()
        prev_a_scaled = np.zeros((1, 7), dtype=np.float32)

        # Maintain strict causal streaming sequence for frozen temporal encoder
        w_hist, pa_hist = [], []
        fz_hist = []
        w_scaled_prev = None
        z_prev_t = None

        renderer_ctx = mujoco.Renderer(env.model, height=env.render_h, width=env.render_w) if (capture_frames_ep == eid) else None
        try:
            for t in range(env.MAX_STEPS):
                w_raw = env.last_w_avg + drift_vec
                w_scaled = (w_raw / CANON_WRENCH_SCALE).astype(np.float32)
                p_cur = env.get_proprio10d()
                p_rel = p_cur.copy()
                p_rel[:3] = (p_cur[:3] - p0) / 0.15
                prop = np.concatenate([p_rel, prev_a_scaled[0]], axis=-1).astype(np.float32)

                w_hist.append(w_scaled)
                pa_hist.append(prev_a_scaled[0].copy())

                if mode == "B0_ProprioOnly":
                    feat = prop
                elif mode == "B1b_RawWrench6D_E2E":
                    feat = np.concatenate([prop, w_scaled], axis=-1)
                else:
                    assert encoder is not None
                    wt = torch.from_numpy(np.stack(w_hist, axis=0)).unsqueeze(0).to(device)
                    pt = torch.from_numpy(np.stack(pa_hist, axis=0)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        z_seq = encoder.encode(wt, pt)
                        z_cur_t = z_seq[:, -1, :]
                        z_cur = z_cur_t.squeeze(0).cpu().numpy()
                    feat = np.concatenate([prop, z_cur], axis=-1)

                with torch.no_grad():
                    xt = (torch.from_numpy(feat).to(device) - xm_t) / xs_t
                    a_pred_scaled = net(xt.unsqueeze(0)).squeeze(0).cpu().numpy()

                act_canon = (a_pred_scaled * CANON_ACT_SCALE).astype(np.float32)

                # Compute 1-step PAEC physical wrench surprise if applicable
                if isinstance(encoder, PAECModel):
                    with torch.no_grad():
                        at_t = torch.from_numpy(a_pred_scaled).unsqueeze(0).to(device)
                        dw_hat_next = encoder.predictor(z_cur_t, at_t, h_idx=0)[0][:, :6].squeeze(0).cpu().numpy()
                        z_prev_pred = dw_hat_next
                else:
                    z_prev_pred = None

                w_next_true_raw, _, haz_gt = env.step_action(act_canon)
                w_next_true_scaled = ((w_next_true_raw + drift_vec) / CANON_WRENCH_SCALE).astype(np.float32)
                dw_true = w_next_true_scaled - w_scaled

                if z_prev_pred is not None and t >= 8:
                    surprise_scores.append(float(np.linalg.norm(dw_true - z_prev_pred)))
                    raw_rate_scores.append(float(np.linalg.norm(dw_true)))
                    raw_norm_scores.append(float(np.linalg.norm(w_next_true_scaled)))
                    gt_labels.append(int(haz_gt))

                fz_hist.append(float(env.TOOL_TARE_Z - w_next_true_raw[2]))
                prev_a_scaled = a_pred_scaled[None, :].astype(np.float32)

                if renderer_ctx is not None and t in (0, 20, 42, 65, 88):
                    renderer_ctx.update_scene(env.data, camera="oblique_cam")
                    captured_imgs.append(renderer_ctx.render().copy())
        finally:
            if renderer_ctx is not None:
                renderer_ctx.close()

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

    out = {
        "success_rate": round(100.0 * float(np.mean(succs)), 2),
        "fn_rmse_n": round(float(np.mean(rmses)), 4),
        "peak_fn_n": round(float(np.mean(max_fns)), 3),
        "safety_viol_rate": round(100.0 * float(np.mean(viols)), 2),
        "mean_final_x": round(float(np.mean(prog_xs)), 4),
    }
    if roc_auc_score is not None and len(set(gt_labels)) == 2:
        out["surprise_auroc"] = round(float(roc_auc_score(gt_labels, surprise_scores)), 4)
        out["raw_rate_auroc"] = round(float(roc_auc_score(gt_labels, raw_rate_scores)), 4)
        out["raw_norm_auroc"] = round(float(roc_auc_score(gt_labels, raw_norm_scores)), 4)
    out["_captured_imgs"] = captured_imgs
    return out


def save_rollout_strip(imgs: List[np.ndarray], out_png: Path, title: str):
    if plt is None or not imgs:
        return
    fig, axes = plt.subplots(1, len(imgs), figsize=(3.4 * len(imgs), 3.0))
    if len(imgs) == 1:
        axes = [axes]
    steps = [0, 20, 42, 65, 88][: len(imgs)]
    for ax, im, st in zip(axes, imgs, steps):
        ax.imshow(im)
        ax.set_title(f"t={st} ({st*0.05:.2f}s)", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="CPU-only expert demo + render verification")
    parser.add_argument("--ckpt", type=str, default="", help="Path to paec_rh20t_a6000_best.pt")
    parser.add_argument("--rh20t-dir", type=str, default="", help="Optional RH20T shards to pretrain matched MAE/JEPA baselines")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n-eval", type=int, default=24)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Smoke test & Expert Demo collection
    t0 = time.time()
    n_demos_collect = 6 if args.smoke_test else max(args.budgets)
    demos = collect_mujoco_demos(n_demos_collect, seed=2026)
    exp_succ = 100.0 * float(np.mean([d["success"] for d in demos]))
    exp_rmse = float(np.mean([d["fn_rmse"] for d in demos]))
    exp_maxf = float(np.max([d["max_fn"] for d in demos]))
    print(f"[Expert Demos] N={len(demos)} | success={exp_succ:.1f}% | fn_rmse={exp_rmse:.3f}N | max_fn={exp_maxf:.2f}N | t={time.time()-t0:.1f}s", flush=True)

    if args.smoke_test:
        for d in demos:
            print(f"  ep{d['ep_id']}: x={d['final_x']:.3f} maxF={d['max_fn']:.2f} meanF={d['mean_fn']:.2f} cFrac={d['c_frac']:.2f} head={d['fz_head']} tail={d['fz_tail']}")
        # Render expert strip cleanly
        env = CompliantWipingEnv()
        env.reset(np.random.default_rng(777))
        imgs = []
        with mujoco.Renderer(env.model, height=360, width=480) as r:
            for t in range(env.MAX_STEPS):
                act = expert_compliant_policy(env, env.get_wrench_tcp(), t)
                env.step_action(act)
                if t in (0, 20, 42, 65, 88):
                    r.update_scene(env.data, camera="oblique_cam")
                    imgs.append(r.render().copy())
        save_rollout_strip(imgs, out_dir / "mujoco_smoke_expert.png", "MuJoCo Compliant Wiping & Snag Strip (Expert Smoke Test)")
        print("Saved smoke test rollout strip:", out_dir / "mujoco_smoke_expert.png")
        return

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load frozen RH20T-10k PAEC checkpoints & optionally pretrain matched MAE/JEPA baselines
    encoders: Dict[str, Optional[nn.Module]] = {"B0_ProprioOnly": None, "B1b_RawWrench6D_E2E": None}
    pre_hashes: Dict[str, str] = {}

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    for k in ("PAEC_Tiny_d16", "PAEC_Small_d32", "PAEC_Base_d32"):
        if k in ckpt["state_dicts"]:
            m = build_policy_model_for_key(k, device)
            m.load_state_dict(ckpt["state_dicts"][k])
            m.eval()
            encoders[k] = m
            pre_hashes[k] = sha256_model(m)

    if args.rh20t_dir:
        base_ckpt_p = out_dir / "matched_rh20t_baselines.pt"
        if base_ckpt_p.exists():
            bckpt = torch.load(base_ckpt_p, map_location="cpu", weights_only=False)
            for bk in ("B2b_MAE_Small_d16", "B3b_JEPA_Small_d16"):
                m = build_policy_model_for_key(bk, device)
                m.load_state_dict(bckpt[bk])
                m.eval()
                encoders[bk] = m
                pre_hashes[bk] = sha256_model(m)
        else:
            print("[Baselines] Pretraining matched B2b_MAE_Small_d16 & B3b_JEPA_Small_d16 on identical RH20T train_pool...", flush=True)
            rh_eps, _ = load_rh20t_shards(Path(args.rh20t_dir))
            sp = make_zero_leakage_splits(rh_eps, seed=2026)
            bsave = {}
            for bk in ("B2b_MAE_Small_d16", "B3b_JEPA_Small_d16"):
                m = build_policy_model_for_key(bk, device)
                pretrain_on_episodes(f"MuJoCoMatch/{bk}", m, rh_eps, sp["train_pool"], device, steps=8000, batch_size=128, seed=42)
                m.eval()
                encoders[bk] = m
                pre_hashes[bk] = sha256_model(m)
                bsave[bk] = {kk: vv.detach().cpu() for kk, vv in m.state_dict().items()}
            torch.save(bsave, base_ckpt_p)

    # 3. Closed-loop evaluation across budgets x seeds x Clean / Drift-OOD
    bench_res: Dict[str, Any] = {
        "expert_demos": {"n": len(demos), "success_rate": exp_succ, "fn_rmse_n": exp_rmse, "max_fn_n": exp_maxf},
        "rh20t_ckpt_n_train_episodes": int(ckpt.get("n_train_episodes", 0)),
        "strictly_frozen_pass": True,
        "methods": {},
    }

    for mode, enc in encoders.items():
        bench_res["methods"][mode] = {}
        for nb in args.budgets:
            sub_demos = demos[:nb]
            clean_runs, drift_runs = [], []
            for sd in args.seeds:
                net, xm, xs = train_closedloop_bc(sub_demos, mode, enc, device, seed=sd, epochs=80)
                cap = 0 if (nb == 10 and sd == args.seeds[0] and mode in ("B0_ProprioOnly", "B1b_RawWrench6D_E2E", "PAEC_Small_d32")) else None
                c_res = evaluate_closed_loop(net, xm, xs, mode, enc, device, n_eval=args.n_eval, eval_seed=9100, drift_sigma=0.0, capture_frames_ep=cap)
                imgs_c = c_res.pop("_captured_imgs", [])
                if imgs_c:
                    save_rollout_strip(imgs_c, out_dir / f"rollout_{mode}_10shot_clean.png", f"{mode} (10-Shot Clean Rollout, Succ={c_res['success_rate']:.0f}%)")
                d_res = evaluate_closed_loop(net, xm, xs, mode, enc, device, n_eval=args.n_eval, eval_seed=9100, drift_sigma=1.5, capture_frames_ep=cap)
                imgs_d = d_res.pop("_captured_imgs", [])
                if imgs_d:
                    save_rollout_strip(imgs_d, out_dir / f"rollout_{mode}_10shot_drift.png", f"{mode} (10-Shot +1.5σ Drift OOD, Succ={d_res['success_rate']:.0f}%)")
                clean_runs.append(c_res)
                drift_runs.append(d_res)

            def _agg(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
                agg = {}
                for metric in runs[0].keys():
                    vals = [r[metric] for r in runs]
                    agg[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
                    agg[f"{metric}_std"] = round(float(np.std(vals)), 4)
                return agg

            bench_res["methods"][mode][f"{nb}demos"] = {"clean": _agg(clean_runs), "drift_ood": _agg(drift_runs)}
            print(
                f"[{mode:20s} | {nb:2d} demos] Clean Succ={bench_res['methods'][mode][f'{nb}demos']['clean']['success_rate_mean']:5.1f}% "
                f"RMSE={bench_res['methods'][mode][f'{nb}demos']['clean']['fn_rmse_n_mean']:.2f}N | "
                f"Drift Succ={bench_res['methods'][mode][f'{nb}demos']['drift_ood']['success_rate_mean']:5.1f}% "
                f"RMSE={bench_res['methods'][mode][f'{nb}demos']['drift_ood']['fn_rmse_n_mean']:.2f}N",
                flush=True,
            )

    # Verify strict SHA-256 weight immutability
    for k, h_before in pre_hashes.items():
        if sha256_model(encoders[k]) != h_before:
            bench_res["strictly_frozen_pass"] = False
            raise RuntimeError(f"Frozen weight hash mismatch for {k}!")

    with open(out_dir / "mujoco_closedloop_results.json", "w", encoding="utf-8") as f:
        json.dump(bench_res, f, indent=2, ensure_ascii=False)
    print(f"Saved MuJoCo closed-loop benchmark to {out_dir / 'mujoco_closedloop_results.json'} (strictly_frozen_pass={bench_res['strictly_frozen_pass']})", flush=True)


if __name__ == "__main__":
    main()
