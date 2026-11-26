#!/usr/bin/env python3
"""
V4.7 Industrial Defect-Inspection MuJoCo Benchmark Suite (3 Tasks, Robotics-Plugin Standards).

Three complementary industrial defect-inspection & micro-force tasks:
  1. BatteryScanEnv (New-Energy EV Battery Cell):
       Scan pouch cell across unknown elevation ([-18,+18] mm) & tilt ([-0.055,+0.055] rad)
       with hidden soft swelling (k_soft in [420, 680] N/m) or weld burr (mu in [1.8, 2.3]).
       Strict EV safety cap: max_fn <= 11.5 N.
  2. AutoConnectorEnv (Automotive Connector Blind Insertion & Jam Discrimination):
       Zoomed workspace RGB (128x128) + stiff industrial Z-servo (kp_z=900 N/m);
       detect hidden internal burr/pin jam (~12 mm depth) and yield softly
       (max_fn <= 15.0 N, final_fz <= 3.0 N) without pin crush or state aliasing.
  3. SemiGraspEnv (Semiconductor Fragile Carrier Micro-Force Anti-Slip Transfer):
       Narrow micro-fingers grasping off-center carrier (x0 in [-26,+26] mm, hidden mass,
       eccentric CoM, unknown friction) and placing onto variable-height chuck (z_chuck in [0,+16] mm)
       under strict summed per-finger & wrist micro-force cap (<= 14.0 N).
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _rot6d_from_rotmat(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1).astype(np.float32)


def _save_strip(imgs: List[np.ndarray], steps: List[int], out_png: Path, title: str):
    if plt is None or not imgs:
        return
    fig, axes = plt.subplots(1, len(imgs), figsize=(3.4 * len(imgs), 3.0))
    if len(imgs) == 1:
        axes = [axes]
    for ax, im, st in zip(axes, imgs, steps):
        ax.imshow(im)
        ax.set_title(f"t={st} ({st * 0.05:.2f}s)", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Task 1: EV Battery Cell Compliant Scanning & Swelling/Burr Detection
# ======================================================================
class BatteryScanEnv:
    DT_SIM = 0.002
    SUBSTEPS = 25
    MAX_STEPS = 105
    X_START = -0.125
    X_TARGET = 0.115
    TARGET_FN = 4.5
    SAFE_LIMIT_N = 11.5
    TOOL_TARE_Z = 2.73

    def __init__(self, rgb_w: int = 64, rgb_h: int = 64):
        self.rgb_w = rgb_w
        self.rgb_h = rgb_h
        self._renderer: Optional[mujoco.Renderer] = None
        self._build_spec("normal", 0.0, 1900.0, 0.0, 0.0, 600.0, 0.38, 2.0)

    def _close_renderer(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def _build_spec(
        self,
        defect_type: str,
        z0: float,
        k_base: float,
        tilt: float,
        defect_x: float,
        k_soft: float,
        base_mu: float,
        burr_mu: float,
    ):
        self._close_renderer()
        spec = mujoco.MjSpec()
        spec.option.timestep = self.DT_SIM
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10.0
        spec.default.geom.solref = [0.025, 1.5]
        spec.default.joint.armature = 0.01
        spec.default.joint.damping = [0.35, 0.35, 0.35]

        spec.worldbody.add_light(
            pos=[0.12, -0.35, 0.85],
            dir=[-0.12, 0.35, -1.0],
            diffuse=[0.95, 0.95, 0.95],
            ambient=[0.38, 0.38, 0.38],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
        spec.worldbody.add_camera(
            name="oblique_cam",
            pos=[0.24, -0.34, 0.24],
            xyaxes=[0.81, 0.58, 0, -0.27, 0.38, 0.88],
            fovy=45,
        )

        spec.worldbody.add_geom(
            name="table_visual",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.35, 0.22, 0.015],
            pos=[0, 0, -0.06],
            contype=0,
            conaffinity=0,
            mass=0.0,
            rgba=[0.16, 0.18, 0.22, 1.0],
        )

        platen = spec.worldbody.add_body(
            name="platen",
            pos=[0.0, 0.0, z0],
            euler=[0.0, tilt, 0.0],
        )
        platen.add_joint(
            name="pz",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, 0, 1],
            stiffness=k_base,
            damping=max(65.0, 0.08 * k_base),
            springref=0.0,
            range=[-0.045, 0.045],
        )

        hw_cell = 0.145
        c_lo = defect_x - 0.027
        c_hi = defect_x + 0.027
        left_w = max(0.01, (c_lo - (-hw_cell)) * 0.5)
        left_cx = -hw_cell + left_w
        right_w = max(0.01, (hw_cell - c_hi) * 0.5)
        right_cx = hw_cell - right_w

        cell_rgba = [0.62, 0.72, 0.82, 1.0]
        for nm, cx, hx in [("cell_left", left_cx, left_w - 0.0005), ("cell_right", right_cx, right_w - 0.0005)]:
            platen.add_geom(
                name=nm,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[hx, 0.060, 0.012],
                pos=[cx, 0.0, -0.012],
                mass=0.30,
                condim=4,
                friction=[base_mu, 0.015, 0.0005],
                rgba=cell_rgba,
            )

        if defect_type == "swelling":
            sub = platen.add_body(name="swell_pad", pos=[defect_x, 0.0, 0.0072])
            k_sub = max(350.0, (k_base * k_soft) / max(k_base - k_soft, 180.0))
            sub.add_joint(
                name="sz_swell",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=[0, 0, 1],
                stiffness=k_sub,
                damping=max(42.0, 0.08 * k_sub),
                springref=0.0,
                range=[-0.0070, 0.005],
            )
            sub.add_geom(
                name="cell_center",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.0265, 0.059, 0.012],
                pos=[0.0, 0.0, -0.012],
                mass=0.08,
                condim=4,
                friction=[base_mu, 0.015, 0.0005],
                rgba=[0.64, 0.74, 0.84, 1.0],
            )
        else:
            platen.add_geom(
                name="cell_center",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.0265, 0.060, 0.012],
                pos=[defect_x, 0.0, -0.012],
                mass=0.12,
                condim=4,
                friction=[base_mu, 0.015, 0.0005],
                rgba=cell_rgba,
            )
            if defect_type == "burr":
                platen.add_geom(
                    name="weld_burr",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.014, 0.056, 0.0003],
                    pos=[defect_x, 0.0, 0.0003],
                    mass=0.02,
                    condim=4,
                    friction=[burr_mu, 0.03, 0.001],
                    rgba=[0.54, 0.62, 0.70, 1.0],
                )

        platen.add_geom(
            name="tab_pos",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.012, 0.018, 0.002],
            pos=[-0.157, 0.025, -0.008],
            contype=0,
            conaffinity=0,
            mass=0.0,
            rgba=[0.85, 0.55, 0.25, 1.0],
        )
        platen.add_geom(
            name="tab_neg",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.012, 0.018, 0.002],
            pos=[-0.157, -0.025, -0.008],
            contype=0,
            conaffinity=0,
            mass=0.0,
            rgba=[0.78, 0.80, 0.84, 1.0],
        )

        bx = spec.worldbody.add_body(name="arm_x", pos=[0.0, 0.0, 0.18], mass=0.5, inertia=[1e-3, 1e-3, 1e-3])
        bx.add_joint(name="jx", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[-0.28, 0.28], damping=3.0)
        by = bx.add_body(name="arm_y", pos=[0, 0, 0], mass=0.5, inertia=[1e-3, 1e-3, 1e-3])
        by.add_joint(name="jy", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], range=[-0.15, 0.15], damping=3.0)
        bz = by.add_body(name="arm_z", pos=[0, 0, 0], mass=0.5, inertia=[1e-3, 1e-3, 1e-3])
        bz.add_joint(name="jz", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.25, 0.10], damping=4.0)
        bp = bz.add_body(name="wrist_pitch", pos=[0, 0, 0])
        bp.add_joint(name="jp", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-0.55, 0.55], damping=0.6)
        bp.add_geom(name="wrist_mount", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.022, 0.016], mass=0.3, contype=0, conaffinity=0, rgba=[0.22, 0.25, 0.30, 1.0])

        tool = bp.add_body(name="tool", pos=[0, 0, -0.038])
        tool.add_site(name="wrist_ft_site", pos=[0, 0, 0.018], size=[0.01])
        tool.add_site(name="tcp_site", pos=[0, 0, -0.055], size=[0.008], rgba=[0.1, 0.9, 0.3, 0.8])
        tool.add_geom(name="probe_stem", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.010, 0.022], pos=[0, 0, -0.015], mass=0.15, contype=0, conaffinity=0, rgba=[0.72, 0.74, 0.78, 1.0])
        tool.add_geom(name="probe_tip", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.014, 0.018], pos=[0, 0, -0.042], mass=0.15, condim=4, friction=[0.55, 0.02, 0.001], rgba=[0.10, 0.52, 0.85, 1.0])

        for jn, kp, dr, rng, flim in [
            ("jx", 450.0, 1.0, [-0.28, 0.28], 40.0),
            ("jy", 450.0, 1.0, [-0.15, 0.15], 40.0),
            ("jz", 240.0, 1.3, [-0.25, 0.10], 40.0),
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
        defect_type = str(rng.choice(["normal", "swelling", "burr"]))
        z0 = float(rng.uniform(-0.015, 0.015))
        k_base = float(rng.uniform(1600.0, 2200.0))
        tilt = float(rng.uniform(-0.050, 0.050))
        defect_x = float(rng.uniform(-0.045, 0.045))
        k_soft = float(rng.uniform(420.0, 680.0))
        base_mu = float(rng.uniform(0.32, 0.45))
        burr_mu = float(rng.uniform(1.80, 2.30))

        self.ep_cfg = {
            "task": "battery_scan",
            "defect_type": defect_type,
            "is_defect": int(defect_type != "normal"),
            "z0": z0,
            "k_base": k_base,
            "tilt": tilt,
            "defect_x": defect_x,
            "k_soft": k_soft if defect_type == "swelling" else k_base,
            "base_mu": base_mu,
            "burr_mu": burr_mu,
        }
        self._build_spec(defect_type, z0, k_base, tilt, defect_x, k_soft, base_mu, burr_mu)
        mujoco.mj_resetData(self.model, self.data)

        # Fixed nominal hover height (no oracle z0/tilt height leak!)
        init_x = self.X_START
        init_jz = -0.042
        self.data.joint("jx").qpos[0] = init_x
        self.data.joint("jy").qpos[0] = 0.0
        self.data.joint("jz").qpos[0] = init_jz
        self.data.joint("jp").qpos[0] = 0.0
        self.ctrl = np.array([init_x, 0.0, init_jz, 0.0], dtype=np.float64)
        self.data.ctrl[:] = self.ctrl
        mujoco.mj_forward(self.model, self.data)
        self.step_idx = 0
        self.last_w_avg = self.get_wrench_tcp()
        return self.ep_cfg

    def get_wrench_tcp(self) -> np.ndarray:
        f = self.data.sensor("wrist_force").data.copy()
        t = self.data.sensor("wrist_torque").data.copy()
        return np.concatenate([f, t]).astype(np.float32)

    def get_proprio10d(self) -> np.ndarray:
        pos = self.data.site("tcp_site").xpos.copy().astype(np.float32)
        xmat = self.data.site("tcp_site").xmat.copy().reshape(1, 3, 3)
        return np.concatenate([pos, _rot6d_from_rotmat(xmat)[0], np.zeros(1, dtype=np.float32)]).astype(np.float32)

    def render_rgb(self, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        w = width or self.rgb_w
        h = height or self.rgb_h
        if self._renderer is None or self._renderer.width != w or self._renderer.height != h:
            self._close_renderer()
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._renderer.update_scene(self.data, camera="oblique_cam")
        return self._renderer.render().copy()

    def step_action(self, act7_canon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        dx = float(np.clip(act7_canon[0], -0.010, 0.010))
        dy = float(np.clip(act7_canon[1], -0.008, 0.008))
        dz = float(np.clip(act7_canon[2], -0.010, 0.010))
        dp = float(np.clip(act7_canon[4], -0.04, 0.04))

        self.ctrl[0] = float(np.clip(self.ctrl[0] + dx, -0.25, 0.25))
        self.ctrl[1] = float(np.clip(self.ctrl[1] + dy, -0.10, 0.10))
        self.ctrl[2] = float(np.clip(self.ctrl[2] + dz, -0.23, 0.06))
        self.ctrl[3] = float(np.clip(self.ctrl[3] + dp, -0.45, 0.45))
        self.data.ctrl[:] = self.ctrl

        w_accum = np.zeros(6, dtype=np.float64)
        for _ in range(self.SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            w_accum += self.get_wrench_tcp()
        self.last_w_avg = (w_accum / float(self.SUBSTEPS)).astype(np.float32)
        self.step_idx += 1

        tcp_x = float(self.data.site("tcp_site").xpos[0])
        in_defect = int(self.ep_cfg["is_defect"] and abs(tcp_x - self.ep_cfg["defect_x"]) <= 0.028)
        return self.last_w_avg.copy(), self.get_proprio10d(), in_defect

    def get_coarse_vis_xy(self, rng_noise: np.random.Generator) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    def expert_policy(self, w_tcp: np.ndarray, step_idx: int) -> np.ndarray:
        fz_norm = float(self.TOOL_TARE_Z - w_tcp[2])
        fx_shear = float(w_tcp[0])
        fy_lat = float(w_tcp[1])
        x_pos = float(self.data.site("tcp_site").xpos[0])
        y_pos = float(self.data.site("tcp_site").xpos[1])
        z_vel = float(self.data.joint("jz").qvel[0])

        if fz_norm < 1.2 and step_idx < 30:
            vx = 0.0010 if x_pos < 0.132 else 0.0
            dz = -0.0022
        else:
            vx = 0.0 if x_pos >= 0.132 else (0.0040 if abs(fx_shear) < 4.5 else 0.0028)
            target_fn = max(2.8, self.TARGET_FN - max(0.0, abs(fx_shear) - 3.5) * 0.25)
            dz = float(np.clip(0.00014 * (fz_norm - target_fn) - 0.004 * z_vel, -0.0020, 0.0020))
        dy = float(np.clip(-0.22 * y_pos + 0.0003 * fy_lat, -0.005, 0.005))
        dp = float(np.clip(-0.003 * np.tanh(float(w_tcp[4])), -0.012, 0.012))
        return np.array([vx, dy, dz, 0.0, dp, 0.0, 0.0], dtype=np.float32)

    def classical_policy(self, w_obs: np.ndarray, step_idx: int) -> np.ndarray:
        return self.expert_policy(w_obs, step_idx)

    def evaluate_episode(self, fz_hist: List[float], extra: Dict[str, Any]) -> Dict[str, Any]:
        fz = self.TOOL_TARE_Z - np.asarray(fz_hist, dtype=np.float64)
        max_fn = float(fz.max())
        c_frac = float((fz[10:] >= 1.5).mean())
        final_x = float(self.data.site("tcp_site").xpos[0])
        viol = bool(max_fn > self.SAFE_LIMIT_N)
        succ = bool((final_x >= self.X_TARGET) and (c_frac >= 0.72) and (not viol))
        return {
            "success": succ,
            "safety_viol": viol,
            "peak_fn": max_fn,
            "fn_rmse": float(np.sqrt(np.mean((fz[10:] - self.TARGET_FN) ** 2))),
            "final_x": final_x,
            "c_frac": c_frac,
        }


# ======================================================================
# Task 2: Automotive Connector Blind Insertion & Burr-Jam Discrimination
# ======================================================================
class AutoConnectorEnv:
    DT_SIM = 0.002
    SUBSTEPS = 25
    MAX_STEPS = 85
    SAFE_LIMIT_N = 15.0
    TOOL_TARE_Z = 2.73
    SOCKET_TOP_Z = 0.032

    def __init__(self, rgb_w: int = 64, rgb_h: int = 64):
        self.rgb_w = rgb_w
        self.rgb_h = rgb_h
        self._renderer: Optional[mujoco.Renderer] = None
        self._build_spec("normal", 0.0, 0.0, 0.22)

    def _close_renderer(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def _build_spec(self, defect_type: str, sock_x: float, sock_y: float, wall_mu: float):
        self._close_renderer()
        spec = mujoco.MjSpec()
        spec.option.timestep = self.DT_SIM
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10.0
        spec.default.geom.solref = [0.020, 1.4]
        spec.default.joint.armature = 0.01

        spec.worldbody.add_light(
            pos=[0.12, -0.28, 0.75],
            dir=[-0.12, 0.28, -1.0],
            diffuse=[0.95, 0.95, 0.95],
            ambient=[0.42, 0.42, 0.42],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
        # Zoomed camera (fovy=28) so +/-12mm socket offset spans ~28 px in 128x128 RGB
        spec.worldbody.add_camera(
            name="oblique_cam",
            pos=[0.09, -0.14, 0.16],
            xyaxes=[0.841, 0.541, 0.0, -0.341, 0.530, 0.776],
            fovy=28,
        )

        spec.worldbody.add_geom(
            name="table_visual",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.30, 0.22, 0.015],
            pos=[0, 0, -0.015],
            mass=0.0,
            condim=4,
            rgba=[0.18, 0.20, 0.25, 1.0],
        )

        sock = spec.worldbody.add_body(name="socket", pos=[sock_x, sock_y, 0.0])
        inner = 0.0175
        wall_t = 0.007
        wall_h = 0.016
        dark_grey = [0.22, 0.24, 0.28, 1.0]
        walls = [
            ("w_px", [inner + wall_t, 0, wall_h], [wall_t, inner + 2 * wall_t, wall_h]),
            ("w_nx", [-(inner + wall_t), 0, wall_h], [wall_t, inner + 2 * wall_t, wall_h]),
            ("w_py", [0, inner + wall_t, wall_h], [inner, wall_t, wall_h]),
            ("w_ny", [0, -(inner + wall_t), wall_h], [inner, wall_t, wall_h]),
        ]
        for wn, wp, ws in walls:
            sock.add_geom(
                name=wn,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=ws,
                pos=wp,
                mass=0.0,
                condim=4,
                friction=[wall_mu, 0.01, 0.0002],
                rgba=dark_grey,
            )
        for cname, cpos in [("collar_px", [inner + wall_t, 0, self.SOCKET_TOP_Z + 0.001]), ("collar_nx", [-(inner + wall_t), 0, self.SOCKET_TOP_Z + 0.001])]:
            sock.add_geom(
                name=cname,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.0035, inner + 2 * wall_t, 0.0012],
                pos=cpos,
                contype=0,
                conaffinity=0,
                mass=0.0,
                rgba=[0.95, 0.74, 0.12, 1.0],
            )
        sock.add_geom(
            name="socket_floor",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[inner, inner, 0.003],
            pos=[0, 0, 0.003],
            mass=0.0,
            condim=4,
            friction=[0.4, 0.01, 0.0002],
            rgba=[0.14, 0.16, 0.20, 1.0],
        )
        if defect_type == "jam_burr":
            sock.add_geom(
                name="internal_jam",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.013, 0.013, 0.002],
                pos=[0.0, 0.0, 0.018],
                mass=0.0,
                condim=4,
                friction=[0.6, 0.01, 0.0002],
                rgba=[0.16, 0.18, 0.22, 1.0],
            )

        bx = spec.worldbody.add_body(name="arm_x", pos=[0.0, 0.0, 0.16], mass=0.5, inertia=[1e-3, 1e-3, 1e-3])
        bx.add_joint(name="jx", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[-0.12, 0.12], damping=3.0)
        by = bx.add_body(name="arm_y", pos=[0, 0, 0], mass=0.5, inertia=[1e-3, 1e-3, 1e-3])
        by.add_joint(name="jy", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], range=[-0.12, 0.12], damping=3.0)
        bz = by.add_body(name="arm_z", pos=[0, 0, 0], mass=0.5, inertia=[1e-3, 1e-3, 1e-3])
        bz.add_joint(name="jz", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.18, 0.08], damping=5.0)

        tool = bz.add_body(name="tool", pos=[0, 0, -0.040])
        tool.add_site(name="wrist_ft_site", pos=[0, 0, 0.018], size=[0.01])
        tool.add_site(name="tcp_site", pos=[0, 0, -0.056], size=[0.006], rgba=[0.1, 0.9, 0.3, 0.8])
        tool.add_geom(name="plug_stem", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.011, 0.020], pos=[0, 0, -0.012], mass=0.14, contype=0, conaffinity=0, rgba=[0.72, 0.75, 0.78, 1.0])
        tool.add_geom(name="plug_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0145, 0.0145, 0.014], pos=[0, 0, -0.042], mass=0.14, condim=4, friction=[wall_mu, 0.01, 0.0002], rgba=[0.12, 0.45, 0.86, 1.0])

        # Industrial stiff Z servo (kp_z=900) so unyielding slams exceed 15 N safety limit
        for jn, kp, dr, rng, flim in [
            ("jx", 420.0, 1.1, [-0.12, 0.12], 40.0),
            ("jy", 420.0, 1.1, [-0.12, 0.12], 40.0),
            ("jz", 900.0, 1.2, [-0.18, 0.08], 50.0),
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
        defect_type = str(rng.choice(["normal", "jam_burr"]))
        sock_x = float(rng.uniform(-0.012, 0.012))
        sock_y = float(rng.uniform(-0.012, 0.012))
        wall_mu = float(rng.uniform(0.18, 0.28))

        self.ep_cfg = {
            "task": "auto_connector",
            "defect_type": defect_type,
            "is_defect": int(defect_type == "jam_burr"),
            "sock_x": sock_x,
            "sock_y": sock_y,
            "wall_mu": wall_mu,
        }
        self._build_spec(defect_type, sock_x, sock_y, wall_mu)
        mujoco.mj_resetData(self.model, self.data)

        init_jz = -0.014  # tcp_z = 0.050 (18 mm above socket rim)
        self.data.joint("jx").qpos[0] = 0.0
        self.data.joint("jy").qpos[0] = 0.0
        self.data.joint("jz").qpos[0] = init_jz
        self.ctrl = np.array([0.0, 0.0, init_jz], dtype=np.float64)
        self.data.ctrl[:] = self.ctrl
        mujoco.mj_forward(self.model, self.data)
        self.step_idx = 0
        self._jam_detected = False
        self._jam_hold_depth = 0.0
        self.last_w_avg = self.get_wrench_tcp()
        return self.ep_cfg

    def get_wrench_tcp(self) -> np.ndarray:
        f = self.data.sensor("wrist_force").data.copy()
        t = self.data.sensor("wrist_torque").data.copy()
        return np.concatenate([f, t]).astype(np.float32)

    def get_proprio10d(self) -> np.ndarray:
        pos = self.data.site("tcp_site").xpos.copy().astype(np.float32)
        xmat = self.data.site("tcp_site").xmat.copy().reshape(1, 3, 3)
        return np.concatenate([pos, _rot6d_from_rotmat(xmat)[0], np.zeros(1, dtype=np.float32)]).astype(np.float32)

    def render_rgb(self, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        w = width or self.rgb_w
        h = height or self.rgb_h
        if self._renderer is None or self._renderer.width != w or self._renderer.height != h:
            self._close_renderer()
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._renderer.update_scene(self.data, camera="oblique_cam")
        return self._renderer.render().copy()

    def step_action(self, act7_canon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        dx = float(np.clip(act7_canon[0], -0.008, 0.008))
        dy = float(np.clip(act7_canon[1], -0.008, 0.008))
        dz = float(np.clip(act7_canon[2], -0.008, 0.008))
        self.ctrl[0] = float(np.clip(self.ctrl[0] + dx, -0.08, 0.08))
        self.ctrl[1] = float(np.clip(self.ctrl[1] + dy, -0.08, 0.08))
        self.ctrl[2] = float(np.clip(self.ctrl[2] + dz, -0.065, 0.03))
        self.data.ctrl[:] = self.ctrl

        w_accum = np.zeros(6, dtype=np.float64)
        for _ in range(self.SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            w_accum += self.get_wrench_tcp()
        self.last_w_avg = (w_accum / float(self.SUBSTEPS)).astype(np.float32)
        self.step_idx += 1

        tcp_z = float(self.data.site("tcp_site").xpos[2])
        depth = max(0.0, self.SOCKET_TOP_Z - tcp_z)
        in_jam = int(self.ep_cfg["is_defect"] and depth >= 0.009)
        return self.last_w_avg.copy(), self.get_proprio10d(), in_jam

    def get_coarse_vis_xy(self, rng_noise: np.random.Generator) -> np.ndarray:
        eps = np.clip(rng_noise.normal(0.0, 0.0020, size=2), -0.0032, 0.0032)
        return np.array([self.ep_cfg["sock_x"] + eps[0], self.ep_cfg["sock_y"] + eps[1]], dtype=np.float32)

    def expert_policy(self, w_tcp: np.ndarray, step_idx: int) -> np.ndarray:
        tcp = self.data.site("tcp_site").xpos
        fz_norm = float(self.TOOL_TARE_Z - w_tcp[2])
        depth = max(0.0, self.SOCKET_TOP_Z - float(tcp[2]))

        ex = float(self.ep_cfg["sock_x"] - tcp[0])
        ey = float(self.ep_cfg["sock_y"] - tcp[1])
        dx = float(np.clip(0.55 * ex, -0.0025, 0.0025))
        dy = float(np.clip(0.55 * ey, -0.0025, 0.0025))

        if self._jam_detected:
            # Markov-clean jam backoff: lift 2.5 mm to unload pin force, then hold steadily inside mouth
            dx, dy = 0.0, 0.0
            if depth > self._jam_hold_depth or fz_norm > 1.5:
                dz = 0.0012
            else:
                dz = 0.0
        elif step_idx < 14 and (abs(ex) > 0.0015 or abs(ey) > 0.0015):
            dz = 0.0
        elif depth < 0.0185 and fz_norm > 2.8:
            self._jam_detected = True
            self._jam_hold_depth = max(0.0070, depth - 0.0025)
            dz = 0.0012
        elif depth >= 0.0235 and fz_norm >= 2.0:
            dz = float(np.clip(0.00010 * (fz_norm - 3.5), -0.0008, 0.0008))
        else:
            dz = -0.0010 if fz_norm < 2.8 else float(np.clip(0.00012 * (fz_norm - 3.5), -0.0006, 0.0008))

        return np.array([dx, dy, dz, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def classical_policy(self, w_obs: np.ndarray, step_idx: int) -> np.ndarray:
        """Classical blind constant-force insertion at kp_z=900 (no RGB XY centering or jam unload)."""
        fz_norm = float(self.TOOL_TARE_Z - w_obs[2])
        if step_idx < 10:
            dz = -0.0012
        else:
            dz = float(np.clip(0.00018 * (fz_norm - 4.5), -0.0014, 0.0014))
        return np.array([0.0, 0.0, dz, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def evaluate_episode(self, fz_hist: List[float], extra: Dict[str, Any]) -> Dict[str, Any]:
        fz = self.TOOL_TARE_Z - np.asarray(fz_hist, dtype=np.float64)
        depths = np.asarray(extra["depths"], dtype=np.float64)
        max_fn = float(fz.max())
        max_depth = float(depths.max())
        final_depth = float(depths[-1])
        final_fz = float(fz[-1])
        viol = bool(max_fn > self.SAFE_LIMIT_N)

        if self.ep_cfg["is_defect"] == 0:
            succ = bool((max_depth >= 0.021) and (not viol))
        else:
            # Engaged socket (>= 7 mm), detected jam, unloaded pin force (final_fz <= 3.0 N), zero pin crush
            succ = bool((max_depth >= 0.007) and (final_fz <= 3.0) and (not viol))
        return {
            "success": succ,
            "safety_viol": viol,
            "peak_fn": max_fn,
            "max_depth_mm": round(1000.0 * max_depth, 2),
            "final_depth_mm": round(1000.0 * final_depth, 2),
            "final_fz": round(final_fz, 2),
        }


# ======================================================================
# Task 3: Semiconductor Fragile Carrier Micro-Force Anti-Slip Transfer
# ======================================================================
class SemiGraspEnv:
    DT_SIM = 0.002
    SUBSTEPS = 25
    MAX_STEPS = 95
    CHUCK_X = 0.085
    SAFE_LIMIT_N = 14.0
    GRIPPER_TARE_Z = 3.924  # 0.40 kg gripper assembly below wrist_ft_site

    def __init__(self, rgb_w: int = 64, rgb_h: int = 64):
        self.rgb_w = rgb_w
        self.rgb_h = rgb_h
        self._renderer: Optional[mujoco.Renderer] = None
        self._build_spec(0.0, 0.28, 0.0, 0.50, 0.004)

    def _close_renderer(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def _build_spec(self, x0: float, mass: float, com_shift: float, carrier_mu: float, z_chuck: float):
        self._close_renderer()
        spec = mujoco.MjSpec()
        spec.option.timestep = self.DT_SIM
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10.0
        spec.default.geom.solref = [0.020, 1.4]
        spec.default.joint.armature = 0.01

        spec.worldbody.add_light(
            pos=[0.12, -0.35, 0.85],
            dir=[-0.12, 0.35, -1.0],
            diffuse=[0.95, 0.95, 0.95],
            ambient=[0.42, 0.42, 0.42],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
        # Closer oblique cam focusing on pick-and-place zone [x in -0.05..0.12]
        spec.worldbody.add_camera(
            name="oblique_cam",
            pos=[0.12, -0.22, 0.20],
            xyaxes=[0.896, 0.443, 0.0, -0.227, 0.459, 0.859],
            fovy=40,
        )

        spec.worldbody.add_geom(
            name="table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.30, 0.20, 0.015],
            pos=[0, 0, -0.015],
            mass=0.0,
            condim=4,
            friction=[0.55, 0.01, 0.0005],
            rgba=[0.18, 0.20, 0.24, 1.0],
        )
        # Variable-height inspection chuck pedestal (top at z = z_chuck + 0.002)
        chuck_h = 0.5 * (z_chuck + 0.002)
        spec.worldbody.add_geom(
            name="chuck_pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.042, 0.042, max(0.001, chuck_h)],
            pos=[self.CHUCK_X, 0.0, max(0.001, chuck_h)],
            mass=0.0,
            condim=4,
            friction=[0.65, 0.01, 0.0005],
            rgba=[0.16, 0.62, 0.42, 1.0],
        )

        car = spec.worldbody.add_body(name="carrier", pos=[x0, 0.0, 0.022])
        car.add_freejoint(name="carrier_free")
        m_main = max(0.08, mass * 0.65)
        m_bal = max(0.04, mass * 0.35)
        car.add_geom(
            name="carrier_box",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.024, 0.020, 0.022],
            pos=[0.0, 0.0, 0.0],
            mass=m_main,
            condim=4,
            friction=[carrier_mu, 0.01, 0.0004],
            rgba=[0.74, 0.78, 0.84, 1.0],
        )
        car.add_geom(
            name="carrier_ballast",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.008, 0.010, 0.010],
            pos=[com_shift, 0.0, 0.0],
            mass=m_bal,
            contype=0,
            conaffinity=0,
            rgba=[0.74, 0.78, 0.84, 0.0],
        )

        bx = spec.worldbody.add_body(name="arm_x", pos=[0.0, 0.0, 0.16], mass=0.5, inertia=[1e-3, 1e-3, 1e-3])
        bx.add_joint(name="jx", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[-0.12, 0.18], damping=3.5)
        by = bx.add_body(name="arm_y", pos=[0, 0, 0], mass=0.5, inertia=[1e-3, 1e-3, 1e-3])
        by.add_joint(name="jy", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], range=[-0.10, 0.10], damping=3.5)
        bz = by.add_body(name="arm_z", pos=[0, 0, 0], mass=0.5, inertia=[1e-3, 1e-3, 1e-3])
        bz.add_joint(name="jz", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.14, 0.08], damping=4.5)

        grip_base = bz.add_body(name="gripper_base", pos=[0, 0, -0.020])
        grip_base.add_site(name="wrist_ft_site", pos=[0, 0, 0.012], size=[0.01])
        grip_base.add_site(name="tcp_site", pos=[0, 0, -0.056], size=[0.006], rgba=[0.1, 0.9, 0.3, 0.8])
        grip_base.add_geom(name="palm", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.044, 0.010], pos=[0, 0, -0.012], mass=0.22, contype=0, conaffinity=0, rgba=[0.24, 0.27, 0.32, 1.0])

        for nm, ax, sign in [("finger_l", [0, -1, 0], -1.0), ("finger_r", [0, 1, 0], 1.0)]:
            fb = grip_base.add_body(name=nm, pos=[0, 0, 0])
            jn = f"j_{nm}"
            fb.add_joint(name=jn, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=ax, range=[0.008, 0.044], damping=2.8)
            fb.add_geom(
                name=f"{nm}_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.011, 0.005, 0.022],
                pos=[0, sign * 0.005, -0.044],
                mass=0.09,
                condim=4,
                friction=[carrier_mu, 0.012, 0.0005],
                rgba=[0.12, 0.48, 0.85, 1.0],
            )

        for jn, kp, dr, rng, flim in [
            ("jx", 420.0, 1.1, [-0.12, 0.18], 40.0),
            ("jy", 420.0, 1.1, [-0.10, 0.10], 40.0),
            ("jz", 550.0, 1.2, [-0.14, 0.08], 45.0),
            ("j_finger_l", 1500.0, 1.2, [0.008, 0.044], 26.0),
            ("j_finger_r", 1500.0, 1.2, [0.008, 0.044], 26.0),
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
        x0 = float(rng.uniform(-0.026, 0.026))
        mass = float(rng.uniform(0.18, 0.45))
        com_shift = float(rng.uniform(-0.012, 0.012))
        carrier_mu = float(rng.uniform(0.38, 0.72))
        z_chuck = float(rng.uniform(0.000, 0.016))

        self.ep_cfg = {
            "task": "semi_grasp",
            "defect_type": "eccentric_slip_risk" if (abs(com_shift) > 0.005 or carrier_mu < 0.48) else "nominal",
            "is_defect": int(abs(com_shift) > 0.005 or carrier_mu < 0.48),
            "x0": x0,
            "mass": mass,
            "com_shift": com_shift,
            "carrier_mu": carrier_mu,
            "z_chuck": z_chuck,
        }
        self._build_spec(x0, mass, com_shift, carrier_mu, z_chuck)
        mujoco.mj_resetData(self.model, self.data)

        init_jz = -0.036
        init_g = 0.036
        self.data.joint("jx").qpos[0] = 0.0
        self.data.joint("jy").qpos[0] = 0.0
        self.data.joint("jz").qpos[0] = init_jz
        self.data.joint("j_finger_l").qpos[0] = init_g
        self.data.joint("j_finger_r").qpos[0] = init_g
        self.ctrl = np.array([0.0, 0.0, init_jz, init_g, init_g], dtype=np.float64)
        self.data.ctrl[:] = self.ctrl
        mujoco.mj_forward(self.model, self.data)
        self.step_idx = 0
        self._placed_touched = False
        self.fl_id = self.model.geom("finger_l_geom").id
        self.fr_id = self.model.geom("finger_r_geom").id
        self.last_w_avg = self.get_wrench_tcp()
        return self.ep_cfg

    def get_wrench_tcp(self) -> np.ndarray:
        f = self.data.sensor("wrist_force").data.copy()
        t = self.data.sensor("wrist_torque").data.copy()
        return np.concatenate([f, t]).astype(np.float32)

    def get_proprio10d(self) -> np.ndarray:
        pos = self.data.site("tcp_site").xpos.copy().astype(np.float32)
        xmat = self.data.site("tcp_site").xmat.copy().reshape(1, 3, 3)
        g_open = float(0.5 * (self.data.joint("j_finger_l").qpos[0] + self.data.joint("j_finger_r").qpos[0]))
        return np.concatenate([pos, _rot6d_from_rotmat(xmat)[0], np.array([g_open], dtype=np.float32)]).astype(np.float32)

    def render_rgb(self, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        w = width or self.rgb_w
        h = height or self.rgb_h
        if self._renderer is None or self._renderer.width != w or self._renderer.height != h:
            self._close_renderer()
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._renderer.update_scene(self.data, camera="oblique_cam")
        return self._renderer.render().copy()

    def get_coarse_vis_xy(self, rng_noise: np.random.Generator) -> np.ndarray:
        eps = np.clip(rng_noise.normal(0.0, 0.0020, size=2), -0.0032, 0.0032)
        return np.array([self.ep_cfg["x0"] + eps[0], float(eps[1])], dtype=np.float32)

    def _get_finger_contact_force(self) -> float:
        """Sum normal contact force across all contact points per finger (not corner max)."""
        cf = np.zeros(6, dtype=np.float64)
        f_l = 0.0
        f_r = 0.0
        for ci in range(self.data.ncon):
            con = self.data.contact[ci]
            has_l = con.geom1 == self.fl_id or con.geom2 == self.fl_id
            has_r = con.geom1 == self.fr_id or con.geom2 == self.fr_id
            if has_l or has_r:
                mujoco.mj_contactForce(self.model, self.data, ci, cf)
                fn = float(abs(cf[0]))
                if has_l:
                    f_l += fn
                if has_r:
                    f_r += fn
        return max(f_l, f_r)

    def step_action(self, act7_canon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        dx = float(np.clip(act7_canon[0], -0.008, 0.008))
        dy = float(np.clip(act7_canon[1], -0.008, 0.008))
        dz = float(np.clip(act7_canon[2], -0.008, 0.008))
        dg = float(np.clip(act7_canon[6], -0.0035, 0.0035))

        self.ctrl[0] = float(np.clip(self.ctrl[0] + dx, -0.10, 0.16))
        self.ctrl[1] = float(np.clip(self.ctrl[1] + dy, -0.08, 0.08))
        self.ctrl[2] = float(np.clip(self.ctrl[2] + dz, -0.082, 0.04))
        g_next = float(np.clip(self.ctrl[3] + dg, 0.0085, 0.040))
        self.ctrl[3] = g_next
        self.ctrl[4] = g_next
        self.data.ctrl[:] = self.ctrl

        w_accum = np.zeros(6, dtype=np.float64)
        f_finger_peak = 0.0
        for _ in range(self.SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            w_accum += self.get_wrench_tcp()
            f_finger_peak = max(f_finger_peak, self._get_finger_contact_force())
        self.last_w_avg = (w_accum / float(self.SUBSTEPS)).astype(np.float32)
        self.step_idx += 1
        return self.last_w_avg.copy(), self.get_proprio10d(), f_finger_peak

    def expert_policy(self, w_tcp: np.ndarray, step_idx: int) -> np.ndarray:
        """Strictly Markov state-conditioned expert policy (zero step_idx phase gating)."""
        tcp = self.data.site("tcp_site").xpos
        tcp_x, tcp_y = float(tcp[0]), float(tcp[1])
        jz = float(self.data.joint("jz").qpos[0])
        g_cur = float(self.ctrl[3])
        x0 = self.ep_cfg["x0"]
        fz_rel = float(self.GRIPPER_TARE_Z - w_tcp[2])  # negative = upward table/chuck contact on payload

        if tcp_x >= self.CHUCK_X - 0.015 and fz_rel < -1.2:
            self._placed_touched = True

        if self._placed_touched or (tcp_x >= self.CHUCK_X - 0.015 and g_cur > 0.024):
            # Release & retract clear above chuck
            dx, dy = 0.0, 0.0
            dg = 0.0032 if g_cur < 0.035 else 0.0
            dz = 0.0025 if g_cur >= 0.028 else 0.0
        elif g_cur > 0.0158 and tcp_x < 0.045:
            # Approach & close gently at pick location
            if jz <= -0.068 and abs(x0 - tcp_x) < 0.006:
                dx, dy, dz = 0.0, 0.0, 0.0
                dg = float(np.clip(0.0154 - g_cur, -0.0024, 0.0))
            else:
                dx = float(np.clip(0.55 * (x0 - tcp_x), -0.0045, 0.0045))
                dy = float(np.clip(-0.50 * tcp_y, -0.0030, 0.0030))
                dz = float(np.clip(0.45 * (-0.072 - jz), -0.0040, 0.0040)) if abs(x0 - tcp_x) < 0.008 else 0.0
                dg = 0.0
        elif tcp_x < 0.055 and jz < -0.006:
            # Lift payload cleanly above chuck range
            dx, dy = 0.0, 0.0
            dz = float(np.clip(0.45 * (0.004 - jz), -0.0040, 0.0040))
            dg = -0.0004 if (abs(float(w_tcp[4])) > 0.028 and g_cur > 0.0148) else 0.0
        elif tcp_x < self.CHUCK_X - 0.004:
            # Transfer horizontally above chuck
            dx = float(np.clip(0.45 * (self.CHUCK_X - tcp_x), -0.0045, 0.0045))
            dy = float(np.clip(-0.40 * tcp_y, -0.0020, 0.0020))
            dz = float(np.clip(0.35 * (0.004 - jz), -0.0020, 0.0020))
            dg = 0.0
        else:
            # Force-guided soft touchdown on unknown chuck height
            dx = float(np.clip(0.40 * (self.CHUCK_X - tcp_x), -0.0020, 0.0020))
            dy = 0.0
            dz = -0.0030
            dg = 0.0

        return np.array([dx, dy, dz, 0.0, 0.0, 0.0, dg], dtype=np.float32)

    def classical_policy(self, w_obs: np.ndarray, step_idx: int) -> np.ndarray:
        """
        Classical blind industrial pick-and-place assuming fixed center x=0, fixed chuck z,
        and high-force blind closing (exceeds micro-force limit or misses off-center x0).
        """
        g_cur = float(self.ctrl[3])
        if step_idx < 14:
            dx = float(np.clip(-0.4 * float(self.data.joint("jx").qpos[0]), -0.003, 0.003))
            dy = 0.0
            dz = float(np.clip(0.45 * (-0.072 - float(self.data.joint("jz").qpos[0])), -0.0035, 0.0035))
            dg = 0.0
        elif step_idx < 28:
            dx, dy, dz = 0.0, 0.0, 0.0
            dg = -0.0028 if g_cur > 0.0092 else 0.0
        elif step_idx < 44:
            dx, dy, dg = 0.0, 0.0, 0.0
            dz = float(np.clip(0.45 * (0.002 - float(self.data.joint("jz").qpos[0])), -0.0038, 0.0038))
        elif step_idx < 68:
            dx = float(np.clip(0.45 * (self.CHUCK_X - float(self.data.joint("jx").qpos[0])), -0.0040, 0.0040))
            dy, dz, dg = 0.0, 0.0, 0.0
        elif step_idx < 84:
            dx, dy, dg = 0.0, 0.0, 0.0
            dz = float(np.clip(0.40 * (-0.070 - float(self.data.joint("jz").qpos[0])), -0.0030, 0.0030))
        else:
            dx, dy, dz = 0.0, 0.0, 0.0022 if step_idx >= 87 else 0.0
            dg = 0.0028 if g_cur < 0.036 else 0.0
        return np.array([dx, dy, dz, 0.0, 0.0, 0.0, dg], dtype=np.float32)

    def evaluate_episode(self, fz_hist: List[float], extra: Dict[str, Any]) -> Dict[str, Any]:
        car_pos = self.data.body("carrier").xpos.copy()
        car_mat = self.data.body("carrier").xmat.copy().reshape(3, 3)
        tilt_cos = float(car_mat[2, 2])
        err_x = float(abs(car_pos[0] - self.CHUCK_X))
        err_y = float(abs(car_pos[1]))
        f_finger_max = float(max(extra["finger_forces"])) if extra.get("finger_forces") else 0.0
        w_fn_max = float(max(abs(float(self.GRIPPER_TARE_Z - x)) for x in fz_hist))
        peak_force = max(f_finger_max, w_fn_max)
        viol = bool(peak_force > self.SAFE_LIMIT_N)
        succ = bool((err_x <= 0.034) and (err_y <= 0.032) and (car_pos[2] >= 0.015) and (tilt_cos >= 0.90) and (not viol))
        return {
            "success": succ,
            "safety_viol": viol,
            "peak_fn": peak_force,
            "finger_peak_n": round(f_finger_max, 2),
            "wrist_peak_n": round(w_fn_max, 2),
            "place_err_mm": round(1000.0 * float(np.hypot(err_x, err_y)), 2),
            "tilt_cos": round(tilt_cos, 3),
        }


def make_env(task_name: str, rgb_w: int = 64, rgb_h: int = 64):
    if task_name == "battery_scan":
        return BatteryScanEnv(rgb_w=rgb_w, rgb_h=rgb_h)
    if task_name == "auto_connector":
        return AutoConnectorEnv(rgb_w=rgb_w, rgb_h=rgb_h)
    if task_name == "semi_grasp":
        return SemiGraspEnv(rgb_w=rgb_w, rgb_h=rgb_h)
    raise ValueError(f"Unknown task_name={task_name}")


def run_smoke_test(out_dir: Path, n_eps: int = 12):
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_idx, (tname, fname) in enumerate(
        [
            ("battery_scan", "smoke_task1_battery.png"),
            ("auto_connector", "smoke_task2_autoconnector.png"),
            ("semi_grasp", "smoke_task3_semigrasp.png"),
        ],
        start=1,
    ):
        env = make_env(tname, rgb_w=480, rgb_h=360)
        rng = np.random.default_rng(2026 + task_idx * 17)
        succs, viols, peaks = [], [], []
        strip_imgs, strip_steps = [], []

        for eid in range(n_eps):
            cfg = env.reset(rng)
            w_cur = env.last_w_avg.copy()
            fz_hist = []
            extra: Dict[str, Any] = {"depths": [], "finger_forces": []}
            cap_steps = [0, env.MAX_STEPS // 4, env.MAX_STEPS // 2, (3 * env.MAX_STEPS) // 4, env.MAX_STEPS - 1]
            for t in range(env.MAX_STEPS):
                if eid == 0 and t in cap_steps:
                    strip_imgs.append(env.render_rgb(480, 360))
                    strip_steps.append(t)
                act = env.expert_policy(w_cur, t)
                w_cur, prop_cur, sig = env.step_action(act)
                fz_hist.append(float(w_cur[2]))
                if tname == "auto_connector":
                    extra["depths"].append(max(0.0, env.SOCKET_TOP_Z - float(prop_cur[2])))
                elif tname == "semi_grasp":
                    extra["finger_forces"].append(float(sig))
            ev = env.evaluate_episode(fz_hist, extra)
            succs.append(float(ev["success"]))
            viols.append(float(ev["safety_viol"]))
            peaks.append(float(ev["peak_fn"]))
            print(f"  [{tname:15s} ep{eid:02d}] {cfg.get('defect_type', '')}: {ev}", flush=True)

        _save_strip(
            strip_imgs,
            strip_steps,
            out_dir / fname,
            f"V4.7 Task {task_idx}: {tname} Expert Smoke Test (Succ={100*np.mean(succs):.1f}%)",
        )
        print(
            f"[SUMMARY {tname:15s}] Succ={100*np.mean(succs):5.1f}% | Viol={100*np.mean(viols):5.1f}% | MeanPeak={np.mean(peaks):.2f}N | Saved {out_dir / fname}\n",
            flush=True,
        )
        env._close_renderer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n-eps", type=int, default=12)
    parser.add_argument("--out-dir", type=str, default="outputs/v4_7_smoke")
    args = parser.parse_args()
    if args.smoke_test:
        run_smoke_test(Path(args.out_dir), n_eps=args.n_eps)
