#!/usr/bin/env python3
"""
V4.8 Controlled Diagnostic Environment Variants.
Keeps envs_industrial_defect.py 100% untouched for V4.7 reproducibility.

Adds AutoConnectorChamferEnv(AutoConnectorEnv):
  - Identical kp_z=900 N/m, SAFE_LIMIT_N=15.0 N, internal jam_burr, and +/-3.2mm coarse_vis_xy noise.
  - Adds realistic 2.5mm 45-deg industrial lead-in chamfer bevels around the 4 top edges
    of the socket mouth so +/-3mm misaligned plugs generate lateral centering wrench cues.
  - Provides no-oracle recovery-aware expert_policy_no_oracle(w_tcp, step_idx, coarse_vis_xy)
    and non-strawman classical_guided_policy(w_obs, step_idx, coarse_vis_xy).
"""

import argparse
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np

from envs_industrial_defect import AutoConnectorEnv


class AutoConnectorChamferEnv(AutoConnectorEnv):
    """Automotive Connector with realistic 45-deg lead-in chamfer and no-oracle force recovery."""

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
        wall_h = 0.0145  # straight bore up to z=0.029; chamfer spans z=[0.029, 0.033]
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

        # 45-degree industrial lead-in chamfer around all 4 inner rim edges (widens mouth by +3.5mm/side)
        cham_mu = min(wall_mu, 0.16)
        cham_rgba = [0.46, 0.50, 0.56, 1.0]
        chamfers = [
            ("ch_px", [inner + 0.0032, 0.0, 0.0308], [0.0016, inner + 0.008, 0.0032], [0.0, 0.7854, 0.0]),
            ("ch_nx", [-(inner + 0.0032), 0.0, 0.0308], [0.0016, inner + 0.008, 0.0032], [0.0, -0.7854, 0.0]),
            ("ch_py", [0.0, inner + 0.0032, 0.0308], [inner + 0.008, 0.0016, 0.0032], [-0.7854, 0.0, 0.0]),
            ("ch_ny", [0.0, -(inner + 0.0032), 0.0308], [inner + 0.008, 0.0016, 0.0032], [0.7854, 0.0, 0.0]),
        ]
        for cn, cp, cs, ce in chamfers:
            sock.add_geom(
                name=cn,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=cs,
                pos=cp,
                euler=ce,
                mass=0.0,
                condim=4,
                friction=[cham_mu, 0.01, 0.0002],
                rgba=cham_rgba,
            )

        for cname, cpos in [
            ("collar_px", [inner + wall_t + 0.0025, 0, self.SOCKET_TOP_Z + 0.001]),
            ("collar_nx", [-(inner + wall_t + 0.0025), 0, self.SOCKET_TOP_Z + 0.001]),
        ]:
            sock.add_geom(
                name=cname,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.0030, inner + 2 * wall_t, 0.0012],
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
        tool.add_geom(
            name="plug_stem",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.011, 0.020],
            pos=[0, 0, -0.012],
            mass=0.14,
            contype=0,
            conaffinity=0,
            rgba=[0.72, 0.75, 0.78, 1.0],
        )
        tool.add_geom(
            name="plug_body",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.0145, 0.0145, 0.014],
            pos=[0, 0, -0.042],
            mass=0.14,
            condim=4,
            friction=[wall_mu, 0.01, 0.0002],
            rgba=[0.12, 0.45, 0.86, 1.0],
        )

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

    def expert_policy_no_oracle(self, w_tcp: np.ndarray, step_idx: int, coarse_vis_xy: np.ndarray) -> np.ndarray:
        """
        No-oracle recovery-aware expert:
          1. Approaches noisy coarse_vis_xy (NOT ground-truth sock_x/sock_y).
          2. During chamfer entry (depth < 6.5 mm), applies lateral force admittance
             from wrist (fx, fy) to slide down the 45-deg chamfer into the bore.
          3. Inside bore (depth >= 7 mm), detects internal jam vs normal bottoming identically to AutoConnectorEnv.
        """
        tcp = self.data.site("tcp_site").xpos
        fz_norm = float(self.TOOL_TARE_Z - w_tcp[2])
        depth = max(0.0, self.SOCKET_TOP_Z - float(tcp[2]))

        ex = float(coarse_vis_xy[0] - tcp[0])
        ey = float(coarse_vis_xy[1] - tcp[1])

        if self._jam_detected:
            dx, dy = 0.0, 0.0
            if depth > self._jam_hold_depth or fz_norm > 1.5:
                dz = 0.0012
            else:
                dz = 0.0
        elif step_idx < 14:
            # Align XY over coarse visual target before descending
            dx = float(np.clip(0.55 * ex, -0.0025, 0.0025))
            dy = float(np.clip(0.55 * ey, -0.0025, 0.0025))
            dz = 0.0 if (abs(ex) > 0.0015 or abs(ey) > 0.0015) else -0.0008
        elif depth < 0.0065:
            # Chamfer entry phase: compliant lateral admittance from wrist wrench
            fx, fy = float(w_tcp[0]), float(w_tcp[1])
            pull_vis = 0.20 if fz_norm < 1.2 else 0.0
            dx = float(np.clip(pull_vis * ex - 0.0012 * fx, -0.0024, 0.0024))
            dy = float(np.clip(pull_vis * ey - 0.0012 * fy, -0.0024, 0.0024))
            dz = -0.0010 if fz_norm < 4.5 else float(np.clip(0.00018 * (fz_norm - 4.8), -0.0006, 0.0008))
        elif depth < 0.0185 and fz_norm > 2.8:
            self._jam_detected = True
            self._jam_hold_depth = max(0.0070, depth - 0.0025)
            dx, dy = 0.0, 0.0
            dz = 0.0012
        elif depth >= 0.0235 and fz_norm >= 2.0:
            dx, dy = 0.0, 0.0
            dz = float(np.clip(0.00010 * (fz_norm - 3.5), -0.0008, 0.0008))
        else:
            fx, fy = float(w_tcp[0]), float(w_tcp[1])
            dx = float(np.clip(-0.0004 * fx, -0.0012, 0.0012))
            dy = float(np.clip(-0.0004 * fy, -0.0012, 0.0012))
            dz = -0.0010 if fz_norm < 2.8 else float(np.clip(0.00012 * (fz_norm - 3.5), -0.0006, 0.0008))

        return np.array([dx, dy, dz, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def classical_guided_policy(self, w_obs: np.ndarray, step_idx: int, coarse_vis_xy: np.ndarray) -> np.ndarray:
        """
        Non-strawman Classical Guided Admittance controller:
        Uses the exact same coarse_vis_xy alignment + lateral chamfer admittance + vertical constant-force insertion
        (no internal jam unload state machine).
        """
        tcp = self.data.site("tcp_site").xpos
        fz_norm = float(self.TOOL_TARE_Z - w_obs[2])
        ex = float(coarse_vis_xy[0] - tcp[0])
        ey = float(coarse_vis_xy[1] - tcp[1])
        fx, fy = float(w_obs[0]), float(w_obs[1])

        if step_idx < 14:
            dx = float(np.clip(0.55 * ex, -0.0025, 0.0025))
            dy = float(np.clip(0.55 * ey, -0.0025, 0.0025))
            dz = 0.0 if (abs(ex) > 0.0015 or abs(ey) > 0.0015) else -0.0008
        else:
            dx = float(np.clip(-0.0010 * fx, -0.0020, 0.0020))
            dy = float(np.clip(-0.0010 * fy, -0.0020, 0.0020))
            dz = float(np.clip(0.00018 * (fz_norm - 4.8), -0.0012, 0.0012))
        return np.array([dx, dy, dz, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def verify_chamfer_env(n_eps: int = 24):
    env = AutoConnectorChamferEnv(rgb_w=64, rgb_h=64)
    rng = np.random.default_rng(2027)
    rng_vis = np.random.default_rng(2431)
    succs, viols, peaks = [], [], []
    norm_s, jam_s = [], []
    try:
        for eid in range(n_eps):
            cfg = env.reset(rng)
            cxy = env.get_coarse_vis_xy(rng_vis)
            w_cur = env.last_w_avg.copy()
            fzs = []
            extra: Dict[str, List[float]] = {"depths": []}
            for t in range(env.MAX_STEPS):
                act = env.expert_policy_no_oracle(w_cur, t, cxy)
                w_cur, p_next, _ = env.step_action(act)
                fzs.append(float(w_cur[2]))
                extra["depths"].append(max(0.0, env.SOCKET_TOP_Z - float(p_next[2])))
            ev = env.evaluate_episode(fzs, extra)
            succs.append(float(ev["success"]))
            viols.append(float(ev["safety_viol"]))
            peaks.append(float(ev["peak_fn"]))
            if cfg["is_defect"] == 0:
                norm_s.append(float(ev["success"]))
            else:
                jam_s.append(float(ev["success"]))
            if not ev["success"]:
                print(f"  [FAIL ep{eid:02d}] {cfg['defect_type']} ev={ev}", flush=True)
    finally:
        env._close_renderer()
    print(
        f"[AutoConnectorChamferEnv Verify N={n_eps}] Succ={100*np.mean(succs):.1f}% "
        f"(Norm={100*np.mean(norm_s):.1f}%, Jam={100*np.mean(jam_s):.1f}%) | "
        f"Viol={100*np.mean(viols):.1f}% | MeanPeak={np.mean(peaks):.2f}N",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--n-eps", type=int, default=24)
    args = parser.parse_args()
    if args.verify:
        verify_chamfer_env(args.n_eps)
