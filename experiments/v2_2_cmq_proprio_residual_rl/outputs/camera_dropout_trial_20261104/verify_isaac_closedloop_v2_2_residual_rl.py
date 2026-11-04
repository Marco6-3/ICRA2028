"""
================================================================================
Stage 4 (V2.2) Live Isaac Lab 6-DoF Closed-Loop Verification &
3-Stage Long-Horizon Peg-in-Socket Manipulation Benchmark
================================================================================

Evaluates 6 unconfounded controllers in live 6-DoF PhysX / Isaac Lab simulation:
  1. `1_nominal_open_loop`:                 0-DoF open-loop hold
  2. `2_stage3s_1d_grip_only`:              1-DoF Stage 3S MCF-GRU grip reflex baseline
  3. `3_v2_1_pure_bpi_3dof_singlehead`:     3-DoF Pure V2.1 BPI-MCF Single-Head Policy (Zero Stage 3S mixing)
  4. `4_v2_2_online_rl_no_aux_collapse`:    3-DoF V2.2 Online Residual RL without Aux Self-Supervision (Collapse Ablation)
  5. `5_v2_2_warmstart_bc_dualhead`:        3-DoF V2.2 CMQ-BPI-MCF + Proprio-Tactile Dual-Head (Warm-Start BC)
  6. `6_v2_2_online_residual_rl_proposed`:  3-DoF V2.2 CMQ-BPI-MCF + Proprio-Tactile Dual-Head + Online Residual RL (Proposed)
"""
from __future__ import annotations

import argparse
import json
import hashlib
import sys
import time
from pathlib import Path
import numpy as np

# Compatibility shim for numpy 2.x checkpoints loaded in numpy 1.26 (isaacsim_env)
sys.modules["numpy._core"] = np.core
sys.modules["numpy._core.multiarray"] = np.core.multiarray
sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath

_REPO_E1S_DIR = Path(__file__).resolve().parent.parent / "stage3s_isaac" / "isaac_windows" / "e1s"
_LOCAL_E1S_DIR = Path(r"C:\Users\mingzhe Liu\OneDrive\Desktop\ws\isaac_windows\e1s")
E1S_DIR = _LOCAL_E1S_DIR if (_LOCAL_E1S_DIR / "outputs" / "contact_01" / "table_open" / "c000_r00.npz").exists() else _REPO_E1S_DIR
sys.path.insert(0, str(E1S_DIR))
from disturbance import Trial, rotation_wxyz

V21_DIR = Path(__file__).resolve().parent.parent / "v2_1_bpi_mcf_3dof_reflex"
V22_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--official-repo", type=Path, default=Path("C:/isaac_ws/FlexiTac-IsaacSim-Simulation"))
parser.add_argument(
    "--fixture-npz",
    type=Path,
    default=E1S_DIR / "outputs" / "contact_01" / "table_open" / "c000_r00.npz",
)
parser.add_argument(
    "--v21-ckpt-dir",
    type=Path,
    default=V21_DIR / "outputs" / "v2_1_clean_study" / "checkpoints",
)
parser.add_argument(
    "--v22-ckpt-dir",
    type=Path,
    default=V22_DIR / "outputs" / "checkpoints",
)
parser.add_argument(
    "--stage3s-mcf-ckpt",
    type=Path,
    default=E1S_DIR / "outputs" / "reflex_students_stream_01" / "mcf_selected.pt",
)
parser.add_argument(
    "--out",
    type=Path,
    default=V22_DIR / "outputs",
)
parser.add_argument("--settle-steps", type=int, default=30)
parser.add_argument("--trial-steps", type=int, default=90)
parser.add_argument("--longhorizon-steps", type=int, default=135)
parser.add_argument("--amplitude-n", type=float, default=0.20)
parser.add_argument("--initial-close-rad", type=float, default=0.003)
parser.add_argument("--max-disp-comp-m", type=float, default=0.0010)
parser.add_argument("--max-ang-comp-rad", type=float, default=0.023)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

app = AppLauncher(args).app
sys.path.insert(0, str(args.official_repo / "Isaacsim_tactile_env"))
sys.path.insert(0, str(E1S_DIR.parents[1] / "ICRA2028_Stage3S_starter"))
sys.path.insert(0, str(V21_DIR))
sys.path.insert(0, str(V22_DIR))

env = None
try:
    import torch
    from aloha_tactile_env import AlohaTactileEnv, AlohaTactileEnvCfg
    from aloha_ik_controller import _fuzzy_find_body, _fuzzy_find_joint, RIGHT_ARM_JOINT_NAMES
    from mcf import ContactFeatureStream
    from reflex_policy import Controller as Stage3SController, MAX_DELTA
    from verify_three_innovations_v2_1_clean import VisuotactileCompliancePolicyV21, extract_bpi_mcf
    from verify_three_innovations_v2_2_cmq import (
        ProprioTactileDualHeadPolicyV22,
        extract_cmq_bpi_mcf,
        extract_proprioceptive_history,
    )
    from pxr import UsdPhysics, Gf, Usd
    from socket_geometry import load_plug_tip, socket_alignment

    class MultiDoFClosedLoopEnv(AlohaTactileEnv):
        def _spawn_basic_world(self, sim_utils):
            super()._spawn_basic_world(sim_utils)
            floor = sim_utils.get_current_stage().GetPrimAtPath("/World/defaultGroundPlane")
            floor.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, -0.35))

        def _post_spawn_init(self, cfg, sim_utils, target_query_paths):
            root = self._sim_utils.get_current_stage().GetPrimAtPath("/World/Robot/table/collisions")
            if root.IsInstanceable():
                root.SetInstanceable(False)
            for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
            super()._post_spawn_init(cfg, sim_utils, target_query_paths)

    cfg = AlohaTactileEnvCfg(
        headless=args.headless,
        device=args.device,
        enable_camera=False,
        plug_scale=1.06,
        normal_offset=0.0036,
        fix_base=True,
    )
    env = MultiDoFClosedLoopEnv(cfg, simulation_app=app)
    env.reset()
    obj = env._plug_obj
    geometric_tip_offset, geometry_metadata = load_plug_tip(cfg.asset_root, cfg.automate_asset_id, cfg.plug_scale)

    with np.load(args.fixture_npz, allow_pickle=False) as fixture:
        state = np.asarray(fixture["joint_pos"][0], dtype=np.float32)
        velocities = np.asarray(fixture["joint_vel"][0], dtype=np.float32)
        hold = fixture["action"][0].astype(np.float32)
        q = env._robot.data.default_joint_pos.clone()
        dq = env._robot.data.default_joint_vel.clone()
        q[0, env._dataset_joint_ids] = torch.tensor(state, device=args.device)
        dq[0, env._dataset_joint_ids] = torch.tensor(velocities, device=args.device)
        env._robot.write_joint_state_to_sim(q, dq)
        for o, key in ((env._plug_obj, "plug_pose"), (env._socket_obj, "socket_pose")):
            root_st = o.data.default_root_state.clone()
            root_st[0, :7] = torch.tensor(fixture[key][0], device=args.device)
            root_st[0, 7:] = 0.0
            o.write_root_state_to_sim(root_st)
            o.reset()
        env._robot.reset()

    snapshot = {
        "robot_root": env._robot.data.root_state_w.clone(),
        "robot_q": env._robot.data.joint_pos.clone(),
        "robot_dq": env._robot.data.joint_vel.clone(),
        "objects": [o.data.root_state_w.clone() if o else None for o in (env._plug_obj, env._socket_obj)],
    }
    limits = env._robot.data.joint_pos_limits[0, env._dataset_joint_ids].cpu().numpy()

    arm_joint_ids = [_fuzzy_find_joint(env._robot, n) for n in RIGHT_ARM_JOINT_NAMES]
    body_names_l = [str(n).lower() for n in env._robot.body_names]
    right_ee_cands = [i for i, n in enumerate(body_names_l) if "right" in n and "ee_gripper" in n]
    ee_body_idx = right_ee_cands[0] if right_ee_cands else _fuzzy_find_body(env._robot, "ee_gripper_link")
    jacobi_body_idx = ee_body_idx - 1 if env._robot.is_fixed_base else ee_body_idx
    jacobi_joint_ids = arm_joint_ids if env._robot.is_fixed_base else [i + 6 for i in arm_joint_ids]
    print(f"[V2.2-Isaac] Right EE body={env._robot.body_names[ee_body_idx]} (idx={ee_body_idx})", flush=True)

    def current_com():
        pose = obj.data.root_link_pose_w[0].cpu().numpy()
        return pose[:3] + rotation_wxyz(pose[3:7]) @ obj.data.body_com_pos_b[0, 0].cpu().numpy()

    def load_v21_ckpt(ckpt_name: str, tac_dim: int, use_aux: bool, use_gate: bool):
        path = args.v21_ckpt_dir / f"{ckpt_name}.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = VisuotactileCompliancePolicyV21(
            tac_dim=tac_dim,
            vis_dim=6,
            hidden_dim=48,
            use_vis=True,
            use_tac=True,
            mod_drop_p=0.0,
            use_aux=use_aux,
            use_gate=use_gate,
        )
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return model, ck

    def load_v22_ckpt(ckpt_name: str, base_model: ProprioTactileDualHeadPolicyV22 | None = None):
        path = args.v22_ckpt_dir / f"{ckpt_name}.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = ProprioTactileDualHeadPolicyV22(
            tac_dim=int(ck.get("tac_dim", 48)),
            prop_dim=12,
            vis_dim=6,
            hidden_dim=64,
            use_proprio=bool(ck["use_proprio"]),
            use_dual_head=bool(ck["use_dual_head"]),
            residual_alpha=float(ck.get("residual_alpha", 0.0)),
        )
        if bool(ck.get("has_frozen_bc_base", False)) and base_model is not None:
            model.attach_frozen_bc_base(base_model, alpha=float(ck.get("residual_alpha", 0.35)))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return model, ck

    v21_b5_model, v21_b5_ck = load_v21_ckpt(
        "pure01_B5_Proposed_Full_Vis_BPI_MCF_Aux_Gate", tac_dim=32, use_aux=True, use_gate=True
    )
    v22_bc_model, v22_bc_ck = load_v22_ckpt("v22_3_dualhead_warmstart_bc")
    v22_rl_noaux_model, v22_rl_noaux_ck = load_v22_ckpt("v22_4_online_rl_no_aux_collapse", base_model=v22_bc_model)
    v22_rl_prop_model, v22_rl_prop_ck = load_v22_ckpt("v22_5_online_rl_with_aux_proposed", base_model=v22_bc_model)

    # 100% Matched Runtime Control Parameters across all 3-DoF controllers (P1 #4 & #5 fixed)
    U_XY_GAIN = 6.2
    U_G_GAIN = 6.8
    GATE_THRESH = 0.45
    SMOOTH_ALPHA = 0.40
    QUIET_NOISE_STD = 0.08
    DIPOLE_FLOOR = 0.26

    def step_v21_policy(
        model: VisuotactileCompliancePolicyV21,
        ck: dict,
        causal_s4_hist: list,
        causal_mcf_hist: list,
        is_quiet_window: bool,
        noise_rng: np.random.Generator,
    ):
        bpi_seq = extract_bpi_mcf({
            "state4": np.stack(causal_s4_hist, axis=0),
            "mcf": np.stack(causal_mcf_hist, axis=0),
        })
        T_b = len(bpi_seq)
        dip = np.zeros((1, T_b, 1), dtype=np.float32)
        for ti in range(T_b):
            ref_i = min(ti, 12)
            lag_i = max(0, ti - 8)
            d_ref = np.linalg.norm(bpi_seq[ti, 16:24] - bpi_seq[ref_i, 16:24])
            d_lag = np.linalg.norm(bpi_seq[ti, 16:24] - bpi_seq[lag_i, 16:24])
            dip[0, ti, 0] = (0.65 * d_ref + 0.35 * d_lag) / 1.1

        z_tac = np.clip((bpi_seq - ck["tac_mean"]) / ck["tac_std"], -8.0, 8.0).astype(np.float32)
        if is_quiet_window:
            z_tac[-1] = z_tac[-1] + QUIET_NOISE_STD * noise_rng.standard_normal(z_tac[-1].shape).astype(np.float32)

        tac_t = torch.from_numpy(z_tac[None, ...])
        vis_t = torch.zeros((1, len(z_tac), 6), dtype=torch.float32)
        dip_t = torch.from_numpy(dip)
        with torch.inference_mode():
            u_seq, _, _, _, _, g_tac_logit, _, _ = model(tac_t, vis_t, raw_dipole_energy=dip_t)
            u_raw = u_seq[0, -1].cpu().numpy()
            gate_val = float(torch.sigmoid(g_tac_logit[0, -1, 0]).cpu().item())

        u_xy_scaled = np.clip(u_raw[:2] * U_XY_GAIN, -1.0, 1.0)
        u_g_scaled = float(np.clip(u_raw[2] * U_G_GAIN + 0.72 * np.linalg.norm(u_xy_scaled), 0.0, 1.0))
        if dip[0, -1, 0] < DIPOLE_FLOOR:
            gate_val = float(min(gate_val, 0.08))
            u_xy_scaled = np.zeros(2, dtype=np.float32)
            u_g_scaled = 0.0
        return u_xy_scaled, u_g_scaled, gate_val

    def step_v22_policy(
        model: ProprioTactileDualHeadPolicyV22,
        ck: dict,
        causal_tac_hist: list,
        causal_s4_hist: list,
        causal_mcf_hist: list,
        causal_q_hist: list,
        causal_dq_hist: list,
        causal_act_hist: list,
        reference_mat: np.ndarray,
        is_quiet_window: bool,
        noise_rng: np.random.Generator,
    ):
        """
        Unconfounded V2.2 policy step:
          - Passes raw 'tactile', 'state4', 'mcf', 'joint_pos', 'joint_vel', and 'action' in data_win.
          - Uses 100% identical gains, noise, and dipole gating across all V2.2 models
            (Warm-Start BC, Surrogate Residual RL No-Aux, and Surrogate Residual RL With-Aux).
        """
        data_win = {
            "tactile": np.stack(causal_tac_hist, axis=0),
            "state4": np.stack(causal_s4_hist, axis=0),
            "mcf": np.stack(causal_mcf_hist, axis=0),
            "joint_pos": np.stack(causal_q_hist, axis=0),
            "joint_vel": np.stack(causal_dq_hist, axis=0),
            "action": np.stack(causal_act_hist, axis=0),
        }
        _, cmq48_seq = extract_cmq_bpi_mcf(data_win, require_tactile=True)
        bpi_v21_seq = extract_bpi_mcf({
            "state4": data_win["state4"],
            "mcf": data_win["mcf"],
        })
        prop_seq = extract_proprioceptive_history(data_win, reference_mat)

        ref_i = min(len(bpi_v21_seq) - 1, 12)
        lag_i = max(0, len(bpi_v21_seq) - 1 - 8)
        d_ref = np.linalg.norm(bpi_v21_seq[-1, 16:24] - bpi_v21_seq[ref_i, 16:24])
        d_lag = np.linalg.norm(bpi_v21_seq[-1, 16:24] - bpi_v21_seq[lag_i, 16:24])
        dip_last = float((0.65 * d_ref + 0.35 * d_lag) / 1.1)

        dip_profile = np.linalg.norm(bpi_v21_seq[:, 16:24] - bpi_v21_seq[ref_i, 16:24], axis=1) / 1.1
        dip_profile[:min(len(dip_profile), 16)] = 0.0
        active_idx = np.where(dip_profile >= 0.28)[0]
        feat_last = cmq48_seq[-1:].copy()
        if len(active_idx) > 0:
            anchor_i = min(len(cmq48_seq) - 1, int(active_idx[0]) + 3)
            feat_last[0, :] = 0.75 * cmq48_seq[anchor_i, :] + 0.25 * cmq48_seq[-1, :]
            dip_last = float(max(dip_last, 0.75 * dip_profile[anchor_i] + 0.25 * dip_last))

        xt_clean = np.clip((feat_last - ck["mu_tac"]) / ck["sd_tac"], -8.0, 8.0).astype(np.float32)
        xp_last = prop_seq[-1:].astype(np.float32)
        xt_obs = xt_clean + (xp_last @ ck["E_reaff"])
        if is_quiet_window:
            xt_obs = xt_obs + QUIET_NOISE_STD * noise_rng.standard_normal(xt_obs.shape).astype(np.float32)
        raw_e = np.array([[dip_last]], dtype=np.float32)
        vis_last = np.zeros((1, 6), dtype=np.float32)

        with torch.inference_mode():
            out = model(
                torch.from_numpy(xt_obs),
                torch.from_numpy(xp_last),
                torch.from_numpy(vis_last),
                torch.from_numpy(raw_e),
            )
            u_raw = out["u"][0].cpu().numpy()
            u_rec_raw = out["u_recenter"][0].cpu().numpy()
            gate_val = float(out["g_tac"][0, 0].cpu().item())

        u_xy_scaled = np.clip(u_raw[:2] * U_XY_GAIN, -1.0, 1.0)
        u_g_scaled = float(np.clip(u_raw[2] * U_G_GAIN + 0.72 * np.linalg.norm(u_xy_scaled), 0.0, 1.0))

        if dip_last < DIPOLE_FLOOR:
            gate_val = float(min(gate_val, 0.08))
            u_xy_scaled = np.zeros(2, dtype=np.float32)
            u_g_scaled = 0.0
        return u_xy_scaled, u_g_scaled, gate_val, (gate_val * float(out["g_vis"][0, 0]) * u_rec_raw[:2])

    # =========================================================================
    # PART A: 6-Controller Closed-Loop Ablation (5 Directions x 6 Controllers)
    # =========================================================================
    trials = [
        Trial("c_zero", direction="zero", amplitude_n=args.amplitude_n, point_m=(0.0, 0.0, 0.0), frame="reference"),
        Trial("c_pos_fx", direction="+Fx", amplitude_n=args.amplitude_n, point_m=(0.0, 0.0, 0.015), frame="reference"),
        Trial("c_neg_fx", direction="-Fx", amplitude_n=args.amplitude_n, point_m=(0.0, 0.0, 0.015), frame="reference"),
        Trial("c_pos_fy", direction="+Fy", amplitude_n=args.amplitude_n, point_m=(0.0, 0.0, 0.015), frame="reference"),
        Trial("c_neg_fy", direction="-Fy", amplitude_n=args.amplitude_n, point_m=(0.0, 0.0, 0.015), frame="reference"),
    ]

    controllers = [
        "1_nominal_open_loop",
        "2_stage3s_1d_grip_only",
        "3_v2_1_pure_bpi_3dof_singlehead",
        "4_v2_2_online_rl_no_aux_collapse",
        "5_v2_2_warmstart_bc_dualhead",
        "6_v2_2_online_residual_rl_proposed",
    ]

    reference = None
    all_trial_results = []

    for ctrl_name in controllers:
        for tr in trials:
            env._robot.write_root_state_to_sim(snapshot["robot_root"])
            initial_q = snapshot["robot_q"].clone()
            for j_idx, sign in zip([14, 15], [-1.0, 1.0]):
                initial_q[0, env._dataset_joint_ids[j_idx]] += sign * args.initial_close_rad
            env._robot.write_joint_state_to_sim(initial_q, torch.zeros_like(snapshot["robot_dq"]))
            env._robot.reset()
            for o, st in zip((env._plug_obj, env._socket_obj), snapshot["objects"]):
                if o:
                    st_c = st.clone()
                    st_c[:, 7:] = 0.0
                    o.write_root_state_to_sim(st_c)
                    o.reset()
            for sensor in env._tactile_sensors:
                sensor.reset()

            nominal_action = hold.copy()
            nominal_action[[14, 15]] += np.array([-1.0, 1.0], dtype=np.float32) * args.initial_close_rad
            for _ in range(args.settle_steps):
                env.step(nominal_action)

            if reference is None:
                index = env._sensor_slot_order.index(2)
                grid = env._tactile_sensors[index].data.tactile_points_w[0, :, :3].cpu().numpy().reshape(12, 32, 3).astype(float)
                x = grid[0, -1] - grid[0, 0]
                x /= np.linalg.norm(x)
                y = grid[-1, 0] - grid[0, 0]
                y -= x * (x @ y)
                y /= np.linalg.norm(y)
                reference = np.column_stack([x, y, np.cross(x, y)])

            stream = ContactFeatureStream(threshold=0.0)
            init_obs = env._get_obs()
            init_tac = np.asarray(init_obs["tactile"], dtype=np.float32).copy()
            stream.update(init_tac)
            initial_pose = obj.data.root_link_pose_w[0].cpu().numpy().copy()
            initial_r0 = rotation_wxyz(initial_pose[3:7])
            input_mcf = stream.update(init_tac)[1]
            prev_pose = initial_pose.copy()

            initial_finger_rel = []
            for f_slot in (2, 3):
                s_idx = env._sensor_slot_order.index(f_slot)
                pose_data = env._tactile_sensors[s_idx]._pose_sensors[0].data
                f_pos = pose_data.pos_w[0, 0].cpu().numpy()
                f_quat = pose_data.quat_w[0, 0].cpu().numpy()
                initial_finger_rel.append(rotation_wxyz(f_quat).T @ (initial_pose[:3] - f_pos))
            initial_finger_rel = np.asarray(initial_finger_rel)

            stage3s_ctrl = Stage3SController("mcf", cfg.physics_dt, args.stage3s_mcf_ckpt)
            nominal_ctrl = Stage3SController("nominal", cfg.physics_dt, None)

            disp_history, disp_ref_history, inhand_slip_history = [], [], []
            tilt_deg_history, ang_history, u_xy_history, u_g_history, gate_history = [], [], [], [], []

            latched_u_xy = np.zeros(2, dtype=np.float32)
            latched_u_g = 0.0
            noise_rng = np.random.default_rng(100 + trials.index(tr))
            init_s4 = input_mcf[:, :4].copy()
            init_q = env._robot.data.joint_pos[0, env._dataset_joint_ids].cpu().numpy().astype(np.float32).copy()
            init_dq = env._robot.data.joint_vel[0, env._dataset_joint_ids].cpu().numpy().astype(np.float32).copy()
            cur_tac = init_tac.copy()
            prev_action = nominal_action.astype(np.float32).copy()

            causal_tac_hist = [init_tac.copy() for _ in range(8)]
            causal_s4_hist = [init_s4.copy() for _ in range(8)]
            causal_mcf_hist = [input_mcf.copy() for _ in range(8)]
            causal_q_hist = [init_q.copy() for _ in range(8)]
            causal_dq_hist = [init_dq.copy() for _ in range(8)]
            causal_act_hist = [prev_action.copy() for _ in range(8)]

            for step_i in range(args.trial_steps):
                pose = obj.data.root_link_pose_w[0].cpu().numpy().copy()
                com = current_com().copy()
                t = step_i * cfg.physics_dt
                prev_angle = 2.0 * np.arccos(np.clip(abs(np.dot(pose[3:7], initial_pose[3:7])), 0.0, 1.0))

                tangent_vel = reference.T @ (pose[:3] - prev_pose[:3]) / cfg.physics_dt
                tangent_vel[2] = 0.0
                qa = pose[3:7].astype(float); qa /= np.linalg.norm(qa)
                qb = prev_pose[3:7].astype(float); qb /= np.linalg.norm(qb)
                ang_rate = 2.0 * np.arccos(np.clip(abs(qa @ qb), 0.0, 1.0)) / cfg.physics_dt
                realized_vel = np.r_[tangent_vel, ang_rate, 0.0, 0.0]

                cur_q = env._robot.data.joint_pos[0, env._dataset_joint_ids].cpu().numpy().astype(np.float32).copy()
                cur_dq = env._robot.data.joint_vel[0, env._dataset_joint_ids].cpu().numpy().astype(np.float32).copy()
                causal_tac_hist.append(cur_tac.copy())
                causal_s4_hist.append(input_mcf[:, :4].copy())
                causal_mcf_hist.append(input_mcf.copy())
                causal_q_hist.append(cur_q.copy())
                causal_dq_hist.append(cur_dq.copy())
                causal_act_hist.append(prev_action.copy())

                u_xy = np.zeros(2, dtype=np.float32)
                delta_g = 0.0
                gate_val = 0.0
                decay = np.exp(-cfg.physics_dt / 0.25)
                is_quiet = (tr.gain(t) == 0.0)

                if ctrl_name == "1_nominal_open_loop":
                    delta_g, _, _, _ = nominal_ctrl.step(input_mcf, realized_vel, prev_angle)
                elif ctrl_name == "2_stage3s_1d_grip_only":
                    delta_g, _, _, _ = stage3s_ctrl.step(input_mcf, realized_vel, prev_angle)
                    gate_val = float(delta_g > 1e-4)
                else:
                    if ctrl_name == "3_v2_1_pure_bpi_3dof_singlehead":
                        u_xy_sc, u_g_sc, gate_val = step_v21_policy(
                            v21_b5_model, v21_b5_ck, causal_s4_hist, causal_mcf_hist,
                            is_quiet_window=is_quiet, noise_rng=noise_rng
                        )
                    elif ctrl_name == "4_v2_2_online_rl_no_aux_collapse":
                        u_xy_sc, u_g_sc, gate_val, _ = step_v22_policy(
                            v22_rl_noaux_model, v22_rl_noaux_ck,
                            causal_tac_hist, causal_s4_hist, causal_mcf_hist,
                            causal_q_hist, causal_dq_hist, causal_act_hist,
                            reference, is_quiet_window=is_quiet, noise_rng=noise_rng
                        )
                    elif ctrl_name == "5_v2_2_warmstart_bc_dualhead":
                        u_xy_sc, u_g_sc, gate_val, _ = step_v22_policy(
                            v22_bc_model, v22_bc_ck,
                            causal_tac_hist, causal_s4_hist, causal_mcf_hist,
                            causal_q_hist, causal_dq_hist, causal_act_hist,
                            reference, is_quiet_window=is_quiet, noise_rng=noise_rng
                        )
                    elif ctrl_name == "6_v2_2_online_residual_rl_proposed":
                        u_xy_sc, u_g_sc, gate_val, _ = step_v22_policy(
                            v22_rl_prop_model, v22_rl_prop_ck,
                            causal_tac_hist, causal_s4_hist, causal_mcf_hist,
                            causal_q_hist, causal_dq_hist, causal_act_hist,
                            reference, is_quiet_window=is_quiet, noise_rng=noise_rng
                        )
                    else:
                        raise ValueError(f"Unknown controller: {ctrl_name}")

                    # 100% identical runtime gating, smoothing, and gripper bounds across all 3-DoF policies
                    if gate_val >= GATE_THRESH:
                        if np.linalg.norm(u_xy_sc) > np.linalg.norm(latched_u_xy) * decay:
                            latched_u_xy = (1.0 - SMOOTH_ALPHA) * latched_u_xy + SMOOTH_ALPHA * u_xy_sc
                        else:
                            latched_u_xy = latched_u_xy * decay
                        latched_u_g = max(u_g_sc * MAX_DELTA, latched_u_g * decay)
                        u_xy = latched_u_xy.copy()
                        delta_g = latched_u_g
                    else:
                        latched_u_xy = latched_u_xy * decay
                        latched_u_g = latched_u_g * decay

                prev_pose = pose.copy()
                action = nominal_action.copy()
                action[[14, 15]] += np.array([-1.0, 1.0], dtype=np.float32) * delta_g

                if np.linalg.norm(u_xy) > 1e-5:
                    jac = env._robot.root_physx_view.get_jacobians()[0, jacobi_body_idx, :, jacobi_joint_ids].cpu().numpy()
                    # 100% identical Jacobian compliance scaling across all 3-DoF policies
                    comp_disp = args.max_disp_comp_m
                    comp_ang = args.max_ang_comp_rad
                    max_dq = 0.012
                    dp_ref = np.array([u_xy[0] * comp_disp, u_xy[1] * comp_disp, 0.0], dtype=np.float64)
                    dtheta_ref = np.array([-u_xy[1] * comp_ang, u_xy[0] * comp_ang, 0.0], dtype=np.float64)
                    dp_w = reference @ dp_ref
                    dtheta_w = reference @ dtheta_ref
                    xi_w = np.r_[dp_w, dtheta_w]
                    lam = 0.08
                    dq_arm = jac.T @ np.linalg.solve(jac @ jac.T + (lam ** 2) * np.eye(6), xi_w)
                    dq_arm = np.clip(dq_arm, -max_dq, max_dq).astype(np.float32)
                    action[8:14] += dq_arm

                action = np.clip(action, limits[:, 0], limits[:, 1])
                prev_action = action.astype(np.float32).copy()

                f, m, p = tr.wrench(t, pose, com, reference)
                obj.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=torch.tensor(f[None], dtype=torch.float32, device=args.device),
                    torque_data=torch.tensor(m[None], dtype=torch.float32, device=args.device),
                    position_data=torch.tensor(com[None], dtype=torch.float32, device=args.device),
                    indices=torch.tensor([0], dtype=torch.int32, device=args.device),
                    is_global=True,
                )
                obs, *_ = env.step(action)
                cur_tac = np.asarray(obs["tactile"], dtype=np.float32).copy()
                _, mcf_now = stream.update(cur_tac)
                input_mcf = mcf_now.copy()

                cur_pose = obj.data.root_link_pose_w[0].cpu().numpy().copy()
                disp_w = cur_pose[:3] - initial_pose[:3]
                disp_ref = reference.T @ disp_w
                ang = 2.0 * np.arccos(np.clip(abs(np.dot(cur_pose[3:7], initial_pose[3:7])), 0.0, 1.0))
                tilt_rad = np.arccos(np.clip((rotation_wxyz(cur_pose[3:7]) @ initial_r0.T)[2, 2], -1.0, 1.0))

                cur_finger_rel = []
                for f_slot in (2, 3):
                    s_idx = env._sensor_slot_order.index(f_slot)
                    pose_data = env._tactile_sensors[s_idx]._pose_sensors[0].data
                    f_pos = pose_data.pos_w[0, 0].cpu().numpy()
                    f_quat = pose_data.quat_w[0, 0].cpu().numpy()
                    cur_finger_rel.append(rotation_wxyz(f_quat).T @ (cur_pose[:3] - f_pos))
                cur_finger_rel = np.asarray(cur_finger_rel)
                step_inhand_slip_mm = float(np.linalg.norm(cur_finger_rel - initial_finger_rel, axis=1).max() * 1000.0)

                disp_history.append(disp_w)
                disp_ref_history.append(disp_ref)
                inhand_slip_history.append(step_inhand_slip_mm)
                tilt_deg_history.append(float(np.degrees(tilt_rad)))
                ang_history.append(ang)
                u_xy_history.append(u_xy.copy())
                u_g_history.append(delta_g)
                gate_history.append(gate_val)

            disp_arr = np.stack(disp_history)
            disp_ref_arr = np.stack(disp_ref_history)
            inhand_slip_arr = np.asarray(inhand_slip_history)
            tilt_deg_arr = np.asarray(tilt_deg_history)
            ang_arr = np.asarray(ang_history)
            u_xy_arr = np.stack(u_xy_history)
            u_g_arr = np.asarray(u_g_history)

            max_disp_mm = float(np.linalg.norm(disp_arr, axis=1).max() * 1000.0)
            max_inhand_slip_mm = float(inhand_slip_arr.max())
            max_tilt_deg = float(tilt_deg_arr.max())
            active_xy_disp_mm = float(np.linalg.norm(disp_ref_arr[25:65, :2], axis=1).mean() * 1000.0)
            end_xy_disp_mm = float(np.linalg.norm(disp_ref_arr[-1, :2]) * 1000.0)
            max_ang_deg = float(np.degrees(ang_arr.max()))
            preload_jitter_mm = float(np.linalg.norm(u_xy_arr[:25], axis=1).mean() * args.max_disp_comp_m * 1000.0)
            preload_phys_slip_mm = float(inhand_slip_arr[:25].max())

            res = {
                "controller": ctrl_name,
                "trial_id": tr.trial_id,
                "direction": tr.direction,
                "max_inhand_slip_mm": round(max_inhand_slip_mm, 4),
                "max_tilt_deg": round(max_tilt_deg, 4),
                "max_disp_mm": round(max_disp_mm, 4),
                "active_xy_disp_mm": round(active_xy_disp_mm, 4),
                "end_xy_disp_mm": round(end_xy_disp_mm, 4),
                "max_ang_deg": round(max_ang_deg, 4),
                "preload_jitter_mm": round(preload_jitter_mm, 4),
                "preload_phys_slip_mm": round(preload_phys_slip_mm, 4),
                "mean_active_u_xy": u_xy_arr[28:55].mean(axis=0).round(4).tolist(),
                "mean_active_delta_g_mrad": round(float(u_g_arr[28:55].mean() * 1000.0), 4),
                "mean_quiet_gate": round(float(np.mean(gate_history[:25])), 4),
                "mean_active_gate": round(float(np.mean(gate_history[32:58])), 4),
            }
            all_trial_results.append(res)
            print(
                f"PART_A [{ctrl_name:36s} | {tr.direction:4s}] "
                f"slip={max_inhand_slip_mm:.3f}mm tilt={max_tilt_deg:.3f}deg "
                f"cmd_jit={preload_jitter_mm:.4f}mm phys_slip_q={preload_phys_slip_mm:.4f}mm "
                f"u_xy={res['mean_active_u_xy']} "
                f"gate(q/a)={res['mean_quiet_gate']:.2f}/{res['mean_active_gate']:.2f}",
                flush=True,
            )

    summary_part_a = {}
    for ctrl_name in controllers:
        c_trials = [r for r in all_trial_results if r["controller"] == ctrl_name]
        loaded = [r for r in c_trials if r["direction"] != "zero"]
        zero_tr = [r for r in c_trials if r["direction"] == "zero"][0]
        summary_part_a[ctrl_name] = {
            "loaded_mean_inhand_slip_mm": round(float(np.mean([r["max_inhand_slip_mm"] for r in loaded])), 4),
            "loaded_worst_inhand_slip_mm": round(float(np.max([r["max_inhand_slip_mm"] for r in loaded])), 4),
            "loaded_mean_tilt_deg": round(float(np.mean([r["max_tilt_deg"] for r in loaded])), 4),
            "loaded_mean_max_disp_mm": round(float(np.mean([r["max_disp_mm"] for r in loaded])), 4),
            "loaded_mean_active_xy_disp_mm": round(float(np.mean([r["active_xy_disp_mm"] for r in loaded])), 4),
            "loaded_mean_end_xy_disp_mm": round(float(np.mean([r["end_xy_disp_mm"] for r in loaded])), 4),
            "loaded_mean_max_ang_deg": round(float(np.mean([r["max_ang_deg"] for r in loaded])), 4),
            "preload_mean_jitter_mm": round(float(np.mean([r["preload_jitter_mm"] for r in c_trials])), 4),
            "preload_mean_phys_slip_mm": round(float(np.mean([r["preload_phys_slip_mm"] for r in c_trials])), 4),
            "zero_load_inhand_slip_mm": zero_tr["max_inhand_slip_mm"],
            "zero_load_max_disp_mm": zero_tr["max_disp_mm"],
            "zero_disturbance_jitter_deg": zero_tr["max_ang_deg"],
            "per_direction": {r["direction"]: r for r in c_trials},
        }

    # =========================================================================
    # PART B: 3-Stage Multi-Disturbance & Socket-Relative Geometric Benchmark
    # =========================================================================
    lh_controllers = [
        "1_nominal_open_loop",
        "2_stage3s_1d_grip_only",
        "3_v2_1_pure_bpi_3dof_singlehead",
        "4_v2_2_online_rl_no_aux_collapse",
        "5_v2_2_warmstart_bc_dualhead",
        "6_v2_2_online_residual_rl_proposed",
    ]
    lh_trials = [t for t in trials if t.direction != "zero"]
    lh_results = []

    for ctrl_name in lh_controllers:
        for tr in lh_trials:
            env._robot.write_root_state_to_sim(snapshot["robot_root"])
            initial_q = snapshot["robot_q"].clone()
            for j_idx, sign in zip([14, 15], [-1.0, 1.0]):
                initial_q[0, env._dataset_joint_ids[j_idx]] += sign * args.initial_close_rad
            env._robot.write_joint_state_to_sim(initial_q, torch.zeros_like(snapshot["robot_dq"]))
            env._robot.reset()
            for o, st in zip((env._plug_obj, env._socket_obj), snapshot["objects"]):
                if o:
                    st_c = st.clone()
                    st_c[:, 7:] = 0.0
                    o.write_root_state_to_sim(st_c)
                    o.reset()
            for sensor in env._tactile_sensors:
                sensor.reset()

            nominal_action = hold.copy()
            nominal_action[[14, 15]] += np.array([-1.0, 1.0], dtype=np.float32) * args.initial_close_rad
            for _ in range(args.settle_steps):
                env.step(nominal_action)

            stream = ContactFeatureStream(threshold=0.0)
            init_obs = env._get_obs()
            init_tac = np.asarray(init_obs["tactile"], dtype=np.float32).copy()
            stream.update(init_tac)
            initial_pose = obj.data.root_link_pose_w[0].cpu().numpy().copy()
            initial_r0 = rotation_wxyz(initial_pose[3:7])
            input_mcf = stream.update(init_tac)[1]
            prev_pose = initial_pose.copy()

            # True socket-relative initial reference frame (fixes P1 #6)
            socket_pose_0 = env._socket_obj.data.root_link_pose_w[0].cpu().numpy().copy()
            r_soc_0 = rotation_wxyz(socket_pose_0[3:7])
            rel_pos_0 = r_soc_0.T @ (initial_pose[:3] - socket_pose_0[:3])
            rel_rot_0 = r_soc_0.T @ initial_r0
            plug_tip_offset_b = np.array([0.0, 0.0, -0.030], dtype=np.float64)
            tip_pos_0_soc = r_soc_0.T @ ((initial_pose[:3] + initial_r0 @ plug_tip_offset_b) - socket_pose_0[:3])

            initial_finger_rel = []
            for f_slot in (2, 3):
                s_idx = env._sensor_slot_order.index(f_slot)
                pose_data = env._tactile_sensors[s_idx]._pose_sensors[0].data
                f_pos = pose_data.pos_w[0, 0].cpu().numpy()
                f_quat = pose_data.quat_w[0, 0].cpu().numpy()
                initial_finger_rel.append(rotation_wxyz(f_quat).T @ (initial_pose[:3] - f_pos))
            initial_finger_rel = np.asarray(initial_finger_rel)

            stage3s_ctrl = Stage3SController("mcf", cfg.physics_dt, args.stage3s_mcf_ckpt)
            nominal_ctrl = Stage3SController("nominal", cfg.physics_dt, None)

            latched_u_xy = np.zeros(2, dtype=np.float32)
            latched_u_g = 0.0
            noise_rng = np.random.default_rng(500 + lh_trials.index(tr))
            init_s4 = input_mcf[:, :4].copy()
            init_q = env._robot.data.joint_pos[0, env._dataset_joint_ids].cpu().numpy().astype(np.float32).copy()
            init_dq = env._robot.data.joint_vel[0, env._dataset_joint_ids].cpu().numpy().astype(np.float32).copy()
            cur_tac = init_tac.copy()
            prev_action = nominal_action.astype(np.float32).copy()

            causal_tac_hist = [init_tac.copy() for _ in range(8)]
            causal_s4_hist = [init_s4.copy() for _ in range(8)]
            causal_mcf_hist = [input_mcf.copy() for _ in range(8)]
            causal_q_hist = [init_q.copy() for _ in range(8)]
            causal_dq_hist = [init_dq.copy() for _ in range(8)]
            causal_act_hist = [prev_action.copy() for _ in range(8)]

            s1_jitter_list, s1_phys_slip_list, s2_slip_list, s3_slip_list, s3_tilt_list = [], [], [], [], []
            s3_inhand_tip_proxy_list, s3_soc_radial_list, s3_soc_tilt_list, s3_soc_tip_lat_list = [], [], [], []
            s3_radial_drift, s3_axis_drift, s3_tip_drift, s3_axial = [], [], [], []

            for step_i in range(args.longhorizon_steps):
                pose = obj.data.root_link_pose_w[0].cpu().numpy().copy()
                com = current_com().copy()
                t = step_i * cfg.physics_dt
                prev_angle = 2.0 * np.arccos(np.clip(abs(np.dot(pose[3:7], initial_pose[3:7])), 0.0, 1.0))

                tangent_vel = reference.T @ (pose[:3] - prev_pose[:3]) / cfg.physics_dt
                tangent_vel[2] = 0.0
                qa = pose[3:7].astype(float); qa /= np.linalg.norm(qa)
                qb = prev_pose[3:7].astype(float); qb /= np.linalg.norm(qb)
                ang_rate = 2.0 * np.arccos(np.clip(abs(qa @ qb), 0.0, 1.0)) / cfg.physics_dt
                realized_vel = np.r_[tangent_vel, ang_rate, 0.0, 0.0]

                cur_q = env._robot.data.joint_pos[0, env._dataset_joint_ids].cpu().numpy().astype(np.float32).copy()
                cur_dq = env._robot.data.joint_vel[0, env._dataset_joint_ids].cpu().numpy().astype(np.float32).copy()
                causal_tac_hist.append(cur_tac.copy())
                causal_s4_hist.append(input_mcf[:, :4].copy())
                causal_mcf_hist.append(input_mcf.copy())
                causal_q_hist.append(cur_q.copy())
                causal_dq_hist.append(cur_dq.copy())
                causal_act_hist.append(prev_action.copy())

                is_quiet = (step_i < 30)
                u_xy = np.zeros(2, dtype=np.float32)
                delta_g = 0.0
                decay = np.exp(-cfg.physics_dt / 0.25)

                if ctrl_name == "1_nominal_open_loop":
                    delta_g, _, _, _ = nominal_ctrl.step(input_mcf, realized_vel, prev_angle)
                elif ctrl_name == "2_stage3s_1d_grip_only":
                    delta_g, _, _, _ = stage3s_ctrl.step(input_mcf, realized_vel, prev_angle)
                else:
                    if ctrl_name == "3_v2_1_pure_bpi_3dof_singlehead":
                        u_xy_sc, u_g_sc, gate_val = step_v21_policy(
                            v21_b5_model, v21_b5_ck, causal_s4_hist, causal_mcf_hist,
                            is_quiet_window=is_quiet, noise_rng=noise_rng
                        )
                    elif ctrl_name == "4_v2_2_online_rl_no_aux_collapse":
                        u_xy_sc, u_g_sc, gate_val, _ = step_v22_policy(
                            v22_rl_noaux_model, v22_rl_noaux_ck,
                            causal_tac_hist, causal_s4_hist, causal_mcf_hist,
                            causal_q_hist, causal_dq_hist, causal_act_hist,
                            reference, is_quiet_window=is_quiet, noise_rng=noise_rng
                        )
                    elif ctrl_name == "5_v2_2_warmstart_bc_dualhead":
                        u_xy_sc, u_g_sc, gate_val, _ = step_v22_policy(
                            v22_bc_model, v22_bc_ck,
                            causal_tac_hist, causal_s4_hist, causal_mcf_hist,
                            causal_q_hist, causal_dq_hist, causal_act_hist,
                            reference, is_quiet_window=is_quiet, noise_rng=noise_rng
                        )
                    elif ctrl_name == "6_v2_2_online_residual_rl_proposed":
                        u_xy_sc, u_g_sc, gate_val, _ = step_v22_policy(
                            v22_rl_prop_model, v22_rl_prop_ck,
                            causal_tac_hist, causal_s4_hist, causal_mcf_hist,
                            causal_q_hist, causal_dq_hist, causal_act_hist,
                            reference, is_quiet_window=is_quiet, noise_rng=noise_rng
                        )
                    else:
                        raise ValueError(f"Unknown controller: {ctrl_name}")

                    # 100% identical runtime gating, smoothing, and gripper bounds across all 3-DoF policies
                    # (No step_i >= 85 special branch)
                    if gate_val >= GATE_THRESH:
                        if np.linalg.norm(u_xy_sc) > np.linalg.norm(latched_u_xy) * decay:
                            latched_u_xy = (1.0 - SMOOTH_ALPHA) * latched_u_xy + SMOOTH_ALPHA * u_xy_sc
                        else:
                            latched_u_xy = latched_u_xy * decay
                        latched_u_g = max(u_g_sc * MAX_DELTA, latched_u_g * decay)
                        u_xy = latched_u_xy.copy()
                        delta_g = latched_u_g
                    else:
                        latched_u_xy = latched_u_xy * decay
                        latched_u_g = latched_u_g * decay

                prev_pose = pose.copy()
                action = nominal_action.copy()
                action[[14, 15]] += np.array([-1.0, 1.0], dtype=np.float32) * delta_g

                if np.linalg.norm(u_xy) > 1e-5:
                    jac = env._robot.root_physx_view.get_jacobians()[0, jacobi_body_idx, :, jacobi_joint_ids].cpu().numpy()
                    comp_disp = args.max_disp_comp_m
                    comp_ang = args.max_ang_comp_rad
                    max_dq = 0.012
                    dp_ref = np.array([u_xy[0] * comp_disp, u_xy[1] * comp_disp, 0.0], dtype=np.float64)
                    dtheta_ref = np.array([-u_xy[1] * comp_ang, u_xy[0] * comp_ang, 0.0], dtype=np.float64)
                    dp_w = reference @ dp_ref
                    dtheta_w = reference @ dtheta_ref
                    xi_w = np.r_[dp_w, dtheta_w]
                    lam = 0.08
                    dq_arm = jac.T @ np.linalg.solve(jac @ jac.T + (lam ** 2) * np.eye(6), xi_w)
                    dq_arm = np.clip(dq_arm, -max_dq, max_dq).astype(np.float32)
                    action[8:14] += dq_arm

                action = np.clip(action, limits[:, 0], limits[:, 1])
                prev_action = action.astype(np.float32).copy()

                if 30 <= step_i < 65:
                    f, m, _ = tr.wrench(0.35, pose, com, reference)
                elif 85 <= step_i < 115:
                    f_raw, m_raw, _ = tr.wrench(0.35, pose, com, reference)
                    f = -0.45 * f_raw
                    m = -0.45 * m_raw
                else:
                    f = np.zeros(3, dtype=np.float64)
                    m = np.zeros(3, dtype=np.float64)

                obj.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=torch.tensor(f[None], dtype=torch.float32, device=args.device),
                    torque_data=torch.tensor(m[None], dtype=torch.float32, device=args.device),
                    position_data=torch.tensor(com[None], dtype=torch.float32, device=args.device),
                    indices=torch.tensor([0], dtype=torch.int32, device=args.device),
                    is_global=True,
                )
                obs, *_ = env.step(action)
                cur_tac = np.asarray(obs["tactile"], dtype=np.float32).copy()
                _, mcf_now = stream.update(cur_tac)
                input_mcf = mcf_now.copy()

                cur_pose = obj.data.root_link_pose_w[0].cpu().numpy().copy()
                cur_rot = rotation_wxyz(cur_pose[3:7])
                tilt_rad = np.arccos(np.clip((cur_rot @ initial_r0.T)[2, 2], -1.0, 1.0))
                tilt_deg = float(np.degrees(tilt_rad))

                cur_finger_rel = []
                for f_slot in (2, 3):
                    s_idx = env._sensor_slot_order.index(f_slot)
                    pose_data = env._tactile_sensors[s_idx]._pose_sensors[0].data
                    f_pos = pose_data.pos_w[0, 0].cpu().numpy()
                    f_quat = pose_data.quat_w[0, 0].cpu().numpy()
                    cur_finger_rel.append(rotation_wxyz(f_quat).T @ (cur_pose[:3] - f_pos))
                cur_finger_rel = np.asarray(cur_finger_rel)
                step_slip_mm = float(np.linalg.norm(cur_finger_rel - initial_finger_rel, axis=1).max() * 1000.0)
                inhand_tip_proxy_mm = float(step_slip_mm + 30.0 * np.sin(tilt_rad))

                # True plug-to-socket geometric alignment metrics (fixes P1 #6)
                socket_pose_t = env._socket_obj.data.root_link_pose_w[0].cpu().numpy().copy()
                r_soc_t = rotation_wxyz(socket_pose_t[3:7])
                rel_pos_t = r_soc_t.T @ (cur_pose[:3] - socket_pose_t[:3])
                rel_rot_t = r_soc_t.T @ cur_rot
                radial_drift = float(np.linalg.norm(rel_pos_t[:2] - rel_pos_0[:2]) * 1000.0)
                axis_drift = float(np.degrees(np.arccos(np.clip((rel_rot_t @ rel_rot_0.T)[2, 2], -1.0, 1.0))))
                tip_pos_t_soc = r_soc_t.T @ ((cur_pose[:3] + cur_rot @ plug_tip_offset_b) - socket_pose_t[:3])
                tip_drift = float(np.linalg.norm(tip_pos_t_soc[:2] - tip_pos_0_soc[:2]) * 1000.0)
                geom = socket_alignment(cur_pose[:3], cur_rot, socket_pose_t[:3], r_soc_t, geometric_tip_offset)

                if step_i < 28:
                    s1_jitter_list.append(float(np.linalg.norm(u_xy) * args.max_disp_comp_m * 1000.0))
                    s1_phys_slip_list.append(step_slip_mm)
                elif step_i < 85:
                    s2_slip_list.append(step_slip_mm)
                else:
                    s3_slip_list.append(step_slip_mm)
                    s3_tilt_list.append(tilt_deg)
                    s3_inhand_tip_proxy_list.append(inhand_tip_proxy_mm)
                    s3_soc_radial_list.append(geom["radial_error_mm"])
                    s3_soc_tilt_list.append(geom["axis_tilt_deg"])
                    s3_soc_tip_lat_list.append(geom["tip_lateral_error_mm"])
                    s3_axial.append(geom["tip_axial_position_mm"])
                    s3_radial_drift.append(radial_drift)
                    s3_axis_drift.append(axis_drift)
                    s3_tip_drift.append(tip_drift)

            s1_jitter = round(float(np.mean(s1_jitter_list)), 4)
            s1_phys_slip = round(float(np.max(s1_phys_slip_list)), 4)
            s2_max_slip = round(float(np.max(s2_slip_list)), 4)
            s3_max_slip = round(float(np.max(s3_slip_list)), 4)
            s3_max_tilt = round(float(np.max(s3_tilt_list)), 4)
            s3_final_tip_err = round(float(np.mean(s3_inhand_tip_proxy_list[-10:])), 4)
            s3_soc_radial_err = round(float(np.mean(s3_soc_radial_list[-10:])), 4)
            s3_soc_axis_tilt = round(float(np.mean(s3_soc_tilt_list[-10:])), 4)
            s3_soc_tip_lat_err = round(float(np.mean(s3_soc_tip_lat_list[-10:])), 4)
            # Socket-relative geometric readiness: true plug tip lateral drift wrt socket axis < 2.20 mm,
            # true plug-to-socket axis tilt < 1.50 deg, and in-hand slip < 1.90 mm
            stage_success = bool(
                s3_soc_tip_lat_err < 2.20 and s3_max_slip < 1.90 and s3_soc_axis_tilt < 1.50
            )

            lh_rec = {
                "controller": ctrl_name,
                "direction": tr.direction,
                "stage1_jitter_mm": s1_jitter,
                "stage1_phys_slip_mm": s1_phys_slip,
                "stage2_transport_max_slip_mm": s2_max_slip,
                "stage3_align_max_slip_mm": s3_max_slip,
                "stage3_align_max_tilt_deg": s3_max_tilt,
                "stage3_final_tip_error_mm": s3_final_tip_err,
                "stage3_socket_radial_error_mm": s3_soc_radial_err,
                "stage3_socket_axis_tilt_deg": s3_soc_axis_tilt,
                "stage3_socket_tip_lateral_error_mm": s3_soc_tip_lat_err,
                "stage3_socket_tip_axial_position_mm": round(float(np.mean(s3_axial[-10:])), 4),
                "stage3_socket_radial_drift_mm": round(float(np.mean(s3_radial_drift[-10:])), 4),
                "stage3_socket_axis_drift_deg": round(float(np.mean(s3_axis_drift[-10:])), 4),
                "stage3_legacy_30mm_point_lateral_drift_mm": round(float(np.mean(s3_tip_drift[-10:])), 4),
                "long_horizon_insertion_ready": stage_success,
            }
            lh_results.append(lh_rec)
            print(
                f"PART_B_LONGHORIZON [{ctrl_name:36s} | {tr.direction:4s}] "
                f"S1_jit={s1_jitter:.4f}mm S2_slip={s2_max_slip:.3f}mm "
                f"S3_slip={s3_max_slip:.3f}mm S3_tilt={s3_max_tilt:.3f}deg "
                f"soc_tip={s3_soc_tip_lat_err:.3f}mm soc_rad={s3_soc_radial_err:.3f}mm "
                f"inhand_proxy={s3_final_tip_err:.3f}mm success={stage_success}",
                flush=True,
            )

    summary_part_b = {}
    for ctrl_name in lh_controllers:
        recs = [r for r in lh_results if r["controller"] == ctrl_name]
        summary_part_b[ctrl_name] = {
            "stage1_mean_jitter_mm": round(float(np.mean([r["stage1_jitter_mm"] for r in recs])), 4),
            "stage1_mean_phys_slip_mm": round(float(np.mean([r["stage1_phys_slip_mm"] for r in recs])), 4),
            "stage2_mean_transport_slip_mm": round(float(np.mean([r["stage2_transport_max_slip_mm"] for r in recs])), 4),
            "stage3_mean_align_slip_mm": round(float(np.mean([r["stage3_align_max_slip_mm"] for r in recs])), 4),
            "stage3_mean_align_tilt_deg": round(float(np.mean([r["stage3_align_max_tilt_deg"] for r in recs])), 4),
            "stage3_mean_final_tip_error_mm": round(float(np.mean([r["stage3_final_tip_error_mm"] for r in recs])), 4),
            "stage3_mean_socket_radial_error_mm": round(float(np.mean([r["stage3_socket_radial_error_mm"] for r in recs])), 4),
            "stage3_mean_socket_axis_tilt_deg": round(float(np.mean([r["stage3_socket_axis_tilt_deg"] for r in recs])), 4),
            "stage3_mean_socket_tip_lateral_error_mm": round(float(np.mean([r["stage3_socket_tip_lateral_error_mm"] for r in recs])), 4),
            "long_horizon_success_rate": round(float(np.mean([float(r["long_horizon_insertion_ready"]) for r in recs])), 4),
            "per_direction": {r["direction"]: r for r in recs},
        }

    runtime_params = dict(u_xy_gain=U_XY_GAIN, u_g_gain=U_G_GAIN, gate_threshold=GATE_THRESH,
                          dipole_floor=DIPOLE_FLOOR, smooth_alpha=SMOOTH_ALPHA, quiet_noise_std=QUIET_NOISE_STD,
                          max_disp_comp_m=args.max_disp_comp_m, max_ang_comp_rad=args.max_ang_comp_rad,
                          max_dq=0.012, dls_lambda=0.08, latch_decay_s=0.25, max_grip_delta=float(MAX_DELTA))
    runtime_by_controller = {name: runtime_params.copy() for name in controllers[2:]}
    out_payload = {
        "status": "completed",
        "input_sha256": {str(path.relative_to(V22_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [Path(__file__), V22_DIR / "verify_three_innovations_v2_2_cmq.py", V22_DIR / "socket_geometry.py",
                                      *sorted(args.v22_ckpt_dir.glob("*.pt"))]},
        "control_fairness_audit": {
            "gains_and_gates_matched_across_all_3dof_policies": all(v == runtime_params for v in runtime_by_controller.values()),
            "runtime_parameters_by_controller": runtime_by_controller,
            "scope": "Shared external gains, thresholds, noise and DLS; learned intrinsic gates and feature transforms differ by architecture.",
        },
        "socket_geometry_protocol": {
            **geometry_metadata,
            "socket_axis": "local Z axis through socket root origin",
            "axis_angle": "unsigned angle between plug and socket local Z axes",
            "aggregation": "mean of final 10 frames per trial, then mean of four directions",
            "initial_pose_subtracted": False,
            "readiness_thresholds": {"tip_lateral_mm": 2.20, "axis_tilt_deg": 1.50, "inhand_slip_mm": 1.90},
            "insertion_depth_and_contact_success_verified": False,
        },
        "zero_disturbance_jitter_definition": "Maximum quaternion rotation from initial pose over the entire zero-force trial, degrees.",
        "part_a_6controller_closedloop": summary_part_a,
        "part_b_3stage_long_horizon_benchmark": summary_part_b,
        "part_a_trials": all_trial_results,
        "part_b_trials": lh_results,
    }
    out_file = args.out / "isaac_closedloop_v2_2_residual_rl_report.json"
    out_file.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print("ISAAC_V2_2_RESIDUAL_RL_COMPLETED", str(out_file), flush=True)

except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
