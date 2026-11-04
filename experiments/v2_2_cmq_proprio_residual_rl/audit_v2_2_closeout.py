"""Recheck deployed step, checkpoint invariants and acceptance metrics without starting Isaac."""
from __future__ import annotations
import ast
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import verify_three_innovations_v2_2_cmq as v
from verify_three_innovations_v2_1_clean import extract_bpi_mcf
from socket_geometry import socket_alignment, quaternion_angle_deg

ROOT = Path(__file__).resolve().parent


def load_runtime_step():
    # Compile the actual function and its literal control constants, not a test copy.
    path = ROOT / "verify_isaac_closedloop_v2_2_residual_rl.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "step_v22_policy")
    ns = dict(np=np, torch=torch, ProprioTactileDualHeadPolicyV22=v.ProprioTactileDualHeadPolicyV22,
              extract_bpi_mcf=extract_bpi_mcf, extract_cmq_bpi_mcf=v.extract_cmq_bpi_mcf,
              extract_proprioceptive_history=v.extract_proprioceptive_history)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"U_XY_GAIN", "U_G_GAIN", "QUIET_NOISE_STD", "DIPOLE_FLOOR"}:
                ns[node.targets[0].id] = ast.literal_eval(node.value)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["step_v22_policy"]


def main():
    torch.set_num_threads(1)
    step = load_runtime_step()
    ck = torch.load(ROOT / "outputs/checkpoints/v22_5_online_rl_with_aux_proposed.pt", weights_only=False)
    model = v.ProprioTactileDualHeadPolicyV22(tac_dim=48)
    model.attach_frozen_bc_base(v.ProprioTactileDualHeadPolicyV22(tac_dim=48), .35)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    base_ck = torch.load(ROOT / "outputs/checkpoints/v22_3_dualhead_warmstart_bc.pt", weights_only=False)
    base_equal = all(torch.equal(p, base_ck["state_dict"][k]) for k, p in model.frozen_bc_base.state_dict().items())
    pure = v.BASE_E1S_OUT / "pure_loading_01"
    ref = np.array(json.loads((pure / "provenance.json").read_text())["reference_rotation_world_from_load"])
    rows = {}
    for i in range(1, 5):
        name = f"c{i:03d}_r00"
        with np.load(pure / f"{name}.npz", allow_pickle=False) as z:
            data = dict(z)
        values = []
        for t in range(30, 37):
            args = [list(data[k][:t+1]) for k in ["tactile", "state4", "mcf", "joint_pos", "joint_vel", "action"]]
            xy, grip, gate, recenter = step(model, ck, *args, ref, False, np.random.default_rng(0))
            ok = (xy[0] < -.7) if i == 1 else (xy[0] > .7) if i == 2 else (np.max(np.abs(xy)) < 1e-6 and gate < .45) if i == 3 else (xy[1] > .7)
            sign_ok = (xy[0] < 0) if i == 1 else (xy[0] > 0) if i == 2 else ok if i == 3 else (xy[1] > 0)
            values.append(dict(t=t, u_xy=xy.tolist(), grip=grip, gate=gate, passed=bool(ok),
                               sign_only_passed=bool(sign_ok),
                               no_camera_recenter_zero=bool(np.max(np.abs(recenter)) == 0)))
        rows[name] = values
    torch.manual_seed(17)
    xt, xp, xv, energy = torch.randn(16, 48), torch.randn(16, 12), torch.zeros(16, 6), torch.ones(16, 1)
    o = model(xt, xp, xv, energy)
    zero_vis = bool(torch.equal(o["u_raw"], o["u_damp"]) and torch.count_nonzero(o["g_vis"]) == 0)
    o["u"].square().mean().backward()
    frozen = base_equal and all(not p.requires_grad and p.grad is None for p in model.frozen_bc_base.parameters())
    residual_bound = float((model.residual_alpha * o["delta_u"]).detach().abs().max())
    # Deliberately nonzero initial misalignment must not report zero error.
    geom = socket_alignment(np.array([.003, .004, .1]), np.eye(3), np.zeros(3), np.eye(3), np.array([0, 0, .03]))
    geometry_ok = abs(geom["radial_error_mm"] - 5.) < 1e-9 and abs(geom["tip_lateral_error_mm"] - 5.) < 1e-9
    angle = .01
    q_test = 1.00001 * np.array([np.cos(np.deg2rad(angle) / 2), 0, 0, np.sin(np.deg2rad(angle) / 2)])
    quaternion_ok = abs(quaternion_angle_deg(q_test, [1.00001, 0, 0, 0]) - angle) < 1e-6
    off = json.loads((ROOT / "outputs/v2_2_verification_results.json").read_text())
    arch = off["innovations_2_and_3_architecture_and_online_rl"]
    comparisons = {}
    for split in ["dev_pure_c005_c008", "blind_fresh_b0_b1"]:
        results = arch[split]
        proposed = results["5_V22_Surrogate_Residual_RL_With_Aux_Proposed"]
        comparisons[split] = {baseline: {metric: bool(proposed[metric] > results[baseline][metric]) if metric == "wrench_r2"
                            else bool(proposed[metric] < results[baseline][metric])
                            for metric in ["clean_action_mse", "occluded_action_mse", "wrench_r2", "est_peak_slip_clean_mm", "est_peak_slip_occluded_mm"]}
                            for baseline in ["3_V22_DualHead_WarmStart_BC", "4_V22_Surrogate_Residual_RL_No_Aux"]}
    report = dict(onset_direction_checks=rows, onset_passed=sum(r["passed"] for rs in rows.values() for r in rs),
                  onset_sign_only_passed=sum(r["sign_only_passed"] for rs in rows.values() for r in rs),
                  onset_total=28, missing_visual_exact_damping=zero_vis, frozen_bc_unchanged_and_no_grad=frozen,
                  residual_correction_max_abs=residual_bound, residual_correction_bounded=residual_bound <= .35 + 1e-7,
                  absolute_socket_geometry_test=geometry_ok, offline_metric_comparisons=comparisons)
    report["normalized_quaternion_small_angle_test"] = quaternion_ok
    cl_path = ROOT / "outputs/isaac_closedloop_v2_2_residual_rl_report.json"
    if cl_path.exists():
        cl = json.loads(cl_path.read_text())
        report["physical_run_inputs_match_current_files"] = bool(cl.get("input_sha256")) and all(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
            for name, digest in cl.get("input_sha256", {}).items())
        a, b = cl["part_a_6controller_closedloop"], cl["part_b_3stage_long_horizon_benchmark"]
        prop = "6_v2_2_online_residual_rl_proposed"
        report["physical_acceptance"] = {
            "matched_external_controls": cl.get("control_fairness_audit", {}).get("gains_and_gates_matched_across_all_3dof_policies", False),
            "part_a_proposed_beats_baselines": {k: a[prop]["loaded_mean_inhand_slip_mm"] < a[k]["loaded_mean_inhand_slip_mm"] for k in ["3_v2_1_pure_bpi_3dof_singlehead", "4_v2_2_online_rl_no_aux_collapse", "5_v2_2_warmstart_bc_dualhead"]},
            "zero_jitter_le_016deg": a[prop].get("zero_disturbance_jitter_deg", float("inf")) <= .16,
            "part_b_proposed_best_by_metric": {metric: b[prop][metric] <= min(r[metric] for r in b.values()) for metric in ["stage3_mean_socket_radial_error_mm", "stage3_mean_socket_axis_tilt_deg", "stage3_mean_socket_tip_lateral_error_mm"]},
        }
    fresh = ROOT / "data/fresh_trials"
    manifest = json.loads((fresh / "manifest.json").read_text())
    report["fresh_data_self_contained"] = all((fresh / f"{trial}{suffix}").exists()
                                              for trial in manifest["completed_trials"] for suffix in [".npz", ".json"])
    report["fresh_completed_trials_including_zero"] = len(manifest["completed_trials"])
    report["unit_audits_five_passed"] = all(off["unit_audits"][k] for k in ["policy_gradient_non_zero_verified",
        "prefix_causality_strictly_zero", "missing_tactile_raises_keyerror", "proprioceptive_inputs_genuinely_used",
        "surrogate_uses_wrench4d_verified"])
    (ROOT / "outputs/secondary_audit_20261104.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: value for k, value in report.items() if k != "onset_direction_checks"}, indent=2))


if __name__ == "__main__":
    main()
