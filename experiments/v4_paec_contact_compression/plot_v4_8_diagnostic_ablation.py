#!/usr/bin/env python3
"""
Generate publication-grade 300 DPI PNG + PDF figures for V4.8 Diagnostic,
Fair-Ablation & Encoder-Side Drift-Fix Suite:
  - fig8_v4_8_drift_and_stiffness_probe.{png,pdf}
  - fig9_v4_8_failure_anatomy_and_tasks.{png,pdf}
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_fig8(probe_json: dict, out_dir: Path):
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    stiff = probe_json["virtual_stiffness_probe"]["regimes"]
    regimes = [
        ("clean", "Clean"),
        ("thermal_drift_0p5", "+0.5σ Static"),
        ("static_drift_1p5", "+1.5σ Static"),
        ("step_drift_1p5_t35", "+1.5σ Step (t≥35)"),
    ]
    encoders = [
        ("PAEC_Small_d32", "PAEC (V4.4)", "#0072b2"),
        ("PAEC_Small_d32_InitTare", "PAEC + InitTare", "#56b4e9"),
        ("B4_NoBG_Small_d32", "B4 NoBG (V4.6)", "#999999"),
        ("PAEC_Small_d32_PureDiff", "PAEC PureDiff (V4.8)", "#009e73"),
        ("PAEC_Small_d32_WideDrift", "PAEC WideDrift (V4.8)", "#d55e00"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    x = np.arange(len(regimes))
    width = 0.155
    offsets = np.linspace(-2 * width, 2 * width, len(encoders))

    # Panel A: Zero-Shot 1D Probe Pearson r
    ax = axes[0]
    for idx, (ek, elbl, col) in enumerate(encoders):
        vals = [stiff[rk][ek]["probe_kz_zero_shot"]["pearson_r_k"] for rk, _ in regimes]
        bars = ax.bar(x + offsets[idx], vals, width=width, color=col, alpha=0.92, label=elbl)
        for br, v in zip(bars, vals):
            y_txt = max(v, 0.0) + 0.02 if v >= 0 else v - 0.06
            ax.text(br.get_x() + br.get_width() / 2.0, y_txt, f"{v:.2f}", ha="center", va="bottom", fontsize=6.0, rotation=90)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([rl for _, rl in regimes], fontsize=8.2)
    ax.set_ylim(-0.18, 1.12)
    ax.set_ylabel("Pearson r (↑)")
    ax.set_title("(a) Zero-Shot 1D Probe Stiffness Corr.")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Panel B: Latent z Ridge R2_k
    ax = axes[1]
    for idx, (ek, elbl, col) in enumerate(encoders):
        vals = [max(-0.25, stiff[rk][ek]["latent_z_ridge"]["r2_k"]) for rk, _ in regimes]
        raw_vals = [stiff[rk][ek]["latent_z_ridge"]["r2_k"] for rk, _ in regimes]
        bars = ax.bar(x + offsets[idx], vals, width=width, color=col, alpha=0.92, label=elbl)
        for br, v, rv in zip(bars, vals, raw_vals):
            lbl = f"{rv:.2f}" if rv >= -0.25 else f"{rv:.1f}"
            y_txt = max(v, 0.0) + 0.02 if v >= 0 else -0.22
            ax.text(br.get_x() + br.get_width() / 2.0, y_txt, lbl, ha="center", va="bottom", fontsize=6.0, rotation=90)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([rl for _, rl in regimes], fontsize=8.2)
    ax.set_ylim(-0.28, 1.08)
    ax.set_ylabel("Held-Out R²_k (↑)")
    ax.set_title("(b) Latent z Stiffness Regression (R²_k)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Panel C: Latent z Soft-Defect Detection AUROC (< 1000 N/m)
    ax = axes[2]
    for idx, (ek, elbl, col) in enumerate(encoders):
        vals = [stiff[rk][ek]["latent_z_ridge"]["soft_defect_auroc"] for rk, _ in regimes]
        bars = ax.bar(x + offsets[idx], vals, width=width, color=col, alpha=0.92, label=elbl)
        for br, v in zip(bars, vals):
            ax.text(br.get_x() + br.get_width() / 2.0, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.0, rotation=90)
    ax.axhline(0.5, color="#888888", linewidth=0.9, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([rl for _, rl in regimes], fontsize=8.2)
    ax.set_ylim(0.40, 1.12)
    ax.set_ylabel("Soft-Defect AUROC (↑)")
    ax.set_title("(c) Soft-Zone Defect AUROC (<1000 N/m)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.4, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle(
        "V4.8 Invariance vs. Preload-Observability Trade-Off on Zero-Dither Stiffness Probe (N=48 Train / 48 Test)",
        fontsize=11.5,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    for ext in ("png", "pdf"):
        p = out_dir / f"fig8_v4_8_drift_and_stiffness_probe.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig9(cl_json: dict, out_dir: Path):
    plt.rcParams.update(
        {
            "font.size": 8.8,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    parts = cl_json["parts"]
    fig, axes = plt.subplots(2, 2, figsize=(16.2, 10.0))

    # Panel A: auto_connector Failure Anatomy (+0.5sigma Thermal Drift)
    ax = axes[0, 0]
    p1 = parts["part1_auto_connector_anatomy"]["act_methods"]
    m_list_1 = [
        ("ACT_RGB_Proprio", "Proprio"),
        ("ACT_RGB_RawFT", "RawFT"),
        ("ACT_RGB_RawHistGRU", "RawHist\nGRU"),
        ("ACT_RGB_MAE", "MAE"),
        ("ACT_RGB_JEPA", "JEPA"),
        ("ACT_RGB_NoBG", "NoBG"),
        ("ACT_RGB_PAEC", "PAEC"),
        ("ACT_RGB_PAEC_PureDiff", "PAEC\nPureDiff"),
        ("ACT_RGB_PAEC_WideDrift", "PAEC\nWideDrift"),
    ]
    x = np.arange(len(m_list_1))
    w = 0.26
    succ_05 = [p1[m]["thermal_drift_0p5"]["success_rate_mean"] for m, _ in m_list_1]
    viol_05 = [p1[m]["thermal_drift_0p5"]["safety_viol_rate_mean"] for m, _ in m_list_1]
    prem_05 = [p1[m]["thermal_drift_0p5"].get("premature_yield_rate_of_failed_mean") or 0.0 for m, _ in m_list_1]

    b1 = ax.bar(x - w, succ_05, width=w, color="#009e73", label="Success % (↑)")
    b2 = ax.bar(x, viol_05, width=w, color="#d55e00", label="Overload Viol % (↓)")
    b3 = ax.bar(x + w, prem_05, width=w, color="#56b4e9", label="Premature Yield % of Fail")
    for bars in (b1, b2, b3):
        for br in bars:
            v = br.get_height()
            if v >= 1.0:
                ax.text(br.get_x() + br.get_width() / 2.0, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in m_list_1], fontsize=7.6)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(a) Task 2 (auto_connector) Failure Anatomy @ +0.5σ Drift")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    # Panel B: battery_scan 9-Method Comparison (Clean / +0.5sigma / +1.5sigma)
    ax = axes[0, 1]
    p2 = parts["part2_battery_scan_fair"]["act_methods"]
    regimes = [
        ("clean", "Clean", "#0072b2"),
        ("thermal_drift_0p5", "+0.5σ Thermal", "#e69f00"),
        ("extreme_drift_1p5", "+1.5σ Extreme", "#d55e00"),
    ]
    offsets = [-w, 0.0, w]
    for idx, (rk, rlbl, col) in enumerate(regimes):
        vals = [p2[m][rk]["success_rate_mean"] for m, _ in m_list_1]
        viols = [p2[m][rk]["safety_viol_rate_mean"] for m, _ in m_list_1]
        bars = ax.bar(x + offsets[idx], vals, width=w, color=col, alpha=0.92, label=rlbl)
        for br, v, vio in zip(bars, vals, viols):
            txt = f"{v:.0f}" if vio < 0.5 else f"{v:.0f}\n({vio:.0f}!)"
            ax.text(br.get_x() + br.get_width() / 2.0, v + 1.2, txt, ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in m_list_1], fontsize=7.6)
    ax.set_ylim(0, 122)
    ax.set_ylabel("Success Rate % (↑)")
    ax.set_title("(b) Task 1 (battery_scan): 9-Method Fair Comparison (72 ep/cell)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=7.8, loc="upper right", ncol=3)

    # Panel C: auto_connector_chamfer (No-Oracle Recovery Demos)
    ax = axes[1, 0]
    p3 = parts["part3_auto_connector_chamfer"]
    m_list_3 = [
        ("Classical_Guided", "Classical\nGuided"),
        ("ACT_RGB_Proprio", "Proprio"),
        ("ACT_RGB_RawFT", "RawFT"),
        ("ACT_RGB_RawHistGRU", "RawHist\nGRU"),
        ("ACT_RGB_NoBG", "NoBG"),
        ("ACT_RGB_PAEC", "PAEC"),
        ("ACT_RGB_PAEC_PureDiff", "PAEC\nPureDiff"),
        ("ACT_RGB_PAEC_WideDrift", "PAEC\nWideDrift"),
    ]
    x3 = np.arange(len(m_list_3))
    for idx, (rk, rlbl, col) in enumerate(regimes):
        vals, viols = [], []
        for mk, _ in m_list_3:
            d = p3["classical_guided"][rk] if mk == "Classical_Guided" else p3["act_methods"][mk][rk]
            vals.append(d["success_rate_mean"])
            viols.append(d["safety_viol_rate_mean"])
        bars = ax.bar(x3 + offsets[idx], vals, width=w, color=col, alpha=0.92, label=rlbl)
        for br, v, vio in zip(bars, vals, viols):
            txt = f"{v:.0f}" if vio < 0.5 else f"{v:.0f}\n({vio:.0f}!)"
            ax.text(br.get_x() + br.get_width() / 2.0, v + 1.2, txt, ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x3)
    ax.set_xticklabels([lbl for _, lbl in m_list_3], fontsize=7.8)
    ax.set_ylim(0, 92)
    ax.set_ylabel("Success Rate % (↑)")
    ax.set_title("(c) Task 2b (auto_connector_chamfer): No-Oracle Recovery")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Panel D: semi_grasp Pure E2E ACT vs Hierarchical FSM + Contact Primitives
    ax = axes[1, 1]
    p4 = parts["part4_semi_grasp_fsm"]
    m_list_4 = [
        ("Pure_E2E_Ref", "Pure E2E\nACT+PAEC\n(V4.7 Ref)"),
        ("FSM_Classical_ForceThresh", "FSM +\nClassical Fz"),
        ("FSM_RawFT_Primitive", "FSM +\nRawFT"),
        ("FSM_NoBG_Primitive", "FSM +\nNoBG"),
        ("FSM_PAEC_Primitive", "FSM +\nPAEC"),
        ("FSM_PAEC_PureDiff_Primitive", "FSM +\nPureDiff"),
        ("FSM_PAEC_WideDrift_Primitive", "FSM +\nWideDrift"),
    ]
    x4 = np.arange(len(m_list_4))
    ref = p4["v4_7_pure_e2e_act_paec_ref"]
    ref_map = {"clean": ref["clean_succ"], "thermal_drift_0p5": ref["thermal_0p5_succ"], "extreme_drift_1p5": ref["extreme_1p5_succ"]}
    for idx, (rk, rlbl, col) in enumerate(regimes):
        vals, viols = [], []
        for mk, _ in m_list_4:
            if mk == "Pure_E2E_Ref":
                vals.append(ref_map[rk])
                viols.append(0.0 if rk != "extreme_drift_1p5" else 11.1)
            else:
                d = p4["fsm_methods"][mk][rk]
                vals.append(d["success_rate_mean"])
                viols.append(d["safety_viol_rate_mean"])
        bars = ax.bar(x4 + offsets[idx], vals, width=w, color=col, alpha=0.92, label=rlbl)
        for br, v, vio in zip(bars, vals, viols):
            txt = f"{v:.0f}" if vio < 0.5 else f"{v:.0f}\n({vio:.0f}!)"
            ax.text(br.get_x() + br.get_width() / 2.0, v + 1.2, txt, ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x4)
    ax.set_xticklabels([lbl for _, lbl in m_list_4], fontsize=7.6)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Success Rate % (↑)")
    ax.set_title("(d) Task 3 (semi_grasp): Pure E2E ACT vs. Hierarchical FSM + Primitive")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle(
        "V4.8 Closed-Loop Diagnostics, Fair Baselines & Structural Fixes (15 Demos × 3 Seeds × 24 Paired Eps = 72 Rollouts/Cell)",
        fontsize=11.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"fig9_v4_8_failure_anatomy_and_tasks.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-json",
        type=str,
        default="experiments/v4_paec_contact_compression/outputs/v4_8_diagnostic_ablation/v4_8_encoder_drift_probe.json",
    )
    parser.add_argument(
        "--closedloop-json",
        type=str,
        default="experiments/v4_paec_contact_compression/outputs/v4_8_diagnostic_ablation/v4_8_closedloop_results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/v4_paec_contact_compression/outputs/v4_8_diagnostic_ablation",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_json = json.load(open(args.probe_json, encoding="utf-8"))
    cl_json = json.load(open(args.closedloop_json, encoding="utf-8"))

    plot_fig8(probe_json, out_dir)
    plot_fig9(cl_json, out_dir)
    print(f"[Plot V4.8] Saved fig8 & fig9 PNG/PDF -> {out_dir}")


if __name__ == "__main__":
    main()
