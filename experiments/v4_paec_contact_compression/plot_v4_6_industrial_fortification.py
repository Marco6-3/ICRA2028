#!/usr/bin/env python3
"""Generate publication-quality 300 DPI summary figure for V4.6 Industrial Scientific Fortification."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-json",
        type=str,
        default="experiments/v4_paec_contact_compression/outputs/v4_6_industrial_fortification/v4_6_results.json",
    )
    parser.add_argument(
        "--v44-json",
        type=str,
        default="experiments/v4_paec_contact_compression/outputs/v4_4_a6000_scale/v4_scaling_results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/v4_paec_contact_compression/outputs/v4_6_industrial_fortification",
    )
    args = parser.parse_args()

    v46 = json.load(open(args.results_json, encoding="utf-8"))
    v44 = json.load(open(args.v44_json, encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))

    # ------------------------------------------------------------------
    # Panel (a): Classical Admittance vs Learned Policies across 3 Regimes (10-shot / Full)
    # ------------------------------------------------------------------
    ax = axes[0]
    regimes = [
        ("clean", "Clean"),
        ("static_drift_1p5", "Static +1.5σ\nDrift"),
        ("step_drift_1p5", "Step +1.5σ\nDrift (t≥35)"),
    ]
    classical = v46["pillar2_classical_admittance"]
    cl_abl = v46["pillar1_closedloop_ablation"]

    series = [
        ("Admittance_FixedFeed", "Classical Fixed-Feed", "#6b7280", True),
        ("Admittance_TunedShear", "Classical Tuned+Shear", "#9ca3af", True),
        ("Admittance_InitTare", "Classical Init-Tare", "#7c3aed", True),
        ("B2b_MAE_Small_d16", "B2b MAE (10d)", "#e69f00", False),
        ("PAEC_Small_d32", "PAEC-Small d32 (10d, Ours)", "#0072b2", False),
    ]
    x = np.arange(len(regimes))
    width = 0.155
    offsets = np.linspace(-2 * width, 2 * width, len(series))

    for idx, (key, label, col, is_cls) in enumerate(series):
        vals = []
        for rk, _ in regimes:
            d = classical[key][rk] if is_cls else cl_abl[key]["10demos"][rk]
            vals.append(d["success_rate_mean"])
        bars = ax.bar(x + offsets[idx], vals, width=width, color=col, alpha=0.92, label=label)
        for br, val in zip(bars, vals):
            ax.text(
                br.get_x() + br.get_width() / 2.0,
                val + 1.5,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([rl for _, rl in regimes], fontsize=8.3)
    ax.set_ylim(0, 128)
    ax.set_ylabel("Closed-Loop Success Rate (%) ↑")
    ax.set_title("(a) Classical Admittance vs. Frozen PAEC")
    ax.legend(frameon=True, framealpha=0.94, edgecolor="none", fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    ax.grid(axis="y", alpha=0.25)

    # ------------------------------------------------------------------
    # Panel (b): Same-Capacity Ablation (Small_d32 / Small_d16)
    # ------------------------------------------------------------------
    ax = axes[1]
    b4_off = v46["pillar1_b4_nobg_small_d32_offline"]
    exp_b = v44["exp_b_capacity_bottleneck"]

    def _rdp10_mean(entry: dict) -> float:
        return float(np.mean([entry[k]["bc"]["10eps"]["test_all_mse_mean"] for k in ("rdp_peel", "rdp_wipe", "rdp_lift")]))

    abl_models = [
        ("PAEC_Small_d32", "PAEC\nSmall d32", "#0072b2"),
        ("B4_NoBG_Small_d32", "B4 No-BG\nSmall d32", "#cc79a7"),
        ("B2b_MAE_Small_d16", "B2b MAE\nSmall d16", "#e69f00"),
        ("B3b_JEPA_Small_d16", "B3b JEPA\nSmall d16", "#6b7280"),
    ]
    xs_b = np.arange(len(abl_models))
    mean_drift_succ = []
    rdp10_mses = []
    for mk, _, _ in abl_models:
        all_dr = []
        for bk in ("5demos", "10demos", "20demos"):
            all_dr.append(cl_abl[mk][bk]["static_drift_1p5"]["success_rate_mean"])
            all_dr.append(cl_abl[mk][bk]["step_drift_1p5"]["success_rate_mean"])
        mean_drift_succ.append(float(np.mean(all_dr)))
        src = b4_off if mk == "B4_NoBG_Small_d32" else exp_b[mk]
        rdp10_mses.append(_rdp10_mean(src))

    b_bars = ax.bar(xs_b - 0.17, mean_drift_succ, width=0.33, color=[c for _, _, c in abl_models], alpha=0.92, label="Drift CL Succ (%) ↑")
    for br, v in zip(b_bars, mean_drift_succ):
        ax.text(br.get_x() + br.get_width() / 2.0, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(xs_b)
    ax.set_xticklabels([lbl for _, lbl, _ in abl_models], fontsize=7.9)
    ax.set_ylim(0, 124)
    ax.set_ylabel("MuJoCo Mean Drift Success (%) ↑")

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    l_mse = ax2.plot(xs_b + 0.17, rdp10_mses, "-o", color="#111827", linewidth=1.9, markersize=5.8, label="RH20T→RDP 10-Shot MSE ↓")
    for xi, ym in zip(xs_b + 0.17, rdp10_mses):
        ax2.text(xi, ym + 0.00045, f"{ym:.4f}", ha="center", va="bottom", fontsize=6.7, bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.75))
    y_lo = min(rdp10_mses) - 0.0032
    y_hi = max(rdp10_mses) + 0.0032
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_ylabel("RDP 10-Shot BC MSE ↓")
    ax.set_title("(b) Same-Scale Ablation (PAEC vs B4_NoBG / MAE / JEPA)")
    ax.legend([b_bars[0], l_mse[0]], ["Mean Drift Succ (%) ↑", "RDP 10-Shot MSE ↓"], frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.0, loc="lower left")
    ax.grid(axis="y", alpha=0.25)

    # ------------------------------------------------------------------
    # Panel (c): Quantitative Virtual Stiffness Probe (PAEC vs B4_NoBG, Clean)
    # ------------------------------------------------------------------
    ax = axes[2]
    pr = v46["pillar3_virtual_stiffness_probe"]
    k_true = np.array(pr["test_platen_k_true"], dtype=np.float64)
    paec_18 = pr["regimes"]["clean"]["PAEC_Small_d32"]["probe_18d_ridge"]
    nobg_18 = pr["regimes"]["clean"]["B4_NoBG_Small_d32"]["probe_18d_ridge"]
    zs_paec = pr["regimes"]["clean"]["PAEC_Small_d32"]["probe_kz_zero_shot"]
    zs_nobg = pr["regimes"]["clean"]["B4_NoBG_Small_d32"]["probe_kz_zero_shot"]
    dr_paec = pr["regimes"]["static_drift_1p5"]["PAEC_Small_d32"]["probe_kz_zero_shot"]

    k_pred_paec = np.clip(np.array(paec_18["test_pred_k"], dtype=np.float64), 350.0, 3200.0)
    k_pred_nobg = np.clip(np.array(nobg_18["test_pred_k"], dtype=np.float64), 350.0, 3200.0)

    k_min, k_max = 380.0, 3100.0
    ax.plot([k_min, k_max], [k_min, k_max], "--", color="#9ca3af", linewidth=1.2, label="Ideal y = x")
    ax.axvline(1000.0, color="#ef4444", linestyle=":", linewidth=1.1, alpha=0.75, label="Soft-zone cutoff (1 kN/m)")
    ax.scatter(
        k_true, k_pred_paec, s=28, color="#0072b2", alpha=0.82,
        label=f"PAEC-Small 18D Probe (R²={paec_18['r2_k']:.2f}, r={paec_18['pearson_r_k']:.2f})",
    )
    ax.scatter(
        k_true, k_pred_nobg, s=26, marker="^", color="#cc79a7", alpha=0.78,
        label=f"B4_NoBG 18D Probe (R²={nobg_18['r2_k']:.2f}, r={nobg_18['pearson_r_k']:.2f})",
    )
    info_box = (
        f"Zero-Shot 1D Probe g_φ(z_t, ±e_z):\n"
        f"  PAEC Clean: r={zs_paec['pearson_r_k']:+.3f}, ρ={zs_paec['spearman_rho_k']:+.3f}, AUC={zs_paec['soft_defect_auroc']:.3f}\n"
        f"  NoBG Clean: r={zs_nobg['pearson_r_k']:+.3f}, ρ={zs_nobg['spearman_rho_k']:+.3f}, AUC={zs_nobg['soft_defect_auroc']:.3f}\n"
        f"  Boundary (+1.5σ Drift): r={dr_paec['pearson_r_k']:+.3f} (collapse)"
    )
    ax.text(
        0.03, 0.97, info_box, transform=ax.transAxes, va="top", ha="left",
        fontsize=6.5, bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95),
    )
    ax.set_xlim(k_min, 2750.0)
    ax.set_ylim(k_min, k_max)
    ax.set_xlabel("Ground-Truth Surface Stiffness platen_k (N/m)")
    ax.set_ylabel("Predicted Stiffness (N/m)")
    ax.set_title("(c) Virtual Stiffness Probe (N=48 Held-Out)")
    ax.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=6.6, loc="lower right")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "V4.6 Industrial Fortification: Classical Admittance, Matched-Capacity B4_NoBG, & Virtual Stiffness Probe",
        y=1.02,
        fontsize=11.3,
        fontweight="bold",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig6_v4_6_industrial_fortification.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig6_v4_6_industrial_fortification.png & .pdf to", out_dir)


if __name__ == "__main__":
    main()
