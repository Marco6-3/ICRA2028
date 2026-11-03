"""Generate publication-ready figures (PNG + PDF) for the V2.1 BPI-MCF & 3-DoF Reflex archive."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent / "outputs" / "v2_1_clean_study"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_fig1_bpi_mcf_ablation(v21_report: dict) -> None:
    stages = v21_report["part1_8stage_bpi_component_ablation"]
    short_labels = [
        "S0: Raw16D",
        "S1: Scalar8D",
        "S2: StdMCF",
        "S3: RefLock(k=5)",
        "S4: +PreloadNorm",
        "S5: +SymAntiDipole",
        "S6: +StaticManifold",
        "S7: Full BPI-MCF",
    ]
    dev_acc = [stages[k]["dev_pure_c005_c008_centroid_acc"] * 100.0 for k in stages]
    fresh_inst = [stages[k]["blind_fresh12_instantaneous_centroid_acc"] * 100.0 for k in stages]
    fresh_b0b1 = [stages[k]["blind_fresh_b0_b1_step_ramp_centroid_acc"] * 100.0 for k in stages]
    pt02_acc = [stages[k]["pt02_36ood_centroid_acc"] * 100.0 for k in stages]

    x = np.arange(len(short_labels))
    width = 0.20

    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=220)
    b1 = ax.bar(x - 1.5 * width, dev_acc, width, label="Dev: pure_loading_01 (c005-c008)", color="#2b5c8f")
    b2 = ax.bar(x - 0.5 * width, fresh_inst, width, label="Blind Held-out: Fresh12 (All 12)", color="#d95f02")
    b3 = ax.bar(x + 0.5 * width, fresh_b0b1, width, label="Blind Held-out: Fresh (b0+b1)", color="#7570b3")
    b4 = ax.bar(x + 1.5 * width, pt02_acc, width, label="OOD: pilot_tangent_02 (36 conds)", color="#1b9e77")

    ax.axhline(25.0, color="gray", linestyle="--", linewidth=1.0, label="4-Class Chance (25%)")
    ax.set_ylabel("Shear Direction Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Innovation 1: 8-Stage Frozen BPI-MCF Component Ablation Across Dev, Blind Fresh12 & OOD Splits", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=18, ha="right", fontsize=9.5)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", ncol=2, fontsize=8.8, framealpha=0.92)

    for bars in (b1, b2, b4):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_bpi_mcf_8stage_ablation.png", dpi=220)
    fig.savefig(FIG_DIR / "fig1_bpi_mcf_8stage_ablation.pdf")
    plt.close(fig)


def plot_fig2_closedloop_comparison(cl_report: dict) -> None:
    summary = cl_report["part_a_unconfounded_7controller_closedloop"]
    order = [
        ("1_nominal_open_loop", "1. Open-Loop\n(0D)"),
        ("4_v2_1_pure_scalar_3dof", "4. Pure Scalar\n3-DoF (A1)"),
        ("2_stage3s_1d_grip_only", "2. Stage 3S\n1D Grip (Base)"),
        ("3_v2_1_pure_bpi_1d_grip_only", "3. Pure BPI\n1D Grip"),
        ("5_v2_1_pure_bpi_3dof_no_gate", "5. Pure BPI\n3-DoF No-Gate (B3)"),
        ("7_v2_1_hybrid_bpi_3dof_plus_stage3s", "7. Hybrid BPI\n3-DoF + Stage3S"),
        ("6_v2_1_pure_bpi_3dof_no_stage3s", "6. Pure BPI 3-DoF\n+ Aux + Gate (B5)"),
    ]

    labels = [lbl for _, lbl in order]
    mean_slip = [summary[k]["loaded_mean_inhand_slip_mm"] for k, _ in order]
    worst_slip = [summary[k]["loaded_worst_inhand_slip_mm"] for k, _ in order]
    preload_jitter = [summary[k]["preload_mean_jitter_mm"] for k, _ in order]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=220)
    b1 = ax.bar(x - width, mean_slip, width, label="Loaded 4-Dir Mean Slip (mm) [Lower is Better]", color="#1f78b4")
    b2 = ax.bar(x, worst_slip, width, label="Worst-Direction (+Fx) Slip (mm) [Lower is Better]", color="#e31a1c")
    b3 = ax.bar(x + width, preload_jitter, width, label="Preload Quiet-Window Jitter (mm) [Lower is Better]", color="#33a02c")

    ax.axhline(2.0, color="#555555", linestyle="--", linewidth=1.0, label="2.0 mm Safety Reference")
    ax.set_ylabel("Displacement / Slip (mm)", fontsize=11, fontweight="bold")
    ax.set_title("Isaac Lab Unconfounded 7-Controller Closed-Loop Benchmark (0.20 N Eccentric Shear @ 15 mm)", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.2)
    ax.set_ylim(0, 24.0)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9.0, framealpha=0.92)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold" if h < 1.0 else "normal",
            )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_isaac_closedloop_7controllers.png", dpi=220)
    fig.savefig(FIG_DIR / "fig2_isaac_closedloop_7controllers.pdf")
    plt.close(fig)


def plot_fig3_longhorizon_benchmark(cl_report: dict) -> None:
    lh = cl_report["part_b_3stage_long_horizon_benchmark"]
    order = [
        ("1_nominal_open_loop", "1. Open-Loop (0D)"),
        ("4_v2_1_pure_scalar_3dof", "4. Pure Scalar 3-DoF (A1)"),
        ("2_stage3s_1d_grip_only", "2. Stage 3S 1D Grip Baseline"),
        ("6_v2_1_pure_bpi_3dof_no_stage3s", "6. Pure BPI-MCF 3-DoF (B5, Proposed)"),
    ]

    labels = [lbl for _, lbl in order]
    s1_jitter = [lh[k]["stage1_mean_jitter_mm"] for k, _ in order]
    s2_slip = [lh[k]["stage2_mean_transport_slip_mm"] for k, _ in order]
    s3_slip = [lh[k]["stage3_mean_align_slip_mm"] for k, _ in order]
    tip_err = [lh[k]["stage3_mean_final_tip_error_mm"] for k, _ in order]

    x = np.arange(len(labels))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=220)
    b1 = ax.bar(x - 1.5 * width, s1_jitter, width, label="Stage 1: Preload Jitter (mm)", color="#6a3d9a")
    b2 = ax.bar(x - 0.5 * width, s2_slip, width, label="Stage 2: Occluded Transport Slip (mm)", color="#ff7f00")
    b3 = ax.bar(x + 0.5 * width, s3_slip, width, label="Stage 3: Reversed-Shear Slip (mm)", color="#1f78b4")
    b4 = ax.bar(x + 1.5 * width, tip_err, width, label="Stage 3: Final Tip Alignment Error (mm)", color="#e31a1c")

    ax.axhline(2.0, color="#555555", linestyle="--", linewidth=1.0, label="2.0 mm Socket Insertion Tolerance")
    ax.set_ylabel("Error / Slip (mm)", fontsize=11, fontweight="bold")
    ax.set_title("Isaac Lab 3-Stage Long-Horizon Manipulation Benchmark (Preload -> Occluded Transport -> Reversed Alignment)", fontsize=11.0, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 10.0)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92)

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_isaac_longhorizon_3stage.png", dpi=220)
    fig.savefig(FIG_DIR / "fig3_isaac_longhorizon_3stage.pdf")
    plt.close(fig)


def main() -> None:
    v21_report = json.loads((BASE_DIR / "v2_1_verification_report.json").read_text(encoding="utf-8"))
    cl_report = json.loads((BASE_DIR / "isaac_closedloop_v2_1_and_longhorizon_report.json").read_text(encoding="utf-8"))
    plot_fig1_bpi_mcf_ablation(v21_report)
    plot_fig2_closedloop_comparison(cl_report)
    plot_fig3_longhorizon_benchmark(cl_report)
    print(f"[OK] Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
