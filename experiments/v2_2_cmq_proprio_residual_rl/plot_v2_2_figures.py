"""
Generate publication-ready figures (PNG + PDF) for the Stage 4 (V2.2)
CMQ-BPI-MCF + Proprioceptive-Tactile Dual-Head + Online Residual RL archive.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_fig1_10stage_ablation(v22_report: dict) -> None:
    stages = v22_report["innovation_1_10stage_operator_ablation"]
    short_labels = [
        "S0: Raw16D",
        "S1: Scalar8D",
        "S2: StdMCF",
        "S3: RefLock(k=5)",
        "S4: +PreloadNorm",
        "S5: +SymAntiDipole",
        "S6: +StaticManifold",
        "S7: V2.1 BPI-MCF",
        "S8: +Creep&ScaleInv",
        "S9: V2.2 CMQ-BPI-MCF",
    ]
    keys = list(stages.keys())
    dev_acc = [stages[k]["dev_pure_c005_c008_centroid_acc"] * 100.0 for k in keys]
    fresh_inst = [stages[k]["blind_fresh12_instantaneous_centroid_acc"] * 100.0 for k in keys]
    fresh_b0b1 = [stages[k]["blind_fresh_b0_b1_step_ramp_centroid_acc"] * 100.0 for k in keys]
    pt02_acc = [stages[k]["pt02_36ood_centroid_acc"] * 100.0 for k in keys]

    x = np.arange(len(short_labels))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13.0, 5.0), dpi=220)
    b1 = ax.bar(x - 1.5 * width, dev_acc, width, label="Dev: pure_loading_01 (c005-c008)", color="#2b5c8f")
    b2 = ax.bar(x - 0.5 * width, fresh_inst, width, label="Held-out (previously evaluated): Fresh12 (Instantaneous Dir, All 12 Trials)", color="#d95f02")
    b3 = ax.bar(x + 0.5 * width, fresh_b0b1, width, label="Held-out (previously evaluated): Fresh (b0+b1 Step+Ramp)", color="#7570b3")
    b4 = ax.bar(x + 1.5 * width, pt02_acc, width, label="OOD: pilot_tangent_02 (36 Table-Supported Conds)", color="#1b9e77")

    ax.axhline(25.0, color="gray", linestyle="--", linewidth=1.0, label="4-Class Chance (25%)")
    ax.axvline(7.5, color="#b2182b", linestyle="-.", linewidth=1.3, label="V2.1 -> V2.2 Upgrade Boundary")
    ax.set_ylabel("Shear Direction Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Innovation 1: 10-Stage Contact Operator Ablation (From Raw State4 & V2.1 BPI-MCF to V2.2 CMQ-BPI-MCF)", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=18, ha="right", fontsize=9.2)
    ax.set_ylim(0, 109)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=8.6, frameon=False)

    for bars in (b1, b2, b3, b4):
        for idx, bar in enumerate(bars):
            if idx in (2, 7, 8, 9):
                h = bar.get_height()
                ax.annotate(
                    f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    fontweight="bold" if idx >= 8 else "normal",
                )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_10stage_cmq_ablation.png", dpi=220)
    fig.savefig(FIG_DIR / "fig1_10stage_cmq_ablation.pdf")
    plt.close(fig)


def plot_fig2_pt02_7category_ood(v22_report: dict) -> None:
    pt = v22_report["innovation_1b_pt02_7category_ood_breakdown"]
    cats = list(pt["9_CMQ_BPI_MCF_48D"].keys())
    std_mcf = [pt["2_Standard_MCF_32D"][c]["centroid_acc"] * 100.0 for c in cats]
    v21_bpi = [pt["7_Full_Frozen_BPI_MCF_32D"][c]["centroid_acc"] * 100.0 for c in cats]
    v22_cmq_c = [pt["9_CMQ_BPI_MCF_48D"][c]["centroid_acc"] * 100.0 for c in cats]
    v22_cmq_r = [pt["9_CMQ_BPI_MCF_48D"][c]["ridge_acc"] * 100.0 for c in cats]

    x = np.arange(len(cats))
    width = 0.21

    fig, ax = plt.subplots(figsize=(11.2, 4.8), dpi=220)
    b1 = ax.bar(x - 1.5 * width, std_mcf, width, label="Standard MCF (32D, Sliding Window)", color="#8c8c8c")
    b2 = ax.bar(x - 0.5 * width, v21_bpi, width, label="V2.1 Frozen BPI-MCF (32D)", color="#377eb8")
    b3 = ax.bar(x + 0.5 * width, v22_cmq_c, width, label="V2.2 CMQ-BPI-MCF (48D, Centroid)", color="#e41a1c")
    b4 = ax.bar(x + 1.5 * width, v22_cmq_r, width, label="V2.2 CMQ-BPI-MCF (48D, Linear Probe)", color="#ff7f00")

    ax.axhline(25.0, color="gray", linestyle="--", linewidth=1.0, label="4-Class Chance (25%)")
    ax.set_ylabel("OOD Direction Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Innovation 1B: 36-Condition Table-Supported OOD Breakdown on pilot_tangent_02", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9.8, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=8.8, frameon=False)

    for bars in (b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.6,
            )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_pt02_7category_ood.png", dpi=220)
    fig.savefig(FIG_DIR / "fig2_pt02_7category_ood.pdf")
    plt.close(fig)


def save_figure(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def decorate(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=14)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", alpha=.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_fig3_online_rl_anticollapse(report):
    res = report["innovations_2_and_3_architecture_and_online_rl"]
    fresh, curves = res["blind_fresh_b0_b1"], res["rl_training_curves"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for key, label, color in [("Surrogate_Residual_RL_No_Aux", "No-Aux", "#bb4430"),
                              ("Surrogate_Residual_RL_With_Aux_Proposed", "With-Aux", "#237a57")]:
        c = curves[key]
        axs[0, 0].plot([r["iter"] for r in c], [r["dev_wrench_r2"] for r in c], "o-", color=color, label=label)
    decorate(axs[0, 0], "(a) Dev wrench reconstruction during refinement", "Wrench R²")
    axs[0, 0].set_xlabel("Surrogate policy-gradient iteration")
    axs[0, 0].legend()
    selected = [("3_V22_DualHead_WarmStart_BC", "WarmStart BC"),
                ("4_V22_Surrogate_Residual_RL_No_Aux", "No-Aux"),
                ("5_V22_Surrogate_Residual_RL_With_Aux_Proposed", "With-Aux")]
    x = np.arange(3)
    colors = ["#537a9c", "#bb4430", "#237a57"]
    for offset, metric, label, color in [(-.18, "clean_action_mse", "Clean", "#537a9c"),
                                       (.18, "occluded_action_mse", "Occluded", "#b89e55")]:
        bars = axs[0, 1].bar(x + offset, [fresh[k][metric] for k, _ in selected], .34, label=label, color=color)
        axs[0, 1].bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    decorate(axs[0, 1], "(b) Fresh b0+b1: action error", "Action MSE (dimensionless)")
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].set_ylim(0, max(fresh[k]["occluded_action_mse"] for k, _ in selected) * 1.3)
    for ax, metric, title, ylabel, factor, fmt in [
        (axs[1, 0], "latent_probe_dir_acc", "(c) Fresh b0+b1: frozen linear probe", "Direction accuracy (%)", 100, "%.2f"),
        (axs[1, 1], "wrench_r2", "(d) Fresh b0+b1: wrench reconstruction", "Wrench R² (unclipped)", 1, "%.3f")]:
        bars = ax.bar(x, [fresh[k][metric] * factor for k, _ in selected], color=colors, width=.6)
        ax.bar_label(bars, fmt=fmt, padding=4, fontsize=9)
        decorate(ax, title, ylabel)
        ax.axhline(0, color="#777777", lw=.7)
        ax.margins(y=.25)
    axs[1, 0].set_ylim(0, 100)
    for ax in [axs[0, 1], axs[1, 0], axs[1, 1]]:
        ax.set_xticks(x, [label for _, label in selected])
    fig.suptitle("Matched residual refinement in a controlled contact surrogate | seed 42", fontsize=14, fontweight="bold")
    save_figure(fig, "fig3_online_rl_anticollapse")


def plot_fig4_isaac_closedloop_and_longhorizon(report):
    a, b = report["part_a_6controller_closedloop"], report["part_b_3stage_long_horizon_benchmark"]
    keys = list(a)
    labels = ["Open-loop", "Stage 3S\n1D grip", "V2.1\n3DoF", "V2.2\nNo-Aux", "V2.2\nBC", "V2.2\nWith-Aux"]
    colors = ["#888888", "#a68b62", "#74669c", "#bb4430", "#537a9c", "#237a57"]
    x = np.arange(len(keys))
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for offset, metric, label, color in [(-.18, "loaded_mean_inhand_slip_mm", "Mean of 4 direction peaks", "#537a9c"),
                                       (.18, "loaded_worst_inhand_slip_mm", "Worst direction peak", "#bb4430")]:
        bars = axs[0, 0].bar(x + offset, [a[k][metric] for k in keys], .34, color=color, label=label)
        axs[0, 0].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    decorate(axs[0, 0], "(a) Part A: in-hand slip, 0.20 N at 15 mm", "Peak slip (mm)")
    axs[0, 0].margins(y=.2)
    axs[0, 0].legend(fontsize=8)
    for offset, metric, label, color in [(-.18, "stage3_mean_socket_radial_error_mm", "Root radial error", "#537a9c"),
                                       (.18, "stage3_mean_socket_tip_lateral_error_mm", "Mesh-tip lateral error", "#237a57")]:
        bars = axs[0, 1].bar(x + offset, [b[k][metric] for k in keys], .34, color=color, label=label)
        axs[0, 1].bar_label(bars, fmt="%.2f", padding=3, fontsize=8, rotation=90)
    decorate(axs[0, 1], "(b) Part B: absolute socket-axis alignment", "Error (mm), final 10-frame mean")
    axs[0, 1].margins(y=.40)
    axs[0, 1].legend(fontsize=8)
    bars = axs[1, 0].bar(x, [a[k]["zero_disturbance_jitter_deg"] for k in keys], color=colors, width=.6)
    axs[1, 0].bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
    axs[1, 0].axhline(.16, ls="--", color="#555555", label="Requested 0.16° criterion")
    decorate(axs[1, 0], "(c) Part A: physical rotation in zero-force trial", "Peak quaternion rotation (degrees)")
    axs[1, 0].set_ylim(0, max(.2, max(a[k]["zero_disturbance_jitter_deg"] for k in keys) * 1.25))
    axs[1, 0].legend(fontsize=8)
    bars = axs[1, 1].bar(x, [b[k]["stage3_mean_socket_axis_tilt_deg"] for k in keys], color=colors, width=.6)
    axs[1, 1].bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
    decorate(axs[1, 1], "(d) Part B: absolute plug/socket axis angle", "Axis angle (degrees), final 10-frame mean")
    axs[1, 1].margins(y=.2)
    for ax in axs.flat:
        ax.set_xticks(x, labels, fontsize=8)
    fig.suptitle("Isaac Lab / PhysX | one fixed scene, four loaded directions | no hardware claim", fontsize=14, fontweight="bold")
    save_figure(fig, "fig4_isaac_closedloop_and_longhorizon")


def main():
    report = json.loads((OUT_DIR / "v2_2_verification_results.json").read_text(encoding="utf-8"))
    cl = json.loads((OUT_DIR / "isaac_closedloop_v2_2_residual_rl_report.json").read_text(encoding="utf-8"))
    plot_fig1_10stage_ablation(report)
    plot_fig2_pt02_7category_ood(report)
    plot_fig3_online_rl_anticollapse(report)
    plot_fig4_isaac_closedloop_and_longhorizon(cl)
    print(f"[OK] Saved four PNG/PDF pairs to {FIG_DIR}")


if __name__ == "__main__":
    main()
