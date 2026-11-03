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
    fresh_trial = [stages[k]["blind_fresh12_trial_label_centroid_acc"] * 100.0 for k in keys]
    fresh_b0b1 = [stages[k]["blind_fresh_b0_b1_step_ramp_centroid_acc"] * 100.0 for k in keys]
    pt02_acc = [stages[k]["pt02_36ood_centroid_acc"] * 100.0 for k in keys]

    x = np.arange(len(short_labels))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13.0, 5.0), dpi=220)
    b1 = ax.bar(x - 1.5 * width, dev_acc, width, label="Dev: pure_loading_01 (c005-c008)", color="#2b5c8f")
    b2 = ax.bar(x - 0.5 * width, fresh_trial, width, label="Blind Held-out: Fresh12 (All 12 Trials)", color="#d95f02")
    b3 = ax.bar(x + 0.5 * width, fresh_b0b1, width, label="Blind Held-out: Fresh (b0+b1 Step+Ramp)", color="#7570b3")
    b4 = ax.bar(x + 1.5 * width, pt02_acc, width, label="OOD: pilot_tangent_02 (36 Table-Supported Conds)", color="#1b9e77")

    ax.axhline(25.0, color="gray", linestyle="--", linewidth=1.0, label="4-Class Chance (25%)")
    ax.axvline(7.5, color="#b2182b", linestyle="-.", linewidth=1.3, label="V2.1 -> V2.2 Upgrade Boundary")
    ax.set_ylabel("Shear Direction Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Innovation 1: 10-Stage Contact Operator Ablation (From Raw State4 & V2.1 BPI-MCF to V2.2 CMQ-BPI-MCF)", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=18, ha="right", fontsize=9.2)
    ax.set_ylim(0, 109)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", ncol=2, fontsize=8.6, framealpha=0.92)

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
    ax.legend(loc="upper left", ncol=2, fontsize=8.8, framealpha=0.92)

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


def plot_fig3_online_rl_anticollapse(v22_report: dict) -> None:
    rl_res = v22_report["innovations_2_and_3_architecture_and_online_rl"]
    fresh_m = rl_res["blind_fresh_b0_b1"]
    curves = rl_res["rl_training_curves"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7), dpi=220)

    # Panel A: Online RL Training Curves (Action MSE & Wrench R2)
    it_no = [pt["iter"] for pt in curves["Online_RL_No_Aux"]]
    mse_no = [pt["dev_clean_action_mse"] for pt in curves["Online_RL_No_Aux"]]
    mse_with = [pt["dev_clean_action_mse"] for pt in curves["Online_RL_With_Aux_Proposed"]]
    r2_no = [pt["dev_wrench_r2"] for pt in curves["Online_RL_No_Aux"]]
    r2_with = [pt["dev_wrench_r2"] for pt in curves["Online_RL_With_Aux_Proposed"]]

    ax1.plot(it_no, mse_no, "o--", color="#d73027", linewidth=2.2, label="Online RL w/o Aux (Collapses: MSE -> 0.274)")
    ax1.plot(it_no, mse_with, "s-", color="#1a9850", linewidth=2.4, label="Online RL + Aux Self-Sup (Ours: MSE -> 0.0045)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Online Residual RL Policy Gradient Iterations", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Dev Active Action MSE (Log Scale, Lower is Better)", fontsize=10.0, fontweight="bold")
    ax1.set_title("(a) Online RL Representation Collapse vs Anti-Collapse Self-Sup", fontsize=10.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="center right", fontsize=8.5, framealpha=0.92)

    # Panel B: Blind Held-Out Fresh (b0+b1) Comparison
    models = [
        ("1_V21_SingleHead_NoProprio", "1. V2.1\nSingle-Head"),
        ("2_V22_DualHead_NoAux_Supervised", "2. V2.2\nNo-Aux Sup"),
        ("3_V22_DualHead_WarmStart_BC", "3. V2.2\nWarmStart BC"),
        ("4_V22_Online_Residual_RL_No_Aux_Collapse", "4. Online RL\nNo-Aux"),
        ("5_V22_Online_Residual_RL_With_Aux_Proposed", "5. Online RL\n+Aux (Ours)"),
    ]
    m_labels = [lbl for _, lbl in models]
    probe_acc = [fresh_m[k]["latent_probe_dir_acc"] * 100.0 for k, _ in models]
    wrn_r2 = [max(-20.0, fresh_m[k]["wrench_r2"] * 100.0) for k, _ in models]

    x = np.arange(len(m_labels))
    width = 0.34
    b1 = ax2.bar(x - 0.5 * width, probe_acc, width, label="Blind Fresh Latent Probe Acc (%)", color="#2b5c8f")
    b2 = ax2.bar(x + 0.5 * width, wrn_r2, width, label="Blind Fresh Wrench R2 (x100, Clipped at -20)", color="#e66101")
    ax2.axhline(0.0, color="black", linewidth=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(m_labels, fontsize=8.8, fontweight="bold")
    ax2.set_ylim(-28, 114)
    ax2.set_ylabel("Score (% / R2 x 100)", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) Blind Held-Out Fresh (b0+b1) Latent Probe & Wrench R2", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8.4, framealpha=0.92)

    for bar in b1:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        val_y = h + 2 if h >= 0 else h - 9
        ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, val_y),
                     ha="center", va="bottom", fontsize=7.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_online_rl_anticollapse.png", dpi=220)
    fig.savefig(FIG_DIR / "fig3_online_rl_anticollapse.pdf")
    plt.close(fig)


def plot_fig4_isaac_closedloop_and_longhorizon(cl_report: dict) -> None:
    part_a = cl_report["part_a_6controller_closedloop"]
    part_b = cl_report["part_b_3stage_long_horizon_benchmark"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0), dpi=220)

    # Panel A: 6-Controller Closed-Loop Benchmark
    order_a = [
        ("1_nominal_open_loop", "1. Open-Loop\n(0D)"),
        ("2_stage3s_1d_grip_only", "2. Stage 3S\n(1D Grip)"),
        ("4_v2_2_online_rl_no_aux_collapse", "4. Online RL\nNo-Aux"),
        ("3_v2_1_pure_bpi_3dof_singlehead", "3. V2.1 BPI\nSingle-Head"),
        ("5_v2_2_warmstart_bc_dualhead", "5. V2.2\nWarmStart BC"),
        ("6_v2_2_online_residual_rl_proposed", "6. V2.2 RL\n+DualHead (Ours)"),
    ]
    labels_a = [lbl for _, lbl in order_a]
    mean_slip = [part_a[k]["loaded_mean_inhand_slip_mm"] for k, _ in order_a]
    worst_slip = [part_a[k]["loaded_worst_inhand_slip_mm"] for k, _ in order_a]
    jitter = [part_a[k]["preload_mean_jitter_mm"] for k, _ in order_a]

    x_a = np.arange(len(labels_a))
    w_a = 0.26
    ba1 = ax1.bar(x_a - w_a, mean_slip, w_a, label="Loaded 4-Dir Mean Slip (mm)", color="#1f78b4")
    ba2 = ax1.bar(x_a, worst_slip, w_a, label="Worst-Direction (+Fx) Slip (mm)", color="#e31a1c")
    ba3 = ax1.bar(x_a + w_a, jitter, w_a, label="Preload Jitter (mm)", color="#33a02c")
    ax1.axhline(2.0, color="#555555", linestyle="--", linewidth=1.0, label="2.0 mm Safety Limit")
    ax1.set_xticks(x_a)
    ax1.set_xticklabels(labels_a, fontsize=8.4, fontweight="bold")
    ax1.set_ylim(0, 24.0)
    ax1.set_ylabel("Displacement / Slip (mm)", fontsize=10.5, fontweight="bold")
    ax1.set_title("(a) Live Isaac Lab 6-DoF Closed-Loop Benchmark (0.20 N @ 15 mm)", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    for bars in (ba1, ba2, ba3):
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2),
                         textcoords="offset points", ha="center", va="bottom", fontsize=7.0)

    # Panel B: 3-Stage Long-Horizon Peg-in-Socket Benchmark
    order_b = [
        ("1_nominal_open_loop", "1. Open-Loop\n(25% Ready)"),
        ("2_stage3s_1d_grip_only", "2. Stage 3S 1D\n(75% Ready)"),
        ("3_v2_1_pure_bpi_3dof_singlehead", "3. V2.1 BPI-MCF\n(75% Ready)"),
        ("6_v2_2_online_residual_rl_proposed", "6. V2.2 Online RL\n(100% 4/4 Ready)"),
    ]
    labels_b = [lbl for _, lbl in order_b]
    s2_slip = [part_b[k]["stage2_mean_transport_slip_mm"] for k, _ in order_b]
    s3_slip = [part_b[k]["stage3_mean_align_slip_mm"] for k, _ in order_b]
    tip_err = [part_b[k]["stage3_mean_final_tip_error_mm"] for k, _ in order_b]
    p_fx_tip = [part_b[k]["per_direction"]["+Fx"]["stage3_final_tip_error_mm"] for k, _ in order_b]

    x_b = np.arange(len(labels_b))
    w_b = 0.20
    bb1 = ax2.bar(x_b - 1.5 * w_b, s2_slip, w_b, label="Stage 2: Transport Mean Slip (mm)", color="#ff7f00")
    bb2 = ax2.bar(x_b - 0.5 * w_b, s3_slip, w_b, label="Stage 3: Reversed-Shear Mean Slip (mm)", color="#1f78b4")
    bb3 = ax2.bar(x_b + 0.5 * w_b, tip_err, w_b, label="Stage 3: 4-Dir Mean Tip Error (mm)", color="#6a3d9a")
    bb4 = ax2.bar(x_b + 1.5 * w_b, p_fx_tip, w_b, label="Stage 3: Worst (+Fx) Tip Error (mm)", color="#e31a1c")
    ax2.axhline(2.60, color="#555555", linestyle="--", linewidth=1.0, label="2.60 mm Socket Chamfer Tolerance")
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(labels_b, fontsize=8.8, fontweight="bold")
    ax2.set_ylim(0, 12.0)
    ax2.set_ylabel("Slip / Tip Alignment Error (mm)", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) Live Isaac Lab 3-Stage Long-Horizon Peg-in-Socket Benchmark", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    for bars in (bb1, bb2, bb3, bb4):
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, min(h, 11.2)), xytext=(0, 2),
                         textcoords="offset points", ha="center", va="bottom", fontsize=7.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_isaac_closedloop_and_longhorizon.png", dpi=220)
    fig.savefig(FIG_DIR / "fig4_isaac_closedloop_and_longhorizon.pdf")
    plt.close(fig)


def main() -> None:
    v22_report = json.loads((OUT_DIR / "v2_2_verification_results.json").read_text(encoding="utf-8"))
    plot_fig1_10stage_ablation(v22_report)
    plot_fig2_pt02_7category_ood(v22_report)
    plot_fig3_online_rl_anticollapse(v22_report)
    cl_path = OUT_DIR / "isaac_closedloop_v2_2_residual_rl_report.json"
    if cl_path.exists():
        cl_report = json.loads(cl_path.read_text(encoding="utf-8"))
        plot_fig4_isaac_closedloop_and_longhorizon(cl_report)
    print(f"[OK] Saved Stage 4 (V2.2) figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
