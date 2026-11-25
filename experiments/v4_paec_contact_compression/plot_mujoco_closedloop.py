#!/usr/bin/env python3
"""Generate publication-quality 300 DPI summary figure for V4.5 MuJoCo closed-loop benchmark."""

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
        default="experiments/v4_paec_contact_compression/outputs/v4_5_mujoco_closedloop/mujoco_closedloop_results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/v4_paec_contact_compression/outputs/v4_5_mujoco_closedloop",
    )
    args = parser.parse_args()

    res = json.load(open(args.results_json, encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = res["methods"]

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    budgets = ["5demos", "10demos", "20demos"]
    bx = np.array([5, 10, 20])
    specs = [
        ("PAEC_Small_d32", "PAEC-Small d32 (Ours)", "#0072b2", "-o", 2.4),
        ("PAEC_Base_d32", "PAEC-Base d32 (Ours)", "#009e73", "--s", 2.0),
        ("B2b_MAE_Small_d16", "B2b MAE+Drift (Small)", "#e69f00", "-.^", 1.9),
        ("B3b_JEPA_Small_d16", "B3b JEPA+VICReg (Small)", "#cc79a7", ":D", 1.8),
        ("B1b_RawWrench6D_E2E", "B1b Raw 6D F/T E2E", "#d55e00", "--v", 1.8),
        ("B0_ProprioOnly", "B0 ProprioOnly", "#6b7280", "--x", 1.6),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))

    # Panel (a): Clean Closed-Loop Success Rate
    ax = axes[0]
    for key, lbl, col, fmt, lw in specs:
        ys = [methods[key][b]["clean"]["success_rate_mean"] for b in budgets]
        ax.plot(bx, ys, fmt, color=col, linewidth=lw, markersize=6, label=lbl)
    ax.set_xticks(bx)
    ax.set_xticklabels(["5", "10", "20"])
    ax.set_ylim(-5, 114)
    ax.set_xlabel("MuJoCo Target Demonstrations")
    ax.set_ylabel("Closed-Loop Success Rate (%) ↑")
    ax.set_title("(a) Clean Closed-Loop Wiping & Snag")
    ax.legend(frameon=True, framealpha=0.90, edgecolor="none", fontsize=7.6, loc="lower left")
    ax.grid(alpha=0.25)

    # Panel (b): +1.5 sigma Drift OOD Closed-Loop Success Rate
    ax = axes[1]
    for key, lbl, col, fmt, lw in specs:
        ys = [methods[key][b]["drift_ood"]["success_rate_mean"] for b in budgets]
        ax.plot(bx, ys, fmt, color=col, linewidth=lw, markersize=6, label=lbl)
    ax.set_xticks(bx)
    ax.set_xticklabels(["5", "10", "20"])
    ax.set_ylim(-5, 114)
    ax.set_xlabel("MuJoCo Target Demonstrations")
    ax.set_ylabel("Closed-Loop Success Rate (%) ↑")
    ax.set_title("(b) +1.5σ Wrench Zero-Drift OOD")
    ax.legend(frameon=True, framealpha=0.90, edgecolor="none", fontsize=7.6, loc="lower left")
    ax.grid(alpha=0.25)

    # Panel (c): Mean Drift-OOD Success vs Normal Force RMSE & Zero-Shot Surprise AUROC
    ax = axes[2]
    bar_keys = [
        ("B0_ProprioOnly", "B0\nProp"),
        ("B1b_RawWrench6D_E2E", "B1b\nRawFT"),
        ("B3b_JEPA_Small_d16", "B3b\nJEPA"),
        ("B2b_MAE_Small_d16", "B2b\nMAE"),
        ("PAEC_Tiny_d16", "PAEC\nTiny"),
        ("PAEC_Small_d32", "PAEC\nSmall"),
        ("PAEC_Base_d32", "PAEC\nBase"),
    ]
    xs = np.arange(len(bar_keys))
    mean_dr_succ = [float(np.mean([methods[k][b]["drift_ood"]["success_rate_mean"] for b in budgets])) for k, _ in bar_keys]
    mean_dr_rmse = [float(np.mean([methods[k][b]["drift_ood"]["fn_rmse_n_mean"] for b in budgets])) for k, _ in bar_keys]
    cols = ["#6b7280", "#d55e00", "#cc79a7", "#e69f00", "#56b4e9", "#0072b2", "#009e73"]
    bars = ax.bar(xs - 0.17, mean_dr_succ, width=0.34, color=cols, alpha=0.92, label="Mean Drift Success (%) ↑")
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    line_rmse = ax2.plot(xs + 0.17, mean_dr_rmse, "-o", color="#111827", linewidth=1.8, markersize=5.5, label="Drift Force RMSE (N) ↓")
    for b_rect, v in zip(bars, mean_dr_succ):
        ax.text(b_rect.get_x() + b_rect.get_width() / 2.0, v + 2.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in bar_keys], fontsize=8.0)
    ax.set_ylim(0, 122)
    ax2.set_ylim(1.2, 4.6)
    ax.set_ylabel("Mean Drift OOD Success (%) ↑")
    ax2.set_ylabel("Mean Force RMSE (N) ↓")
    ax.set_title("(c) All-Budget Mean Drift Succ vs. Force RMSE")
    ax.legend([bars[0], line_rmse[0]], ["Mean Drift Succ (%) ↑", "Force RMSE (N) ↓"], frameon=True, framealpha=0.90, edgecolor="none", fontsize=7.6, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    n_tr = res.get("rh20t_ckpt_n_train_episodes", 6289)
    fig.suptitle(
        f"V4.5 MuJoCo Closed-Loop Sim-to-Real / Zero-Shot Frozen Transfer (Pretrained on {n_tr} RH20T Real Episodes, 3 Seeds × 24 Rollouts)",
        y=1.03,
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig5_mujoco_closedloop_and_drift.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5_mujoco_closedloop_and_drift.png & .pdf to", out_dir)


if __name__ == "__main__":
    main()
