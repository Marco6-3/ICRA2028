#!/usr/bin/env python3
"""Generate publication-quality 300 DPI 2x2 summary figure for V4.7 3-Task ACT Suite."""

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
        default="experiments/v4_paec_contact_compression/outputs/v4_7_act_defect_suite/v4_7_results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/v4_paec_contact_compression/outputs/v4_7_act_defect_suite",
    )
    parser.add_argument("--budget-key", type=str, default="15demos")
    args = parser.parse_args()

    res = json.load(open(args.results_json, encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bkey = args.budget_key

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.2))
    regimes = [
        ("clean", "Clean", "#0072b2"),
        ("thermal_drift_0p5", "+0.5σ Thermal Drift", "#e69f00"),
        ("extreme_drift_1p5", "+1.5σ Extreme Drift", "#d55e00"),
    ]
    methods = [
        ("Classical_Baseline", "Classical\nBaseline"),
        ("ACT_RGB_Proprio", "ACT\nRGB+Prop"),
        ("ACT_RGB_RawFT", "ACT\n+RawFT"),
        ("ACT_RGB_MAE", "ACT\n+MAE"),
        ("ACT_RGB_JEPA", "ACT\n+JEPA"),
        ("ACT_RGB_PAEC", "ACT\n+PAEC\n(Ours)"),
    ]
    task_panels = [
        (axes[0, 0], "battery_scan", "(a) Task 1: EV Battery Scan (≤ 11.5 N)"),
        (axes[0, 1], "auto_connector", "(b) Task 2: Auto Connector Jam (≤ 15.0 N)"),
        (axes[1, 0], "semi_grasp", "(c) Task 3: Semi Carrier Transfer (≤ 14.0 N)"),
    ]

    x = np.arange(len(methods))
    width = 0.25
    offsets = [-width, 0.0, width]

    for ax, tname, title in task_panels:
        tdata = res["tasks"][tname]
        for idx, (rk, rlbl, col) in enumerate(regimes):
            vals = []
            viols = []
            for mk, _ in methods:
                d = tdata["classical_baseline"][rk] if mk == "Classical_Baseline" else tdata["act_methods"][mk][bkey][rk]
                vals.append(d["success_rate_mean"])
                viols.append(d["safety_viol_rate_mean"])
            bars = ax.bar(x + offsets[idx], vals, width=width, color=col, alpha=0.92, label=rlbl)
            for br, v, vio in zip(bars, vals, viols):
                txt = f"{v:.0f}" if vio < 0.5 else f"{v:.0f}\n({vio:.0f}!)"
                ax.text(br.get_x() + br.get_width() / 2.0, v + 1.5, txt, ha="center", va="bottom", fontsize=6.4)
        ax.set_xticks(x)
        ax.set_xticklabels([ml for _, ml in methods], fontsize=8.0)
        ax.set_ylim(0, 126)
        ax.set_ylabel("Closed-Loop Success Rate (%) ↑")
        ax.set_title(title)
        ax.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.6, loc="upper left", ncol=3)
        ax.grid(axis="y", alpha=0.25)

    # Panel (d): Online Defect-Detection AUROC (Task 1 Battery & Task 2 Connector) + Latency Inset
    ax = axes[1, 1]
    t1_paec = res["tasks"]["battery_scan"]["act_methods"]["ACT_RGB_PAEC"][bkey]
    t2_paec = res["tasks"]["auto_connector"]["act_methods"]["ACT_RGB_PAEC"][bkey]
    auroc_groups = [
        (
            "Battery\n(Clean)",
            t1_paec["clean"].get("raw_norm_max_auroc_mean", 0.0),
            t1_paec["clean"].get("raw_rate_max_auroc_mean", 0.0),
            t1_paec["clean"].get("paec_rel_surprise_auroc_mean", 0.0),
            t1_paec["clean"].get("paec_stiff_contrast_auroc_mean", 0.0),
        ),
        (
            "Battery\n(+0.5σ)",
            t1_paec["thermal_drift_0p5"].get("raw_norm_max_auroc_mean", 0.0),
            t1_paec["thermal_drift_0p5"].get("raw_rate_max_auroc_mean", 0.0),
            t1_paec["thermal_drift_0p5"].get("paec_rel_surprise_auroc_mean", 0.0),
            t1_paec["thermal_drift_0p5"].get("paec_stiff_contrast_auroc_mean", 0.0),
        ),
        (
            "Battery\n(+1.5σ)",
            t1_paec["extreme_drift_1p5"].get("raw_norm_max_auroc_mean", 0.0),
            t1_paec["extreme_drift_1p5"].get("raw_rate_max_auroc_mean", 0.0),
            t1_paec["extreme_drift_1p5"].get("paec_rel_surprise_auroc_mean", 0.0),
            t1_paec["extreme_drift_1p5"].get("paec_stiff_contrast_auroc_mean", 0.0),
        ),
        (
            "Connector\n(Clean)",
            t2_paec["clean"].get("raw_norm_max_auroc_mean", 0.0),
            t2_paec["clean"].get("raw_rate_max_auroc_mean", 0.0),
            t2_paec["clean"].get("paec_rel_surprise_auroc_mean", 0.0),
            t2_paec["clean"].get("paec_stiff_contrast_auroc_mean", 0.0),
        ),
        (
            "Connector\n(+0.5σ)",
            t2_paec["thermal_drift_0p5"].get("raw_norm_max_auroc_mean", 0.0),
            t2_paec["thermal_drift_0p5"].get("raw_rate_max_auroc_mean", 0.0),
            t2_paec["thermal_drift_0p5"].get("paec_rel_surprise_auroc_mean", 0.0),
            t2_paec["thermal_drift_0p5"].get("paec_stiff_contrast_auroc_mean", 0.0),
        ),
    ]
    xg = np.arange(len(auroc_groups))
    w4 = 0.19
    offs4 = [-1.5 * w4, -0.5 * w4, 0.5 * w4, 1.5 * w4]
    detectors = [
        ("Raw Norm Max", "#9ca3af", 1),
        ("Raw Rate Max", "#e69f00", 2),
        ("PAEC Rel Surprise", "#0072b2", 3),
        ("PAEC Stiff Contrast", "#009e73", 4),
    ]
    for (dlbl, dcol, didx), off in zip(detectors, offs4):
        vals = [g[didx] for g in auroc_groups]
        bars = ax.bar(xg + off, vals, width=w4, color=dcol, alpha=0.92, label=dlbl)
        for br, v in zip(bars, vals):
            ax.text(br.get_x() + br.get_width() / 2.0, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3)

    lat = res["edge_latency"]
    lat_txt = (
        f"Batch=1 (ResNet18 + O(1) PAEC + ACT):\n"
        f"  GPU: P50={lat['gpu']['p50_ms']:.2f} ms, P95={lat['gpu']['p95_ms']:.2f} ms ({lat['gpu']['max_hz']:.0f} Hz)\n"
        f"  CPU: P50={lat['cpu']['p50_ms']:.2f} ms, P95={lat['cpu']['p95_ms']:.2f} ms ({lat['cpu']['max_hz']:.0f} Hz)"
    )
    ax.text(
        0.02, 0.98, lat_txt, transform=ax.transAxes, va="top", ha="left",
        fontsize=6.9, bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95),
    )
    ax.axhline(0.5, color="#9ca3af", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xticks(xg)
    ax.set_xticklabels([g[0] for g in auroc_groups], fontsize=8.0)
    ax.set_ylim(0, 1.28)
    ax.set_ylabel("Online Defect-Detection AUROC ↑")
    ax.set_title("(d) Closed-Loop Defect AUROC & Single-Step Latency")
    ax.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.0, loc="lower left", ncol=2)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "V4.7 Industrial 3-Task Visual-Tactile ACT Suite (15 Demos × 3 Seeds × 12 Rollouts, Pretrained Frozen RH20T Encoders)",
        y=1.01,
        fontsize=12.0,
        fontweight="bold",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig7_v4_7_act_defect_suite.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig7_v4_7_act_defect_suite.png & .pdf to", out_dir)


if __name__ == "__main__":
    main()
