#!/usr/bin/env python3
"""
Generate publication-grade 300 DPI PNG + PDF figures for V4.1–V4.4 PAEC Scaling & Transfer.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save_fig(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / stem}.png & .pdf")


def plot_fig1_v4_1_rdp(v4_1_dir: Path, fig_dir: Path):
    tasks = [("peel_37d", "peel_w6_si", "Peel (v3)"), ("wipe_37d", "wipe_w6_si", "Wipe (v1.5)"), ("lift_full", "lift_w6_si", "Lift (v2)")]
    methods = [
        ("B0_ProprioOnly", "B0 ProprioOnly", "#6b7280"),
        ("B1_RawFT_End2End", "B1 RawFT E2E", "#9ca3af"),
        ("B2b_MAE_DriftAug", "B2b MAE+Drift", "#e69f00"),
        ("B3b_JEPA_d16_Drift", "B3b JEPA-d16+Drift", "#cc79a7"),
        ("B4_NoBG", "B4 NoBG", "#56b4e9"),
        ("Ours_PAEC_z", "Ours PAEC (z16)", "#0072b2"),
        ("Ours_PAEC_Probe", "Ours PAEC+Probe", "#009e73"),
    ]
    available = []
    for f37, fw6, label in tasks:
        p37 = v4_1_dir / f37 / "v4_paec_results.json"
        pw6 = v4_1_dir / fw6 / "v4_paec_results.json"
        if p37.exists() and pw6.exists():
            available.append((json.loads(p37.read_text(encoding="utf-8")), json.loads(pw6.read_text(encoding="utf-8")), label))
    if not available:
        print("[SKIP] fig1: V4.1 JSONs not found yet.")
        return

    fig, axes = plt.subplots(2, len(available), figsize=(4.6 * len(available), 6.8), sharey=False)
    if len(available) == 1:
        axes = axes[:, None]

    for col, (d37, dw6, tlabel) in enumerate(available):
        for row, (d, sub_title) in enumerate([(d37, f"{tlabel} — Full Modality"), (dw6, f"{tlabel} — 6D Wrench SI")]):
            ax = axes[row, col]
            bc = d["downstream_lowshot_bc"].get("100pct") or list(d["downstream_lowshot_bc"].values())[-1]
            b0 = bc["B0_ProprioOnly"]["test_all_mse_mean"]
            xs = np.arange(len(methods))
            vals = [bc.get(m, {}).get("test_all_mse_mean", np.nan) for m, _, _ in methods]
            drifts = [bc.get(m, {}).get("test_drift_ood_mse_mean", np.nan) for m, _, _ in methods]
            cols = [c for _, _, c in methods]
            w = 0.38
            ax.bar(xs - w / 2, vals, width=w, color=cols, alpha=0.92, label="Clean MSE" if (row == 0 and col == 0) else "")
            ax.bar(xs + w / 2, drifts, width=w, color=cols, alpha=0.45, hatch="//", edgecolor="black", linewidth=0.5, label="Drift OOD (+1.5σ)" if (row == 0 and col == 0) else "")
            ax.axhline(b0, color="#6b7280", linestyle="--", linewidth=1.0)
            ax.set_xticks(xs)
            ax.set_xticklabels([lbl for _, lbl, _ in methods], rotation=28, ha="right", fontsize=8)
            ax.set_title(sub_title)
            if col == 0:
                ax.set_ylabel("Action-Chunk Test MSE (↓)")
            finite = [v for v in vals + drifts if np.isfinite(v)]
            if finite:
                lo, hi = min(finite), max(finite)
                pad = max((hi - lo) * 0.25, 0.01)
                ax.set_ylim(max(0.0, lo - pad * 0.5), hi + pad)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("V4.1 Audited Fair Baselines Across 3 RDP Tasks (Full Modality vs. 6D Wrench SI)", y=1.06, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig1_v4_1_rdp_fair_benchmark")


def plot_fig2_data_scaling(scaling: Dict[str, Any], fig_dir: Path):
    exp_a = scaling.get("exp_a_data_scaling", {})
    if not exp_a:
        return
    keys = sorted(exp_a.keys(), key=lambda k: exp_a[k]["n_episodes"])
    ns = [exp_a[k]["n_episodes"] for k in keys]

    models = [
        ("PAEC_Tiny_d16", "Ours PAEC (b+g, d16)", "#0072b2", "-o"),
        ("B4_NoBG_Tiny_d16", "B4 NoBG (d16)", "#56b4e9", "--s"),
        ("B2b_MAE_Tiny_d16", "B2b MAE+Drift (d16)", "#e69f00", "-.^"),
        ("B3b_JEPA_Tiny_d16", "B3b JEPA+VICReg (d16)", "#cc79a7", ":d"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    # Panel 1: Zero-Shot Wrench R2 on Unseen RH20T Tasks & Unseen Embodiment
    ax = axes[0]
    for arch, lbl, col, fmt in [("PAEC_Tiny_d16", "PAEC Unseen-Task", "#0072b2", "-o"), ("B4_NoBG_Tiny_d16", "NoBG Unseen-Task", "#56b4e9", "--s")]:
        ys = [exp_a[k]["models"][arch]["rh20t_heldout_task"]["geom"].get("dw3_clean_r2", np.nan) for k in keys]
        ax.plot(ns, ys, fmt, color=col, linewidth=2.0, markersize=6, label=lbl)
    for arch, lbl, col, fmt in [("PAEC_Tiny_d16", "PAEC Unseen-Robot", "#009e73", "-^"), ("B4_NoBG_Tiny_d16", "NoBG Unseen-Robot", "#66c2a5", "--v")]:
        ys = [exp_a[k]["models"][arch]["rh20t_heldout_cfg"]["geom"].get("dw3_clean_r2", np.nan) for k in keys]
        ax.plot(ns, ys, fmt, color=col, linewidth=2.0, markersize=6, label=lbl)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("RH20T Pretraining Episodes (log scale)")
    ax.set_ylabel("Zero-Shot 3-Step Wrench $R^2$ (↑)")
    ax.set_title("(a) Physics Prediction Scaling (Held-Out)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 2: Downstream Low-Shot BC MSE on RH20T Unseen-Task (adaptive budget key)
    ax = axes[1]
    b0_task_dict = scaling["anchors"]["b0"]["rh20t_heldout_task"]
    task_bkey = sorted(b0_task_dict.keys(), key=lambda x: int(x.replace("eps", "")))[0]
    b0_task = b0_task_dict[task_bkey]["test_all_mse_mean"]
    ax.axhline(b0_task, color="#6b7280", linestyle="--", linewidth=1.2, label="B0 ProprioOnly")
    for arch, lbl, col, fmt in models:
        ys = [exp_a[k]["models"][arch]["rh20t_heldout_task"]["bc"][task_bkey]["test_all_mse_mean"] for k in keys]
        ax.plot(ns, ys, fmt, color=col, linewidth=2.0, markersize=6, label=lbl)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("RH20T Pretraining Episodes (log scale)")
    ax.set_ylabel(f"{task_bkey} BC Test MSE (↓)")
    ax.set_title(f"(b) RH20T Unseen-Task ({task_bkey} BC)")
    ax.legend(frameon=True, framealpha=0.88, edgecolor="none", fontsize=7.8, loc="lower left")
    ax.grid(alpha=0.25)

    # Panel 3: Cross-Dataset Zero-Shot Mean BC MSE across all 3 RDP tasks (10-shot)
    ax = axes[2]
    rdp_keys = ["rdp_peel", "rdp_wipe", "rdp_lift"]
    b0_rdp = np.mean([scaling["anchors"]["b0"][rk]["10eps"]["test_all_mse_mean"] for rk in rdp_keys])
    id_rdp = np.mean([scaling["anchors"]["indomain_rdp_paec"][rk]["bc"]["10eps"]["test_all_mse_mean"] for rk in rdp_keys])
    ax.axhline(b0_rdp, color="#6b7280", linestyle="--", linewidth=1.2, label="RDP B0 ProprioOnly")
    ax.axhline(id_rdp, color="#111827", linestyle=":", linewidth=1.4, label="RDP In-Domain PAEC")
    for arch, lbl, col, fmt in models:
        ys = [
            float(np.mean([exp_a[k]["models"][arch][rk]["bc"]["10eps"]["test_all_mse_mean"] for rk in rdp_keys]))
            for k in keys
        ]
        ax.plot(ns, ys, fmt, color=col, linewidth=2.0, markersize=6, label=lbl)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("RH20T Pretraining Episodes (log scale)")
    ax.set_ylabel("Mean 10-Shot BC MSE on RDP (↓)")
    ax.set_title("(c) Cross-Dataset RH20T → RDP (10-Shot)")
    ax.legend(frameon=True, framealpha=0.88, edgecolor="none", fontsize=7.5, loc="upper left")
    ax.set_ylim(top=0.2112)
    ax.grid(alpha=0.25)

    fig.suptitle("V4.3 Data Scaling Law on RH20T (PAEC-Tiny 96/2L/d16, Canonical 6D SI)", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig2_rh20t_data_scaling_law")


def plot_fig3_capacity_bottleneck(scaling: Dict[str, Any], fig_dir: Path):
    exp_b = scaling.get("exp_b_capacity_bottleneck", {})
    if not exp_b:
        return

    rdp_keys = ["rdp_peel", "rdp_wipe", "rdp_lift"]
    cap_archs = [("PAEC_Tiny_d16", "Tiny\n(96,2L)"), ("PAEC_Small_d16", "Small\n(256,2L)"), ("PAEC_Base_d16", "Base\n(512,4L)")]
    bot_archs = [("PAEC_Small_d8", "d=8"), ("PAEC_Small_d16", "d=16"), ("PAEC_Small_d32", "d=32"), ("PAEC_Small_d64", "d=64")]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    # Panel 1: Capacity sweep (d_z=16) on Unseen-Task / Unseen-Robot R2 & RDP MSE
    ax = axes[0]
    cap_present = [(a, l) for a, l in cap_archs if a in exp_b]
    xs = np.arange(len(cap_present))
    r2_task = [exp_b[a]["rh20t_heldout_task"]["geom"].get("dw3_clean_r2", np.nan) for a, _ in cap_present]
    r2_cfg = [exp_b[a]["rh20t_heldout_cfg"]["geom"].get("dw3_clean_r2", np.nan) for a, _ in cap_present]
    w = 0.35
    ax.bar(xs - w / 2, r2_task, width=w, color="#0072b2", label="Unseen-Task $R^2$")
    ax.bar(xs + w / 2, r2_cfg, width=w, color="#009e73", label="Unseen-Robot $R^2$")
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in cap_present])
    ax.set_ylabel("Zero-Shot 3-Step Wrench $R^2$ (↑)")
    ax.set_title("(a) Model Capacity Scaling (Fixed $d_z=16$)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    # Panel 2: Bottleneck sweep (Small 256/2L) - Effective Rank & Unseen-Task R2
    ax = axes[1]
    bot_present = [(a, l) for a, l in bot_archs if a in exp_b]
    xs2 = np.arange(len(bot_present))
    eff_r = [exp_b[a]["rh20t_heldout_task"]["geom"]["test_eff_rank"] for a, _ in bot_present]
    r2_b = [exp_b[a]["rh20t_heldout_task"]["geom"].get("dw3_clean_r2", np.nan) for a, _ in bot_present]
    ax2 = ax.twinx()
    b_h = ax.bar(xs2 - 0.18, eff_r, width=0.36, color="#56b4e9", label="Effective Rank")
    l_h = ax2.plot(xs2 + 0.18, r2_b, "-o", color="#d55e00", linewidth=2.0, label="Unseen-Task $R^2$")
    ax.set_xticks(xs2)
    ax.set_xticklabels([l for _, l in bot_present])
    ax.set_ylabel("Latent Effective Rank")
    ax2.set_ylabel("Unseen-Task Wrench $R^2$ (↑)")
    ax.set_title("(b) Bottleneck $d_z$ Sweep (Small 256/2L)")
    ax.legend([b_h, l_h[0]], ["Effective Rank", "Unseen-Task $R^2$"], frameon=True, framealpha=0.88, edgecolor="none", loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # Panel 3: Cross-Dataset RDP 10-shot MSE across Bottleneck & Capacity
    ax = axes[2]
    all_cmp = cap_present + [(a, f"Small\n{l}") for a, l in bot_present if a != "PAEC_Small_d16"]
    xs3 = np.arange(len(all_cmp))
    rdp_mse = [float(np.mean([exp_b[a][rk]["bc"]["10eps"]["test_all_mse_mean"] for rk in rdp_keys])) for a, _ in all_cmp]
    b0_rdp = float(np.mean([scaling["anchors"]["b0"][rk]["10eps"]["test_all_mse_mean"] for rk in rdp_keys]))
    ax.bar(xs3, rdp_mse, color=["#0072b2" if "d16" in a else "#009e73" for a, _ in all_cmp], alpha=0.9)
    ax.axhline(b0_rdp, color="#6b7280", linestyle="--", linewidth=1.2, label="RDP B0 ProprioOnly")
    ax.set_xticks(xs3)
    ax.set_xticklabels([l for _, l in all_cmp], fontsize=8)
    ax.set_ylabel("Mean RDP 10-Shot BC MSE (↓)")
    ax.set_title("(c) Cross-Dataset Transfer vs. Capacity/$d_z$")
    ax.legend(frameon=False)
    lo, hi = min(rdp_mse + [b0_rdp]), max(rdp_mse + [b0_rdp])
    ax.set_ylim(max(0.0, lo - 0.3 * (hi - lo)), hi + 0.3 * (hi - lo))
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("V4.4 Capacity & Bottleneck Scaling at Maximum RH20T Scale", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig3_capacity_and_bottleneck_scaling")


def plot_fig4_rh20t_to_rdp_fewshot(scaling: Dict[str, Any], fig_dir: Path):
    exp_b = scaling.get("exp_b_capacity_bottleneck", {})
    exp_a = scaling.get("exp_a_data_scaling", {})
    if not exp_a:
        return
    n_max_key = sorted(exp_a.keys(), key=lambda k: exp_a[k]["n_episodes"])[-1]
    src_models = exp_a[n_max_key]["models"]
    paec_d16 = exp_b.get("PAEC_Small_d16", src_models["PAEC_Tiny_d16"])
    paec_d32 = exp_b.get("PAEC_Small_d32", paec_d16)
    mae_mod = exp_b.get("B2b_MAE_Small_d16", src_models["B2b_MAE_Tiny_d16"])
    jepa_mod = exp_b.get("B3b_JEPA_Small_d16", src_models["B3b_JEPA_Tiny_d16"])

    rdp_tasks = [("rdp_peel", "RDP Peel (v3)"), ("rdp_wipe", "RDP Wipe (v1.5)"), ("rdp_lift", "RDP Lift (v2)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for col, (rk, title) in enumerate(rdp_tasks):
        ax = axes[col]
        b0 = scaling["anchors"]["b0"][rk]
        id_paec = scaling["anchors"]["indomain_rdp_paec"][rk]["bc"]
        budgets = sorted(b0.keys(), key=lambda x: int(x.replace("eps", "")))
        xs = np.arange(len(budgets))
        ax.plot(xs, [b0[b]["test_all_mse_mean"] for b in budgets], "--o", color="#6b7280", label="B0 ProprioOnly")
        ax.plot(xs, [id_paec[b]["test_all_mse_mean"] for b in budgets], ":s", color="#111827", label="In-Domain PAEC (d16)")
        ax.plot(xs, [mae_mod[rk]["bc"][b]["test_all_mse_mean"] for b in budgets], "-.^", color="#e69f00", label="RH20T MAE+Drift (Small)")
        ax.plot(xs, [jepa_mod[rk]["bc"][b]["test_all_mse_mean"] for b in budgets], ":d", color="#cc79a7", label="RH20T JEPA+VICReg (Small)")
        ax.plot(xs, [paec_d16[rk]["bc"][b]["test_all_mse_mean"] for b in budgets], "-o", color="#56b4e9", linewidth=1.8, label="RH20T PAEC-Small (d16)")
        ax.plot(xs, [paec_d32[rk]["bc"][b]["test_all_mse_mean"] for b in budgets], "-*", color="#0072b2", linewidth=2.3, markersize=8, label="RH20T PAEC-Small (d32)")
        ax.set_xticks(xs)
        ax.set_xticklabels(budgets)
        ax.set_xlabel("Target RDP Training Episodes")
        if col == 0:
            ax.set_ylabel("Action-Chunk Test MSE (↓)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        if col == 2:
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Cross-Dataset Zero-Shot Encoder Transfer: RH20T → Unseen RDP Tasks (5-Shot to 100%)", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig4_rh20t_to_rdp_fewshot_transfer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-1-dir", type=str, default="experiments/v4_paec_contact_compression/outputs/v4_1_rdp")
    parser.add_argument("--scaling-json", type=str, default="experiments/v4_paec_contact_compression/outputs/v4_scaling_20261124/v4_scaling_results.json")
    parser.add_argument("--fig-dir", type=str, default="experiments/v4_paec_contact_compression/outputs/v4_scaling_20261124/figures")
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    plot_fig1_v4_1_rdp(Path(args.v4_1_dir), fig_dir)
    s_path = Path(args.scaling_json)
    if s_path.exists():
        scaling = json.loads(s_path.read_text(encoding="utf-8"))
        plot_fig2_data_scaling(scaling, fig_dir)
        plot_fig3_capacity_bottleneck(scaling, fig_dir)
        plot_fig4_rh20t_to_rdp_fewshot(scaling, fig_dir)


if __name__ == "__main__":
    main()
