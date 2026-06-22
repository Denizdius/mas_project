#!/usr/bin/env python3
"""Generate IEEE paper figures from comparison_optimized JSON."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "comparison_optimized_20260101_215957.json"
OUT_DIR = ROOT / "idap_2026" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_data():
    with open(JSON_PATH) as f:
        return json.load(f)


def plot_metric_comparison(data):
    sa_agg = data["single_agent"]["aggregated"]
    mas_agg = data["multi_agent_system"]["aggregated"]

    metrics = [
        ("GRC\nGrounding", "grounding_mean", None),
        ("Citation\nPrecision", "citation_precision_mean", "citation_precision_std"),
        ("Citation\nF1", "citation_f1_mean", "citation_f1_std"),
        ("GRC\nCompleteness", "completeness_mean", None),
    ]

    labels = [m[0] for m in metrics]
    x = np.arange(len(labels))
    width = 0.35

    sa_vals, mas_vals = [], []
    sa_err, mas_err = [], []
    for _, mean_key, std_key in metrics:
        sa_vals.append(sa_agg[mean_key])
        mas_vals.append(mas_agg[mean_key])
        sa_err.append(sa_agg[std_key] if std_key else 0)
        mas_err.append(mas_agg[std_key] if std_key else 0)

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(x - width / 2, sa_vals, width, yerr=sa_err, capsize=2, label="Single Agent", color="#4C72B0")
    ax.bar(x + width / 2, mas_vals, width, yerr=mas_err, capsize=2, label="MAS", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(mas_vals + sa_vals) * 1.15)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.savefig(OUT_DIR / "metric_comparison.pdf")
    fig.savefig(OUT_DIR / "metric_comparison.png")
    plt.close(fig)


def plot_grounding_lsa_tradeoff(data):
    sa_samples = data["single_agent"]["samples"]
    mas_samples = data["multi_agent_system"]["samples"]
    mas_by_id = {s["id"]: s for s in mas_samples}

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    for sa in sa_samples:
        sid = sa["id"]
        mas = mas_by_id[sid]
        sa_lsa = sa["metrics"]["lsa_similarity"]
        sa_g = sa["grc_scores"]["grounding"]
        mas_lsa = mas["metrics"]["lsa_similarity"]
        mas_g = mas["stats"]["grounding"]

        ax.annotate(
            "",
            xy=(mas_lsa, mas_g),
            xytext=(sa_lsa, sa_g),
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8, alpha=0.7),
        )
        ax.scatter(sa_lsa, sa_g, color="#4C72B0", s=28, zorder=3)
        ax.scatter(mas_lsa, mas_g, color="#DD8452", s=28, zorder=3)
        if sid == "552153":
            ax.annotate("552153", (mas_lsa, mas_g), textcoords="offset points", xytext=(4, 4), fontsize=6)

    ax.scatter([], [], color="#4C72B0", s=28, label="Single Agent")
    ax.scatter([], [], color="#DD8452", s=28, label="MAS")
    ax.set_xlabel("LSA Similarity")
    ax.set_ylabel("GRC Grounding")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(alpha=0.3, linewidth=0.5)
    fig.savefig(OUT_DIR / "grounding_lsa_tradeoff.pdf")
    fig.savefig(OUT_DIR / "grounding_lsa_tradeoff.png")
    plt.close(fig)


def plot_revision_trajectories(data):
    mas_samples = data["multi_agent_system"]["samples"]

    by_revision = defaultdict(lambda: {"grounding": [], "citation_accuracy": []})
    for sample in mas_samples:
        for entry in sample.get("revision_history", []):
            rev = entry["revision"]
            by_revision[rev]["grounding"].append(entry["grounding"])
            by_revision[rev]["citation_accuracy"].append(entry["citation_accuracy"])

    revisions = sorted(by_revision.keys())
    g_means = [np.mean(by_revision[r]["grounding"]) for r in revisions]
    c_means = [np.mean(by_revision[r]["citation_accuracy"]) for r in revisions]

    representative = ["552153", "568189", "570015"]
    rep_data = {}
    for sample in mas_samples:
        if sample["id"] in representative:
            hist = sample["revision_history"]
            by_rev = defaultdict(list)
            for e in hist:
                by_rev[e["revision"]].append(e["grounding"])
            rep_data[sample["id"]] = {
                r: np.mean(by_rev[r]) for r in sorted(by_rev.keys())
            }

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(revisions, g_means, "o-", color="#DD8452", linewidth=1.5, markersize=4, label="Mean grounding (all)")
    for sid, rev_map in rep_data.items():
        rs = sorted(rev_map.keys())
        ax.plot(rs, [rev_map[r] for r in rs], "--", linewidth=0.9, alpha=0.75, label=f"Sample {sid}")

    ax.set_xlabel("Revision Index")
    ax.set_ylabel("GRC Grounding")
    ax.set_xticks(revisions)
    ax.legend(loc="best", frameon=True, fontsize=6)
    ax.grid(alpha=0.3, linewidth=0.5)
    fig.savefig(OUT_DIR / "revision_trajectories.pdf")
    fig.savefig(OUT_DIR / "revision_trajectories.png")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    plot_metric_comparison(data)
    plot_grounding_lsa_tradeoff(data)
    plot_revision_trajectories(data)
    print(f"Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
