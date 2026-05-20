"""
Generate presentation-quality figures for Single RLAIF vs Ensemble RLAIF results.

Usage:
    PYTHONPATH=. python eval/presentation_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

EVAL_DIR = Path("eval")
FIG_DIR = Path("figures")

# ── colour palette ──────────────────────────────────────────────────────────
BASE_BLUE   = "#3b82f6"   # Qwen base
SFT_TEAL    = "#14b8a6"   # SFT
DPO_PURPLE  = "#8b5cf6"   # DPO
SINGLE_RED  = "#ef4444"   # RLAIF-single
ENSEMB_GRN  = "#22c55e"   # RLAIF-ensemble
BASELINE    = "#94a3b8"   # 0.5 line colour


def _load(fname: str) -> dict:
    return json.loads((EVAL_DIR / fname).read_text())["summary"]


def fig_pipeline_progression() -> None:
    """
    Horizontal bar chart showing the win-rate of each model vs the preceding
    step, telling the training-pipeline story left-to-right.
    """
    stages = [
        {"label": "SFT\nvs Base",          "mean": 0.60, "lo": 0.42, "hi": 0.78, "color": SFT_TEAL},
        {"label": "DPO\nvs SFT",           "mean": 0.51, "lo": 0.33, "hi": 0.70, "color": DPO_PURPLE},
        {"label": "RLAIF-single\nvs DPO",  "mean": 0.36, "lo": 0.20, "hi": 0.55, "color": SINGLE_RED},
        {"label": "RLAIF-ensemble\nvs DPO","mean": 0.56, "lo": 0.39, "hi": 0.74, "color": ENSEMB_GRN},
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(stages))

    for i, s in enumerate(stages):
        err_lo = s["mean"] - s["lo"]
        err_hi = s["hi"]  - s["mean"]
        ax.bar(i, s["mean"], color=s["color"], edgecolor="black",
               linewidth=0.8, width=0.55, zorder=3)
        ax.errorbar(i, s["mean"], yerr=[[err_lo], [err_hi]],
                    fmt="none", color="black", capsize=6, linewidth=1.5, zorder=4)
        ax.text(i, s["hi"] + 0.035, f"{s['mean']:.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(0.5, color=BASELINE, linestyle="--", linewidth=1.5,
               label="50 % (no improvement)", zorder=2)

    ax.set_xticks(xs)
    ax.set_xticklabels([s["label"] for s in stages], fontsize=10)
    ax.set_ylabel("Win-rate of A (95 % CI)", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Training-pipeline progression — pairwise win-rates\n"
                 "Judge: gpt-oss:120b-cloud · 20 prompts × 2 orderings",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = FIG_DIR / "fig_pipeline_progression.png"
    fig.savefig(out, dpi=180)
    print(f"  saved: {out}")
    plt.close(fig)


def fig_single_vs_ensemble() -> None:
    """
    Two-panel figure:
      Left  — RLAIF-single vs DPO  and  RLAIF-ensemble vs DPO
      Right — RLAIF-ensemble vs RLAIF-single (the head-to-head)
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 5),
                                             gridspec_kw={"width_ratios": [2, 1]})

    # ── Left panel: both vs DPO ────────────────────────────────────────────
    vs_dpo = [
        {"label": "RLAIF-single\nvs DPO",   "mean": 0.36, "lo": 0.20, "hi": 0.55, "color": SINGLE_RED},
        {"label": "RLAIF-ensemble\nvs DPO", "mean": 0.56, "lo": 0.39, "hi": 0.74, "color": ENSEMB_GRN},
    ]
    for i, s in enumerate(vs_dpo):
        err = [[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]]
        ax_left.bar(i, s["mean"], color=s["color"], edgecolor="black",
                    linewidth=0.8, width=0.45, zorder=3)
        ax_left.errorbar(i, s["mean"], yerr=err,
                         fmt="none", color="black", capsize=7, linewidth=1.8, zorder=4)
        ax_left.text(i, s["hi"] + 0.04, f"{s['mean']:.2f}",
                     ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax_left.axhline(0.5, color=BASELINE, linestyle="--", linewidth=1.5, label="50 % baseline")
    ax_left.set_xticks([0, 1])
    ax_left.set_xticklabels([s["label"] for s in vs_dpo], fontsize=11)
    ax_left.set_ylabel("Win-rate of A (95 % CI)", fontsize=11)
    ax_left.set_ylim(0, 1.0)
    ax_left.set_title("Both RLAIF variants vs DPO baseline", fontsize=11)
    ax_left.legend(fontsize=9, loc="upper right")
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    # Annotate the degradation / recovery arrows
    ax_left.annotate("", xy=(0, 0.36), xytext=(0, 0.5),
                     arrowprops=dict(arrowstyle="->", color=SINGLE_RED, lw=2))
    ax_left.text(0.18, 0.43, "⚠ hurts", color=SINGLE_RED, fontsize=9, ha="left")

    ax_left.annotate("", xy=(1, 0.56), xytext=(1, 0.5),
                     arrowprops=dict(arrowstyle="->", color=ENSEMB_GRN, lw=2))
    ax_left.text(1.05, 0.53, "recovers", color=ENSEMB_GRN, fontsize=9, ha="left")

    # ── Right panel: head-to-head ─────────────────────────────────────────
    mean, lo, hi = 0.65, 0.49, 0.81
    err = [[mean - lo], [hi - mean]]
    ax_right.bar(0, mean, color=ENSEMB_GRN, edgecolor="black",
                 linewidth=0.8, width=0.45, zorder=3)
    ax_right.errorbar(0, mean, yerr=err,
                      fmt="none", color="black", capsize=7, linewidth=1.8, zorder=4)
    ax_right.text(0, hi + 0.04, f"{mean:.2f}",
                  ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax_right.axhline(0.5, color=BASELINE, linestyle="--", linewidth=1.5)
    ax_right.set_xticks([0])
    ax_right.set_xticklabels(["RLAIF-ensemble\nvs RLAIF-single"], fontsize=11)
    ax_right.set_ylim(0, 1.0)
    ax_right.set_title("Head-to-head:\nensemble vs single", fontsize=11)
    ax_right.set_ylabel("")
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    # decisive counts annotation
    ax_right.text(0, 0.08,
                  "Decisive:\n10 ensemble / 2 single",
                  ha="center", fontsize=9, color="#374151",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d1d5db"))

    fig.suptitle("Ensemble RLAIF (4-Critic) vs Single-Critic RLAIF\n"
                 "Judge: gpt-oss:120b-cloud · 20 prompts × 2 orderings",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = FIG_DIR / "fig_single_vs_ensemble.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  saved: {out}")
    plt.close(fig)


def fig_summary_table() -> None:
    """Clean table figure — all 5 comparisons at a glance."""
    rows = [
        ("SFT vs Qwen-0.5B-base",         "0.60", "[0.42, 0.78]", "8 / 4 / 8",   SFT_TEAL),
        ("DPO vs SFT",                     "0.51", "[0.33, 0.70]", "7 / 7 / 6",   DPO_PURPLE),
        ("RLAIF-single vs DPO",            "0.36", "[0.20, 0.55]", "4 / 10 / 6",  SINGLE_RED),
        ("RLAIF-ensemble vs DPO",          "0.56", "[0.39, 0.74]", "8 / 5 / 7",   ENSEMB_GRN),
        ("RLAIF-ensemble vs RLAIF-single", "0.65", "[0.49, 0.81]", "10 / 2 / 8",  ENSEMB_GRN),
    ]
    col_labels = ["Comparison (A vs B)", "A win-rate", "95 % CI", "A / B / Split"]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")

    tbl = ax.table(
        cellText=[[r[0], r[1], r[2], r[3]] for r in rows],
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.7)

    # Header style
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#1e293b")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Row colours
    for i, (_, _, _, _, color) in enumerate(rows, start=1):
        for j in range(len(col_labels)):
            alpha = 0.18
            tbl[(i, j)].set_facecolor((*_hex_to_rgb(color), alpha))
        # Bold the win-rate column
        tbl[(i, 1)].set_text_props(fontweight="bold")

    fig.suptitle("Pairwise win-rates summary — Precise RLAIF Markov Game",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()

    out = FIG_DIR / "fig_summary_table.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  saved: {out}")
    plt.close(fig)


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))  # type: ignore


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    print("Generating presentation figures...")
    fig_pipeline_progression()
    fig_single_vs_ensemble()
    fig_summary_table()
    print("Done.")


if __name__ == "__main__":
    main()
