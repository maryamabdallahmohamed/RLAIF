"""
Build all figures for the presentation:

  fig1_winrate.png         — bar chart of pairwise win-rates with 95% CI
  fig2_critic_diversity.png — heatmap: % overlap of sentence_idx across constitution pairs
  fig3_perturbation_examples.md — side-by-side qualitative table (markdown)

Usage:
    PYTHONPATH=. python eval/figures.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("data/perturbation")
EVAL_DIR = Path("eval")
FIG_DIR = Path("figures")
CONSTITUTIONS = ["instruction_following", "helpfulness", "truthfulness", "combined"]


def fig_critic_diversity() -> None:
    """Heatmap: for each pair of constitutions, fraction of (prompt, response) pairs
    where they agree on the targeted sentence_idx. Lower = more diverse policies."""
    import matplotlib.pyplot as plt
    import numpy as np

    by_const: dict[str, dict[tuple[str, str], list[int]]] = defaultdict(lambda: defaultdict(list))
    for c in CONSTITUTIONS:
        path = DATA_DIR / f"{c}.jsonl"
        if not path.exists():
            print(f"  missing: {path} — skip diversity figure")
            return
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = (rec["prompt"], rec["original_response"])
                by_const[c][key].append(rec["perturbed_sentence_idx"])

    # Modal idx per (constitution, pair)
    modal: dict[str, dict[tuple[str, str], int]] = {}
    for c, m in by_const.items():
        from collections import Counter
        modal[c] = {k: Counter(v).most_common(1)[0][0] for k, v in m.items()}

    # Pairwise agreement
    n = len(CONSTITUTIONS)
    M = np.zeros((n, n))
    for i, ci in enumerate(CONSTITUTIONS):
        for j, cj in enumerate(CONSTITUTIONS):
            shared = set(modal[ci].keys()) & set(modal[cj].keys())
            if not shared:
                continue
            agree = sum(1 for k in shared if modal[ci][k] == modal[cj][k])
            M[i, j] = agree / len(shared)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([c.replace("_", "\n") for c in CONSTITUTIONS], fontsize=9)
    ax.set_yticklabels([c.replace("_", "\n") for c in CONSTITUTIONS], fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.5 else "black", fontsize=10)
    ax.set_title("Critic-policy diversity\n(fraction of pairs with same target sentence)")
    fig.colorbar(im, ax=ax, label="agreement rate")
    fig.tight_layout()
    out = FIG_DIR / "fig2_critic_diversity.png"
    fig.savefig(out, dpi=150)
    print(f"  saved: {out}")


def fig_winrate() -> None:
    """Bar chart over win-rate JSONs in eval/."""
    import matplotlib.pyplot as plt

    results = []
    for p in sorted(EVAL_DIR.glob("winrate_*.json")):
        d = json.loads(p.read_text())
        s = d["summary"]
        results.append(s)
    if not results:
        print("  no winrate_*.json found — skip win-rate figure")
        return

    labels = [f"{s['label_a']}\nvs\n{s['label_b']}" for s in results]
    means = [s["a_winrate_mean"] for s in results]
    lows = [s["a_winrate_ci95"][0] for s in results]
    highs = [s["a_winrate_ci95"][1] for s in results]
    err_low = [m - lo for m, lo in zip(means, lows)]
    err_high = [hi - m for m, hi in zip(means, highs)]

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(results)), 4.5))
    bars = ax.bar(range(len(results)), means,
                  yerr=[err_low, err_high], capsize=5,
                  color=["#3b82f6"] * len(results), edgecolor="black")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("A win-rate (95% CI)")
    ax.set_ylim(0, 1)
    ax.set_title("Pairwise win-rates (Judge: gpt-oss:120b-cloud)")
    for i, (m, hi) in enumerate(zip(means, highs)):
        ax.text(i, hi + 0.02, f"{m:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "fig1_winrate.png"
    fig.savefig(out, dpi=150)
    print(f"  saved: {out}")


def fig_perturbation_examples(n_examples: int = 2) -> None:
    """Markdown table: same prompt+response, 4 different constitution perturbations."""
    rows_by_pair: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for c in CONSTITUTIONS:
        path = DATA_DIR / f"{c}.jsonl"
        if not path.exists():
            print(f"  missing: {path} — skip examples")
            return
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = (rec["prompt"], rec["original_response"])
                if c not in rows_by_pair[key]:
                    rows_by_pair[key][c] = rec["perturbed_response"]

    complete = [(k, v) for k, v in rows_by_pair.items() if len(v) == len(CONSTITUTIONS)]
    complete = complete[:n_examples]
    out = FIG_DIR / "fig3_perturbation_examples.md"
    with open(out, "w") as f:
        f.write("# Critic-policy diversity: side-by-side perturbations\n\n")
        f.write("Same prompt + same original response, perturbed by each of the 4 constitution-conditioned Critic policies.\n\n")
        for i, ((prompt, orig), perts) in enumerate(complete, 1):
            f.write(f"## Example {i}\n\n")
            f.write(f"**Prompt:** {prompt[:300]}{'…' if len(prompt) > 300 else ''}\n\n")
            f.write(f"**Original response:** {orig[:500]}{'…' if len(orig) > 500 else ''}\n\n")
            f.write("| Constitution | Perturbation |\n|---|---|\n")
            for c in CONSTITUTIONS:
                txt = perts[c].replace("\n", " ")[:400]
                f.write(f"| **{c}** | {txt}{'…' if len(perts[c]) > 400 else ''} |\n")
            f.write("\n")
    print(f"  saved: {out}")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)
    print("Building figures...")
    fig_critic_diversity()
    fig_winrate()
    fig_perturbation_examples()
    print("Done.")


if __name__ == "__main__":
    main()
