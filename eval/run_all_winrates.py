"""
Run all pairwise win-rate comparisons for the report.

The load-bearing MARL result is rlaif-ensemble vs rlaif-single.
Other comparisons are progression sanity-checks.

Usage:
    PYTHONPATH=. python eval/run_all_winrates.py --n-prompts 30
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.agents.judge.winrate import run_winrate
from src.utils.env import load_env


COMPARISONS = [
    # (adapter_a, label_a, adapter_b, label_b, output_basename)
    # The MARL comparison — most important.
    ("rlaif-ensemble", "RLAIF-ensemble", "rlaif-single", "RLAIF-single",
     "winrate_ensemble_vs_single"),
    # Does RLAIF help over DPO?
    ("rlaif-single",   "RLAIF-single",   "dpo-qwen-0.5b", "DPO",
     "winrate_rlaif_single_vs_dpo"),
    ("rlaif-ensemble", "RLAIF-ensemble", "dpo-qwen-0.5b", "DPO",
     "winrate_rlaif_ensemble_vs_dpo"),
    # Did DPO help over SFT? (training progression)
    ("dpo-qwen-0.5b",  "DPO",            "sft-qwen-0.5b", "SFT",
     "winrate_dpo_vs_sft"),
    # And did SFT help over base?
    ("sft-qwen-0.5b",  "SFT",            None,            "Qwen-0.5B-base",
     "winrate_sft_vs_base"),
]


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=30)
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip comparisons where an adapter dir is missing.")
    args = parser.parse_args()

    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-oss:120b-cloud")
    ollama_url = os.environ.get("OLLAMA_URL", "https://ollama.com")
    print(f"Judge: {judge_model} @ {ollama_url}")
    print(f"Prompts per comparison: {args.n_prompts}")
    print(f"Total Judge calls: {2 * args.n_prompts * len(COMPARISONS)}")

    summaries: list[dict] = []
    for ad_a, lbl_a, ad_b, lbl_b, basename in COMPARISONS:
        out = eval_dir / f"{basename}.json"
        # Sanity: both adapters must exist (or be None for base)
        missing = []
        for ad in (ad_a, ad_b):
            if ad is not None and not Path(ad).exists():
                missing.append(ad)
        if missing:
            msg = f"missing adapter(s): {missing}"
            if args.skip_missing:
                print(f"\n[SKIP {basename}] {msg}")
                continue
            else:
                raise SystemExit(f"[FATAL {basename}] {msg} — pass --skip-missing to continue.")

        if out.exists():
            print(f"\n[SKIP {basename}] result already exists: {out}")
            continue

        print(f"\n=== {lbl_a} vs {lbl_b} → {out.name} ===")
        s = run_winrate(
            adapter_a=ad_a,
            adapter_b=ad_b,
            label_a=lbl_a,
            label_b=lbl_b,
            n_prompts=args.n_prompts,
            judge_model=judge_model,
            ollama_url=ollama_url,
            output_path=out,
        )
        summaries.append(s)

    print("\n=== Summary ===")
    for s in summaries:
        mean = s["a_winrate_mean"]
        lo, hi = s["a_winrate_ci95"]
        print(f"  {s['label_a']:18s} vs {s['label_b']:18s}: {mean:.3f}  [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
