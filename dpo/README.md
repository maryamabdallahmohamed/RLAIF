# DPO Training (Member A, Week 2)

Trains a Direct Preference Optimization (DPO) baseline on top of the Week-1 SFT'd
Qwen 2.5 0.5B. This checkpoint is:

1. **The baseline Week 3's Precise-RLAIF must outperform**, and
2. **The reference policy (π_ref) and starting point (π_θ) for Week 3's RLAIF runs.**

## What it does

For each prompt in `openbmb/UltraFeedback`, the four model completions come with
quality ratings. We pick the highest-rated as `chosen`, the lowest as `rejected`,
drop ties (score gap < 0.5), and train Qwen-2.5-0.5B-SFT to assign higher
likelihood to `chosen` than to `rejected` — with the SFT model itself acting as
a frozen KL anchor so the policy doesn't drift off the manifold.

TRL's `DPOTrainer` is the workhorse; we don't reimplement the loss.

## Files

- [`preprocessing.py`](preprocessing.py) — UltraFeedback → `(prompt, chosen, rejected)` triples
- [`train_dpo.py`](train_dpo.py) — TRL `DPOTrainer` loop; loads SFT adapter, trains it further
- [`dpo_colab.ipynb`](dpo_colab.ipynb) — Colab notebook (driven from the VS Code Colab extension)
- Tests live at `../tests/dpo/test_preprocessing.py`

## How to run

### Prereqs
- The SFT LoRA adapter must be at the repo root: `sft-qwen-0.5b/`. This folder is
  in `.gitignore` (weights stay out of git) — copy it in manually before running.
- A GPU. Designed for a Colab T4 (≈15 GB VRAM), but any CUDA card with ≥10 GB
  works. CPU and MPS paths exist for sanity tests only.

### Run via the Colab VS Code extension (preferred)
1. Open `dpo/dpo_colab.ipynb` in VS Code.
2. Connect a Colab T4 runtime.
3. Execute cells in order: install deps → GPU check → layout check → smoke run → full run → generation sanity check.

If the Colab VS Code extension misbehaves, the same notebook can be uploaded to
colab.research.google.com directly, or `dpo/train_dpo.py` can be invoked from
a terminal on any CUDA-equipped Colab/Lightning session.

### Run from a terminal
```bash
# Smoke (≈10 min on T4): 200 steps, ~500 pairs
python -m dpo.train_dpo --fast

# Full (≈2–4 h on T4): 1 epoch, 10k pairs → checkpoints/dpo-qwen-0.5b/
python -m dpo.train_dpo
```

### Run the unit tests (CPU-only, no model loading)
```bash
python -m pytest tests/dpo/test_preprocessing.py -q
```

## Hyperparameters

| Param | Value | Notes |
|---|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` | unchanged from Week 1 |
| Warm start | `sft-qwen-0.5b/` (LoRA r=16, α=32, q/k/v/o\_proj) | provided by Member A's SFT step |
| β (KL strength) | 0.2 | paper default |
| Learning rate | 5e-6 | DPO-on-LoRA standard; lower than SFT |
| Effective batch | 16 (per-device 2 × grad-accum 8) | T4 VRAM budget |
| Epochs | 1 | |
| max\_length / max\_prompt\_length | 1024 / 512 | |
| Preference pairs | 10 000 | filtered, score gap ≥ 0.5 |

## Deviations from the docx plan

These are conscious trade-offs for working on a free-tier GPU. List them in the
report's Limitations subsection.

| Plan said | We do | Why |
|---|---|---|
| Full fine-tune | LoRA (same as Week 1 SFT) | Free-tier VRAM; consistent with the SFT we received |
| 30 000 preference pairs | 10 000 pairs | Fits a Colab T4 session window |
| lr = 1e-5 | lr = 5e-6 | Standard for DPO with LoRA; 1e-5 can be unstable on small models |
| batch 4 × accum 4 | batch 2 × accum 8 | T4 VRAM headroom |
| W&B logging | TensorBoard + stdout | No W&B account assumed |
| SFT trained 1 epoch over Smoltalk | Provided SFT adapter is partial (~235 steps, ~7.5k samples) | Member A produced it under the same FAST\_MODE-ish constraints |

## Output

`dpo-qwen-0.5b/` (worktree root, gitignored) containing
`adapter_model.safetensors`, `adapter_config.json`, and the tokenizer files.
Hand the folder to Member C for win-rate evaluation against the baseline and
the SFT checkpoint.

## Run

Trained on a free Colab T4 (15 GB VRAM). Full training log is in
[`training_log.txt`](training_log.txt). Headline numbers from the final run:

| Metric | Start | End | Direction |
|---|---|---|---|
| DPO loss | 0.671 | 0.566 | ↓ |
| `rewards/margins` | 0.049 | 0.539 | ↑ (~11×) |
| `rewards/accuracies` | 0.59 | 0.80 | ↑ |
| `rewards/chosen` | -0.013 | -0.277 | ↓ |
| `rewards/rejected` | -0.062 | -0.816 | ↓ (further than chosen — this is what opens the margin) |
| `grad_norm` | 14–16 | 14–16 | Stable, no NaN/explosion |

- **Total runtime:** ~38 min wall-clock for 313 steps over 5 000 pairs.
- **Throughput:** ~7.2 s/step on T4 (fp16, no gradient checkpointing, max_length=512). This is the figure to use when scheduling Week 3's Precise-RLAIF runs — expect ~40 min per variant on a similar-sized dataset.
- **Precision:** fp16. Bf16 on T4 is emulated and runs ~2× slower; the script auto-selects fp16 on sm_75 GPUs and bf16 on Ampere+ (sm_80+).

### Generation sanity check (greedy decoding)

The model produces coherent answers on most prompts (photosynthesis, unit
tests) but loops on the haiku prompt and trails into stray template artifacts
on others. This is **not a DPO problem** — it is a known interaction between
greedy decoding (`do_sample=False`) on small models and the tokenizer's EOS
alignment, surfaced by the
"tokenizer has new PAD/BOS/EOS tokens" warning at the start of training.
Pairwise win-rate evaluation (Member C's task) compares two responses to the
same prompt under the same decoding regime, so these artifacts cancel out in
the metric we care about. Flag in the report's Limitations section.

## Handoff to Member C

For the win-rate evaluation step, Member C needs both adapters loaded on top
of `Qwen/Qwen2.5-0.5B`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-0.5B"
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="float16", device_map="cuda")

# SFT baseline
tok_sft = AutoTokenizer.from_pretrained("sft-qwen-0.5b")
sft_model = PeftModel.from_pretrained(base, "sft-qwen-0.5b").eval()

# DPO baseline (load on a separate base copy; you cannot stack both adapters on the same base instance)
base2 = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="float16", device_map="cuda")
tok_dpo = AutoTokenizer.from_pretrained("dpo-qwen-0.5b")
dpo_model = PeftModel.from_pretrained(base2, "dpo-qwen-0.5b").eval()
```

## Out of scope

- Win-rate / Nemotron evaluation — Member C's task.
- Hyperparameter sweep — single seed only (documented limitation).
- Direct comparison to the paper's reported 68.5% — Member C produces the
  number; this module only produces the checkpoint.
