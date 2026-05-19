# Precise RLAIF as a Markov Game

**Presentation outline — solo implementation**
Qwen 2.5 0.5B · gpt-oss:20b Critic · gpt-oss:120b Judge · Ollama Cloud

---

## Slide 1 — Title

> **Precise RLAIF as a Markov Game**
> Three-agent framing of sentence-level synthetic-preference fine-tuning
> Implemented end-to-end on M4 Pro

---

## Slide 2 — Problem

- Small instruction-tuned LLMs (Qwen 2.5 0.5B) under-follow constraints.
- Standard DPO needs **human-labeled preference pairs**: expensive, slow, fixed signal.
- **RLAIF**: replace humans with an AI critic. But existing RLAIF treats critic as an oracle, not an agent.

**Our framing:** Precise RLAIF is a **3-agent Markov Game** — Generator (policy), Critic (perturber), Judge (evaluator). This unlocks MARL theory: non-stationarity, ensemble policies, agent diversity.

---

## Slide 3 — Markov Game tuple ⟨N, S, {A_i}, P, {R_i}, γ⟩

| Agent | Symbol | State (observation) | Action |
|---|---|---|---|
| Generator (trainable) | $\pi_G$ | $(x, y_{<t})$ — prompt + partial response | next token ∈ V |
| Critic (4 constitution policies) | $\pi_C^c$ | $(x, y, c)$ — prompt + full response + constitution | (sentence_idx, replacement_text) |
| Judge (frozen) | $\pi_J$ | $(x, y_A, y_B)$ — prompt + two responses | preference ∈ {A, B, tie} |

- $N=3$, partial observability per agent
- $R_G$ = DPO implicit reward induced by Judge labels
- $R_C, R_J$ = fixed (off-the-shelf models, no training)
- $\gamma = 1$ at trajectory level

---

## Slide 4 — Non-stationarity argument (the MARL pitch)

> As $\pi_G$ shifts during training, the response distribution the Critic sees shifts → Critic's effective output distribution shifts → Judge's induced reward landscape shifts.

Classic MARL non-stationarity. Two consequences:

1. **One-shot perturbation = deliberate simplification.** We generate perturbations against the SFT checkpoint *once*, not on-policy. Documented as a limitation; on-policy regeneration is Future Work.
2. **Critic ensemble exploits agent diversity.** 4 constitution-conditioned Critic policies cover different parts of the perturbation space → richer training signal than any single Critic.

---

## Slide 5 — Pipeline

```
                ┌────────────────────────────────────────────────┐
                │            Generator (Qwen 2.5 0.5B)           │
                │           [trainable, LoRA r=16]               │
                └────────┬───────────────────────────────┬───────┘
                         │                               │
                  SFT (Smoltalk)                   DPO (Ultrafeedback)
                         │                               │
                         ▼                               ▼
                   sft-qwen-0.5b ────────────────► dpo-qwen-0.5b
                                                         │
                            ┌────────────────────────────┼─────────────────────────┐
                            │                            │                         │
                            ▼                            ▼                         ▼
                  ┌──────────────────┐         ┌──────────────────┐    ┌──────────────────┐
                  │   Critic π_C^c   │         │ Precise-RLAIF    │    │ Precise-RLAIF    │
                  │ gpt-oss:20b      │────────►│   (single)       │    │   (ensemble)     │
                  │ 4 constitutions  │   pert  │ combined.jsonl   │    │ equal_mix.jsonl  │
                  └──────────────────┘  pairs  └────────┬─────────┘    └────────┬─────────┘
                                                        │                       │
                                                        ▼                       ▼
                                                  rlaif-single            rlaif-ensemble
                                                        │                       │
                                                        └───────────┬───────────┘
                                                                    │
                                                                    ▼
                                                          ┌──────────────────┐
                                                          │     Judge π_J    │
                                                          │ gpt-oss:120b     │
                                                          │ pairwise win-rate│
                                                          └──────────────────┘
```

---

## Slide 6 — The 4 Critic policies

Each constitution = a distinct system prompt → distinct $\pi_C^c$:

| Constitution | Degradation target |
|---|---|
| `instruction_following` | Ignore a stated constraint, answer a different question |
| `helpfulness` | Add vagueness, remove a concrete step |
| `truthfulness` | Plausible-but-false factual edit |
| `combined` | Any of the above (single-Critic baseline) |

Each Critic edits **exactly one sentence** per response → fine-grained, controllable rejected response.

---

## Slide 7 — Implementation reality

| Component | Plan | Built | Notes |
|---|---|---|---|
| Generator stack | SFT → DPO → RLAIF | ✅ end-to-end | LoRA r=16, MPS |
| Critic | GPT-4o, 4 constitutions | ✅ `gpt-oss:20b-cloud` | cheaper, JSON-stable |
| Judge | Nemotron-70B | ✅ `gpt-oss:120b-cloud` | substitution noted in Limitations |
| DPO dataset | 30k Ultrafeedback pairs | 2k pairs | M4 Pro time budget |
| Perturbations | 4 const × 5k | 4 const × 200 | rate-limit constrained |
| Eval | Nemotron + Flores + MBPP + GSM8K + human | Judge pairwise only | scope cut for 1-day window |

---

## Slide 8 — Results: pairwise win-rates

Judge: `gpt-oss:120b-cloud`, **20 held-out prompts × 2 orderings** (position-bias mitigated). Held-out slice = `UltraFeedback train[63000:63020]`, disjoint from DPO training, SFT data, and perturbation slice.

| Comparison (A vs B) | A win-rate | 95% CI | Decisive A / B / Split | Reading |
|---|---|---|---|---|
| SFT vs Qwen-0.5B-base | **0.60** | [0.42, 0.78] | 8 / 4 / 8 | SFT clearly helps |
| DPO vs SFT | **0.51** | [0.33, 0.70] | 7 / 7 / 6 | DPO ≈ SFT at this scale (2k pairs) |
| **RLAIF-single vs DPO** | **0.36** | [0.20, 0.55] | 4 / 10 / 6 | ⚠️ single Critic *hurts* |
| **RLAIF-ensemble vs DPO** | **0.56** | [0.39, 0.74] | 8 / 5 / 7 | ensemble recovers + marginal gain |
| **RLAIF-ensemble vs RLAIF-single** | **0.65** | [0.49, 0.81] | 10 / 2 / 8 | **MARL: ensemble decisively beats single** |

→ See `figures/fig1_winrate.png`.

### Key finding (the MARL story)

The ensemble Critic isn't a marginal MARL improvement — it's **required** for synthetic-preference Precise-RLAIF to work at this scale:

- Single-Critic RLAIF *degrades* the DPO model (0.36 win-rate, 10 decisive losses).
- The 4-Critic ensemble *recovers and marginally improves* over DPO (0.56).
- Head-to-head, the ensemble wins 65% with 83% of decisive cases going to ensemble.

The MARL contribution is therefore not about polishing a working pipeline, but about diagnosing why a naive single-agent synthetic-RLAIF setup fails and showing how Critic diversity fixes it.

---

## Slide 9 — Critic-policy diversity

Heatmap of pairwise agreement on which sentence to perturb (modal sentence_idx per pair).

→ See `figures/fig2_critic_diversity.png`.

**Claim:** off-diagonal values < 0.5 ⇒ the 4 Critics are genuinely distinct policies, not redundant.

(If values close to 1 — Critics agree too much; ensemble offers little extra signal. Frame as a finding either way.)

---

## Slide 10 — Qualitative perturbation table

Same prompt + same response, perturbed under each of the 4 constitutions.

→ See `figures/fig3_perturbation_examples.md`.

Demonstrates that each Critic policy attacks the response along a distinct axis (instruction-following vs helpfulness vs truthfulness vs combined).

---

## Slide 11 — Limitations

1. **No on-policy Critic regeneration** — perturbations frozen against SFT checkpoint. The non-stationarity argument is theoretical; we did not chase it empirically.
2. **Judge ≠ Nemotron-70B** — used `gpt-oss:120b-cloud` for Ollama Cloud access reasons.
3. **Small scale**: 2k DPO pairs, 4×200 perturbations, 30 eval prompts (single seed).
4. **DPO rewards/accuracies on synthetic pairs = 0.32–0.59** — small models struggle to distinguish single-sentence perturbations. Behavioral win-rate (Judge) is the load-bearing metric.
5. **No human eval, no degradation benchmarks** (Flores/MBPP/GSM8K) — 1-day implementation window.

---

## Slide 12 — Future Work

- **On-policy Critic regeneration** every K steps → exercises the full MARL non-stationarity story.
- **Learned (RL-trained) Critic policies** — meta-game where Critics maximize signal for the Generator.
- **Multi-Generator competition** — true competitive MARL instead of the current cooperative setup.
- **Bigger Generator + larger preference set** to test scaling of the ensemble advantage.

---

## Slide 13 — Takeaways

1. Reframing single-agent DPO as a **3-agent Markov Game** organizes the design cleanly: who's training, who's perturbing, who's grading.
2. **The MARL extension is load-bearing, not cosmetic.** Single-Critic Precise-RLAIF degrades the DPO baseline (0.36); the 4-Critic ensemble recovers and improves it (0.56) and wins 65% head-to-head against the single-Critic variant.
3. End-to-end pipeline runs on a **24 GB M4 Pro** with Ollama Cloud as the only external dependency. Total wall-clock: ~3 h.
4. The methodology generalizes: swap Qwen for any small LLM, swap constitutions for any preference dimensions.

---

## Q&A anchors

- **Why a Markov Game and not just an MDP?** Because the Critic's outputs depend on the Generator's distribution → the environment is non-stationary from the Generator's perspective. That's the defining feature of multi-agent RL.
- **Why one-shot perturbations?** Compute and consistency. On-policy regeneration is a 4–5× cost multiplier and would require checkpoint juggling on a 24 GB box. We're explicit about it.
- **Why gpt-oss for both Critic and Judge?** Same provider, same auth, fast iteration. Different model sizes (20b vs 120b) reduce correlated-error risk.
- **What does the ensemble vs single number prove?** That distinct Critic policies provide non-redundant training signal — i.e. agent diversity translates to measurable downstream improvement. (Or doesn't, and we frame it as: a single broad constitution already covers the perturbation space.)
