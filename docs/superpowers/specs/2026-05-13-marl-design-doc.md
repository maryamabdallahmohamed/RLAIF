# Precise RLAIF — MARL Design Document & Related Work

**Date:** 2026-05-13
**Author:** Member A
**Status:** Final — hand to Member B by Day 9
**Companion spec:** `2026-05-12-markov-game-formulation-design.md` (full formal Markov Game tuple)

---

## Part 1: MARL Design Document

*Purpose: implementation contract for Member B; analysis lens for Member C.*

---

### §1 Agent Specifications

Three agents operate in the pipeline. Only the Generator is trained; the Critic and Judge are
fixed inference-only APIs.

#### Generator (π_G)

| Property | Value |
|---|---|
| Model | Qwen 2.5 0.5B |
| Adapter | LoRA (r=16, α=32, target: q/k/v/o_proj) |
| Trainable | **Yes** |
| Observation | (x, y_{<t}) — prompt and partial response |
| Action space | Token vocabulary 𝒱 |
| Reward | DPO implicit reward (see §2) |
| Update rule | DPO loss ∇L_DPO over offline preference dataset D_k |

The Generator is the sole learning agent. Its policy π_G is initialized from the SFT checkpoint
and updated via DPO at the end of each training phase.

#### Critic (π_C)

| Property | Value |
|---|---|
| Model | GPT-4o (API) under a constitutional system prompt |
| Trainable | **No** |
| Observation | (x, y) — prompt and completed Generator response |
| Action space | {sentence index} × {replacement string} |
| Output | Rejected response y_l = perturb(y, idx, ŝ) |
| Reward | None — off-policy, receives no training signal |

The Critic's role is adversarial perturbation: given a Generator response, it selects one
sentence and substitutes a degraded version, producing the rejected response for the DPO pair.
The constitutional prompt encodes the perturbation taxonomy (factual error, hedging removal,
coherence degradation, etc.).

**Implementation note for B:** The Critic is called once per (x, y_w) sample during dataset
construction. It is not invoked during the DPO training loop itself.

#### Judge (π_J)

| Property | Value |
|---|---|
| Model | Nemotron-70B (API); human evaluators at final eval |
| Trainable | **No** |
| Observation | (x, y_w, y_l) — prompt, chosen response, rejected response |
| Action space | {win, lose, tie} |
| Output | Preference label ℓ |
| Reward | None — off-policy, receives no training signal |

The Judge arbitrates each Critic-perturbed pair. Only pairs where ℓ = win (Generator preferred)
enter the DPO dataset; ties and losses are discarded.

---

### §2 Interaction Protocol

Each training iteration k executes the following sequential pipeline:

```
Step 1  [Generator]  Sample y_w ~ π_G(· | x)  for each prompt x in batch
Step 2  [Critic]     For each (x, y_w): call π_C → (idx, ŝ) → y_l = perturb(y_w, idx, ŝ)
Step 3  [Judge]      For each (x, y_w, y_l): call π_J → ℓ ∈ {win, lose, tie}
Step 4  [Filter]     Retain only pairs where ℓ = win → preference dataset D_k
Step 5  [Generator]  DPO update: minimize L_DPO(π_G; π_ref, D_k, β)
```

**DPO loss:**

$$\mathcal{L}_{\text{DPO}}(\pi_G;\,\pi_{\text{ref}},\,\mathcal{D}_k,\,\beta) =
-\mathbb{E}_{(x,\,y_w,\,y_l)\sim\mathcal{D}_k}\!\left[
  \log\sigma\!\left(
    \beta\log\frac{\pi_G(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
    -\beta\log\frac{\pi_G(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
  \right)
\right]$$

where π_ref is the frozen SFT checkpoint and β ∈ (0,1] controls KL regularization strength.

**Dataset generation timing (critical for B).** D_k is constructed **offline before each
training phase**, not on-policy during training. Concretely:

1. Draw a fixed prompt set X_k from the held-out pool.
2. Run the full Steps 1–4 pipeline to produce D_k.
3. Train the Generator on D_k for one DPO epoch.
4. Increment k; repeat from step 1 with the updated π_G.

The Critic and Judge APIs are never called during the DPO training loop — only during dataset
construction. This decoupling simplifies implementation: dataset construction and model training
are two separate scripts with a checkpoint written between them.

---

### §3 Non-Stationarity Treatment

#### 3.1 Why Independent Learners (not Centralized Critic / CTDE)

Centralized Training with Decentralized Execution (CTDE) requires a joint value function
V(s) over the full global state — including the Critic's constitutional outputs and the Judge's
scoring internals. Both are black-box API calls: no gradient flows through them, and their
internal states are unobservable. A centralized critic is therefore **untrainable**, not merely
inconvenient. The Independent Learners (IL) design is a necessity given the architecture, not a
simplification.

Additionally: the research question is whether a single small model (Generator) benefits from a
structured multi-agent preference signal. Training the feedback providers would be orthogonal to
that question and would conflate the source of any observed gains.

#### 3.2 Joint vs. Independent Policies

We adopt **independent policies**: each agent acts under its own policy without explicit
communication or joint action coordination. Coordination is **implicit**, mediated entirely
through the reward signal chain:

```
Critic perturbation choice
    → quality of rejected response y_l
        → Judge preference label ℓ
            → which pairs enter D_k
                → DPO gradient on Generator
```

There is no joint policy π_{G,C,J} and no shared parameter update. The Generator is unaware of
the Critic's perturbation index; the Critic is unaware of the Judge's label; the Judge is unaware
of the DPO update magnitude.

#### 3.3 Offline Dataset Generation: Deliberate Simplification

**The choice.** Perturbation datasets are generated once before each training phase rather than
on-policy (i.e., generated continuously as π_G evolves during training).

**Why this is a simplification.** In a fully on-policy MARL setup, D_k would be regenerated
after every gradient step, keeping y_w drawn from the current π_G at all times. This eliminates
distributional lag: the DPO signal always reflects the Generator's current behavior.

**What we lose.** Offline generation introduces a lag: by the end of a training phase, the
Generator has shifted away from the π_G that produced D_k. The preference pairs become slightly
off-policy. In practice, this lag is bounded by the number of gradient steps per phase; if phases
are short (≤500 steps), the distributional shift is small and the off-policy error is tolerable.

**What on-policy regeneration would require:**
- ~3–5× more API calls per gradient step (Critic + Judge per sample, per step)
- GPU idle time during each Critic/Judge inference call (latency ~1–3s per sample at GPT-4o
  speeds), effectively serializing what is currently a batched offline process
- Significantly higher cost per training run; incompatible with the API budget of this project

**Documentation for Week 3.** When Member B implements the multi-Critic ensemble, this
architecture means the ensemble runs during dataset construction (offline), not during training.
Each Critic variant produces its own D_k^{(i)}; these are merged or interleaved before the DPO
step. No changes to the training loop are required.

---

## Part 2: Related Work (Section 2 — Paper Draft)

*For Member C's analysis lens. Positions Precise RLAIF within the literature.*

---

### 2. Related Work

**Reinforcement Learning from AI Feedback (RLAIF).** Lee et al. (2024) demonstrate that
AI-generated preference labels can match or exceed the quality of human annotations when the
labeling model is sufficiently capable, enabling scalable alignment without per-task human
raters. Precise RLAIF extends this line by introducing a structured adversarial perturbation
step (the Critic) between response generation and preference labeling, ensuring that the
preference signal is grounded in specific, localizable quality dimensions rather than holistic
impressions.

**Constitutional AI.** Bai et al. (2022b) show that a model can critique and revise its own
outputs according to a set of human-authored principles (a "constitution"), removing the need for
human feedback in the RLHF inner loop. Our Critic agent directly inherits this design: GPT-4o
operates under a constitutional system prompt that defines the perturbation taxonomy
(factual accuracy, hedging, coherence). The key departure is directionality — Constitutional AI
revises responses upward toward the constitution; our Critic degrades them downward to construct
contrastive rejected samples for DPO.

**Direct Preference Optimization.** Rafailov et al. (2023) show that the RLHF objective with
a KL-regularized reward model has a closed-form optimal policy, enabling preference-based
alignment to be reduced to a supervised classification loss over (chosen, rejected) pairs. This
eliminates the explicit reward model and the PPO training loop, substantially reducing
implementation complexity and training instability. We adopt DPO as the Generator's update rule
because its offline, dataset-first interface is architecturally compatible with our offline dataset
construction pipeline (§2 above): preference pairs are assembled before training begins, matching
DPO's static-dataset assumption exactly.

**Multi-Agent Reinforcement Learning with LLM Agents.** A growing body of work treats LLMs as
autonomous agents embedded in multi-agent environments. Li et al. (2023) introduce CAMEL, in
which two LLMs play cooperative roles (instructor and assistant) to solve tasks through dialogue,
demonstrating that role-constrained prompting produces coherent multi-turn coordination without
explicit joint training. Du et al. (2023) show that debate between multiple LLM instances
improves factual accuracy and reasoning, providing an early proof-of-concept for adversarial
multi-agent interaction improving output quality. More recent work on LLM-based MARL (e.g.,
Agashe et al. 2023; Chen et al. 2024) explores non-stationarity in settings where multiple LLMs
fine-tune concurrently, finding that independent learners with frozen feedback providers is a
stable and practical baseline. Our three-agent pipeline is consistent with this design:
the Generator is the sole learner; the Critic and Judge are fixed, role-constrained agents whose
effective behavior shifts only insofar as the Generator's output distribution shifts — the
canonical source of non-stationarity in IL-MARL systems.

---

## Handoff Checklist (Member A → Member B, by Day 9)

- [ ] Generator SFT checkpoint saved at `sft-qwen-0.5b/`
- [ ] DPO training script scaffolded (dataset loader + `DPOTrainer` call)
- [ ] Critic API wrapper implemented (GPT-4o + constitutional prompt → perturbed response)
- [ ] Judge API wrapper implemented (Nemotron-70B → preference label)
- [ ] Dataset construction pipeline: Steps 1–4 in §2 above, output format `{prompt, chosen, rejected}`
- [ ] Week 3 multi-Critic ensemble: implement as parallel Critic calls during dataset construction;
      merge D_k^{(i)} datasets before passing to DPO trainer
