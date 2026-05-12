# Section 3: Problem Formulation — Design Spec

**Date:** 2026-05-12  
**Author:** Member A  
**Format:** Markdown with inline LaTeX  
**Style:** CS224R course report  
**Diagram:** Mermaid (inline)  

---

## Scope

This spec defines the full content of Section 3 (Problem Formulation) of the Precise RLAIF paper.
It reframes the single-agent SFT → DPO pipeline as a three-agent Markov Game to satisfy rubric
criteria #1 (Markov Game formulation) and #2 (MARL theory / non-stationarity).

---

## Section Structure (Approach C — agent-centric)

```
3. Problem Formulation
  3.1 Generator Agent (π_G)
  3.2 Critic Agent (π_C)
  3.3 Judge Agent (π_J)
  3.4 Full Markov Game Tuple ⟨N, S, {A_i}, P, {R_i}, γ⟩
  3.5 Agent-Interaction Diagram
  3.6 Markov Game vs. Single-Agent MDP
      - Non-stationarity argument
      - Independent Learners design choice
      - Why no centralized critic
```

---

## Full Draft

### 3. Problem Formulation

We formalize the Precise RLAIF pipeline as a **partially observable Markov Game** among three
agents: a Generator, a Critic, and a Judge. The Generator is the trainable policy; the Critic and
Judge are fixed, off-the-shelf models that together produce the preference signal used to update
the Generator via Direct Preference Optimization (DPO).

---

#### 3.1 Generator Agent ($\pi_G$)

The Generator is the trainable policy, instantiated as Qwen 2.5 0.5B. It operates
autoregressively over a prompt $x \in \mathcal{X}$ and produces a response
$y = (y_1, y_2, \ldots, y_T) \in \mathcal{V}^*$.

**Partial observation.** At step $t$, the Generator observes $o_G^t = (x, y_{<t})$ — the prompt
and all tokens emitted so far. It does not observe the Critic's perturbation history or the
Judge's preference labels.

**Action space.** $\mathcal{A}_G = \mathcal{V}$, the full token vocabulary. Each action is the
selection of the next token $y_t \sim \pi_G(\cdot \mid x, y_{<t})$.

**Reward.** The Generator receives the DPO implicit reward, derived post-hoc from the Judge's
binary preference over a chosen–rejected pair $(y_w, y_l)$:

$$r_G(x, y_w, y_l) = \beta \log \frac{\pi_G(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_G(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

where $\pi_{\text{ref}}$ is the SFT-initialized reference policy and $\beta$ controls the KL
penalty strength.

---

#### 3.2 Critic Agent ($\pi_C$)

The Critic is a fixed GPT-4o instance operating under a constitutional prompt. It is not updated
during training; its policy $\pi_C$ is held constant.

**Partial observation.** The Critic observes $o_C = (x, y)$ — the prompt and the Generator's
completed response. It does not observe the Generator's internal token probabilities or the
Judge's labels.

**Action space.** $\mathcal{A}_C = \{1, \ldots, |S(y)|\} \times \Sigma^*$, where $S(y)$ denotes
the set of sentences in $y$ and $\Sigma^*$ is the space of replacement strings. A Critic action
is a pair $(\text{idx}, \hat{s})$: select sentence $\text{idx}$ and substitute it with degraded
text $\hat{s}$, producing the rejected response $y_l$.

**Reward.** Undefined. The Critic is off-policy and receives no training signal in this framework.

---

#### 3.3 Judge Agent ($\pi_J$)

The Judge is a fixed Nemotron-70B instance (with human evaluation substituted at final evaluation
time). Like the Critic, it is not updated.

**Partial observation.** The Judge observes $o_J = (x, y_w, y_l)$ — the prompt, the Generator's
original response (chosen), and the Critic's degraded response (rejected). It does not observe
the Generator's policy weights or the Critic's selection index.

**Action space.** $\mathcal{A}_J = \{\texttt{win}, \texttt{lose}, \texttt{tie}\}$. The Judge
emits a preference label indicating whether the chosen response $y_w$ is preferred over $y_l$,
vice versa, or tied.

**Reward.** Undefined. The Judge is off-policy and receives no training signal.

---

#### 3.4 Full Markov Game Tuple

We formalize the pipeline as a Markov Game
$\langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}, P, \{R_i\}, \gamma \rangle$:

**Agents.** $\mathcal{N} = \{G, C, J\}$ with $|\mathcal{N}| = 3$.

**Global state.** The global state at turn $k$ is:

$$s_k = (x,\; y^{(k)},\; h_k) \in \mathcal{S}$$

where $x$ is the prompt, $y^{(k)}$ is the Generator's response at iteration $k$, and
$h_k = \{(\text{idx}_j, \hat{s}_j)\}_{j<k}$ is the perturbation history accumulated by the
Critic. The state is only partially observed: each agent sees the projection $o_i(s_k)$ as
defined in §3.1–3.3.

**Action spaces.** As defined per agent above:

$$\mathcal{A}_G = \mathcal{V}, \quad \mathcal{A}_C = \{1,\ldots,|S(y)|\} \times \Sigma^*, \quad \mathcal{A}_J = \{\texttt{win}, \texttt{lose}, \texttt{tie}\}$$

**Transition kernel.** $P: \mathcal{S} \times \mathcal{A}_G \times \mathcal{A}_C \times \mathcal{A}_J \to \Delta(\mathcal{S})$
decomposes into three sequential stages within each turn:

1. *Generator step* — deterministic concatenation: $y_{<t+1} = y_{<t} \circ y_t$.
2. *Critic step* — stochastic LLM sampling: $(\text{idx}, \hat{s}) \sim \pi_C(\cdot \mid x, y)$, producing $y_l$.
3. *Judge step* — deterministic scoring: $\ell = \pi_J(x, y_w, y_l) \in \mathcal{A}_J$.

The full response generation (stage 1) is deterministic given the token samples; stochasticity
enters only through the Critic's constitutional LLM call.

**Reward functions.**

$$R_G(s, a) = \beta \log \frac{\pi_G(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_G(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}, \quad R_C = R_J = \varnothing$$

**Discount factor.** $\gamma \in (0, 1)$. Because DPO optimizes a one-shot preference objective
rather than a multi-step return, we treat each $(x, y_w, y_l, \ell)$ tuple as a single-step
episode, effectively setting $\gamma = 0$ for the Generator's optimization horizon while
retaining the Markov Game structure for analysis.

---

#### 3.5 Agent-Interaction Diagram

```mermaid
sequenceDiagram
    autonumber
    participant G as Generator (π_G)<br/>Qwen 2.5 0.5B
    participant C as Critic (π_C)<br/>GPT-4o + Constitution
    participant J as Judge (π_J)<br/>Nemotron-70B

    Note over G,J: Turn k — one DPO training step

    G->>G: Sample y_w ~ π_G(· | x)
    G->>C: (x, y_w)
    C->>C: Select sentence idx, generate ŝ
    C->>J: (x, y_w, y_l = perturb(y_w, idx, ŝ))
    J->>J: Emit preference label ℓ ∈ {win, lose, tie}
    J->>G: Preference pair (y_w, y_l) if ℓ = win
    G->>G: DPO update: ∇ L_DPO(π_G; y_w, y_l)

    Note over G: π_G shifts → distribution of y_w shifts
    Note over C: Input distribution shifts → effective non-stationarity
```

---

#### 3.6 Markov Game vs. Single-Agent MDP

**Why this is a Markov Game, not an MDP.** A single-agent MDP requires that the transition
kernel and reward function be fixed with respect to the optimizing agent's policy. Here, the
Critic's action distribution $\pi_C(\cdot \mid x, y)$ is conditioned on the Generator's output
$y$. As $\pi_G$ is updated via DPO, the marginal distribution of $y$ shifts, which shifts the
distribution of inputs to the Critic, which in turn shifts the distribution of rejected responses
$y_l$ presented to the Judge. The reward signal received by the Generator therefore depends on
the joint behavior of all three agents — the defining property of a Markov Game.

This is the canonical *non-stationarity* argument in MARL: from any single agent's perspective,
the "environment" (here, the Critic + Judge pipeline) is non-stationary because it includes other
agents whose effective behavior changes as policies evolve. Formally, let
$\mathcal{D}_k = \{(x, y_w^{(k)}, y_l^{(k)})\}$ denote the preference dataset at iteration $k$.
Because $y_l^{(k)} = \text{Critic}(x, y_w^{(k)})$ and $y_w^{(k)} \sim \pi_G^{(k)}$, we have
$\mathcal{D}_k \neq \mathcal{D}_{k'}$ whenever $\pi_G^{(k)} \neq \pi_G^{(k')}$. The training
distribution is therefore non-stationary by construction.

**Independent Learners design.** We adopt the Independent Learners (IL) paradigm: only the
Generator's policy $\pi_G$ is updated; the Critic and Judge are fixed, off-the-shelf models.
This choice is justified on two grounds. First, *compute*: jointly fine-tuning GPT-4o and
Nemotron-70B alongside Qwen 2.5 0.5B is infeasible within the resource constraints of this
project. Second, *contribution*: the research question is whether a single small model benefits
from a structured multi-agent preference signal; training the feedback providers is orthogonal
to that question and would conflate the source of any observed gains.

**Why no centralized critic.** Centralized training with decentralized execution (CTDE) would
require a joint value function $V(s)$ over the full global state, including the Critic's
constitutional outputs and the Judge's scoring internals — both of which are black-box API calls.
No gradient signal flows through these components, making a centralized critic untrainable. The
IL design is therefore not merely a simplification but a necessity given the partially-observable,
black-box nature of the Critic and Judge.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Output format | Markdown + LaTeX | Easy to edit; renders in GitHub |
| Section structure | Agent-centric (Approach C) | Maps directly to rubric criteria |
| Diagram format | Mermaid sequenceDiagram | Inline, renders without tooling |
| γ treatment | Effectively 0 per episode | DPO is a one-shot objective; game structure preserved for analysis |
| IL justification | Compute + contribution scope | CTDE untrainable with black-box APIs |
