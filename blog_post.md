---
title: "DataForge Arena: I Built a Benchmark Where the Data Fights Back"
thumbnail: https://huggingface.co/spaces/Vivek567/dataforge-arena/resolve/main/thumbnail.png
authors:
  - user: Vivek567
---

# DataForge Arena: I Built a Benchmark Where the Data Fights Back

*Most AI demos quietly assume the data is already clean. This one doesn't.*

---

There's a lie baked into almost every enterprise AI demo you've ever watched.

The agent walks up to a beautiful, well-structured table. Every column is the right type. Every row makes sense. The null values are gone. The duplicates have been removed. The foreign keys point to things that actually exist.

Then the agent does something clever, the crowd applauds, and nobody mentions the intern who spent three days cleaning that data before the demo started.

**Real enterprise data is nothing like that.**

Real data has an `age` field set to `145`. It has a `birth_year` of `1979` sitting right next to it, and nobody has noticed the contradiction in six months of production. It has format mismatches, duplicated records with one mutated field designed to slip past naive deduplication, and type errors that look plausible until you run the numbers.

An agent that can't handle that isn't a production agent. It's a demo.

That's the gap **DataForge Arena** is built to close.

---

## The Core Idea: Data Cleaning as World Modeling

Most data cleaning tools are static. You write a rule. The rule fires or it doesn't. You don't learn anything. The agent doesn't learn anything.

DataForge Arena turns data repair into a **world-modeling task**.

The agent receives a structured observation: schema information, sample rows, corruption context. It emits a JSON repair action — which tool to use, which column to target, which row to fix, and *why*. The environment applies that action and computes reward from the actual state delta. Not from a rubric. Not from a judge. From whether the table measurably got closer to ground truth.

This distinction matters enormously.

When reward comes from the actual outcome of an action in a real environment, the agent is forced to build a genuine model of the world it operates in. It can't fake it. It can't generate a confident explanation and hope nobody checks. It has to be *right* — or the number goes down.

That's the design principle. That's what makes this an OpenEnv benchmark and not just another eval dataset.

---

## What the Agent Actually Faces

Let me show you a concrete example. This is not a toy.

The environment surfaces a corrupted row:

```
age        = 145
birth_year = 1979
schema_range = [0, 120]
column_mean  = 42
column_std   = 18
```

To earn maximum reward on this row, the agent must simultaneously:

**1. Check the type system.**
`age` is declared `int` with range `[0, 120]`. The value `145` violates the upper bound. That fires `constraint_alignment`. A cell-level classifier catches this. Good start — but not enough.

**2. Cross-reference columns.**
`birth_year = 1979` implies this patient should be approximately 45 years old in 2024. The gap between 145 and 45 is not random noise — it's a specific, structured corruption. The agent must reason *across* two columns and apply temporal inference to understand what actually happened here. A statistical model doesn't do this. A rule engine doesn't do this without being explicitly programmed for it.

**3. Compute statistics.**
With `column_mean = 42` and `column_std = 18`, the z-score of 145 is `(145 − 42) / 18 = 5.7`. That fires `outlier_targeting`.

All three signals must fire **together**. Getting one right by luck earns almost nothing. The reward structure is designed so that you have to be right about the violation *type*, the right *tool*, and the right *target cell*, simultaneously, to score well.

This is what I mean by relational reasoning. The agent needs a model of the schema — type system, foreign-key map, statistical distributions, temporal constraints — held simultaneously. Not rules. A *model*.

---

## Before and After Training

Here's what that looks like empirically.

**Before training — no world model:**

```json
{
  "reasoning": "fix",
  "tool_id": 0,
  "column": 0,
  "row_id": 0
}
```

Wrong cell. Wrong tool. No justification. The model is randomly poking at the table.

**After 300 steps of GRPO training — world model acquired:**

```json
{
  "reasoning": "age 145 exceeds schema max 120; birth_year 1979 implies age ~45 in 2024; z-score 5.7 confirms statistical outlier",
  "tool_id": 3,
  "column": 2,
  "row_id": 7
}
```

Correct cell. Correct tool. Causal justification that references schema range, temporal inference, and statistical distribution — simultaneously and correctly.

That's not a hallucination. That's a world model.

---

## The Training Architecture

The training path uses **TRL GRPO** to optimize a language-model surgeon over structured repair actions.

The reward shaping is deliberate:

| Signal | Weight | What It Measures |
|---|---|---|
| `accuracy_delta` | ×50 | Did the table actually improve? |
| `constraint_alignment` | +3.0 | Did the agent identify the right violation type? |
| `schema_alignment` | +2.0 | Did it target the right column and tool? |
| `reasoning_quality` | +1.5 | Is the justification causally coherent? |

The `accuracy_delta` term dominates by design. The agent cannot win by writing elegant reasoning chains and choosing the wrong cell. It cannot win by fixing the right cell with the wrong tool. The shaped signals exist to reward the *structure* of correct reasoning, but the ground truth signal is always whether the corrupted table became less corrupted.

The corruptor runs across three tiers of adversarial difficulty — from straightforward nulls and type mismatches at Tier 1, up to cross-column relational corruptions and mutated duplicate rows at Tier 3.

---

## What the Numbers Actually Say (No Spin)

Let me be honest about both what worked and what the limits are.

**300 steps of GRPO on a T4 GPU.** Here's the raw scorecard:

| Metric | Value |
|---|---|
| Steps completed | 300 |
| Reward improvement | +2.54 (1.93 → 4.47, smoothed) |
| Best reward | 6.95 at step 30 |
| Parse success | 100% sustained |
| GRPO destruction ratio | 0.102 |
| GRPO advantage vs. random | +0.44 percentage points |
| Heuristic baseline win rate | 50% |
| GRPO win rate | 5% |

The GRPO agent is **9.8× less destructive than random action**. Parse success held at 95–100% throughout. The reward trend is real and upward.

Win rate is 5%. I'm not hiding that. The heuristic baseline at 50% win rate represents the performance ceiling for this model size at this step count — and it exists precisely to *prove the environment is learnable*. Something can get to 50% win rate. The RL agent is early in the climb.

The story here is not "GRPO solved data cleaning." The story is: **the environment is real, the signal is clean, and the learning curve is moving in the right direction.** That's the foundation you need to scale.

---

## The Environment Architecture

Four components, each doing one job cleanly:

**OpenEnv Environment** — exposes `reset()` and `step()` around a tabular repair world. Fully compatible with the OpenEnv interface spec.

**Adversarial Corruptor** — injects solvable corruptions at configurable tiers. Nulls, type errors, format mismatches, FK inconsistencies, mutated duplicate rows. Every corruption is reversible; every corruption has a ground truth. The agent is never fighting unsolvable problems.

**Repair Action Space** — explicit tools: imputation, format correction, type casting, row deletion, flagging, no-op. The agent must choose the right tool, not just the right cell. A model that imputes when it should delete earns less reward than one that makes the correct surgical choice.

**Reward Computer** — measures the actual state delta after each action. The ground truth table exists. The corrupted table exists. The distance between them is computable. Reward is that distance shrinking. No proxy metrics. No self-evaluation. Just the table.

---

## The Gradio Demo

The demo is designed for visibility, not theater.

A judge can generate a Tier 1 (straightforward) or Tier 3 (adversarial) scenario and run any of the three execution paths:

- **Naive Baseline** — random tool selection, no reasoning
- **Heuristic Surgeon** — rule-based repair, deterministic and fast
- **Live GRPO Model** — the trained checkpoint, when available locally

The UI shows mode provenance, dataset health before and after repair, accuracy delta, cumulative reward, and the full action trajectory. Every claim in the writeup is inspectable on screen. If something doesn't add up, you can see exactly where it breaks.

That's the standard I'm holding this to.

---

## Why This Matters Beyond the Hackathon

Enterprise AI is being deployed into systems where the data is never clean. The choice isn't "clean data or messy data." The choice is "agents that can handle messy data or agents that quietly fail."

The standard approach is to paper over this with preprocessing pipelines written by humans, maintained by humans, and inevitably failing when the data distribution shifts in ways the humans didn't anticipate.

DataForge Arena is a small environment. It runs on a T4. You can clone it, reproduce every number in this post, and extend it in an afternoon. But its structure captures the core difficulty that scales: **actions change the world, and the world should grade those actions**. The grounding is real. The feedback is real. The learning is real.

That's the foundation for training agents that are safe to deploy in imperfect systems — not because they've been told to be careful, but because they've been trained in an environment where being wrong has measurable consequences.

---

## Try It

```bash
git clone https://github.com/vivekyarra/dataforge-arena
cd dataforge-arena
pip install -r requirements.txt

# Run the full test suite (130 tests)
python -m pytest -q

# Reproduce the heuristic baseline
python eval/evaluate.py --agent-mode heuristic

# Reproduce the GRPO checkpoint eval
python eval/evaluate.py --agent-mode grpo

# After training your own checkpoint
python eval/evaluate.py --agent-mode grpo --model-path outputs/dataforge-surgeon
```

| Resource | Link |
|---|---|
| 🤗 Live HF Space | [Vivek567/dataforge-arena](https://huggingface.co/spaces/Vivek567/dataforge-arena) |
| 📓 Colab Notebook | DataForge_Arena_Colab.ipynb |
| 💻 GitHub | [vivekyarra/dataforge-arena](https://github.com/vivekyarra/dataforge-arena) |

---

*Built for the Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026.*

*Built with PyTorch, TRL GRPO, OpenEnv, and a stubborn belief that agents should be graded by what they actually fix — not how confidently they describe fixing it.*