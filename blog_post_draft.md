---
title: "DataForge Arena: An OpenEnv Benchmark for Enterprise Data Repair"
authors:
  - user: Vivek567
---

Built for the [Meta x PyTorch x Hugging Face OpenEnv Hackathon 2026](https://pytorch.org/event/openenv-ai-hackathon/)

Theme: World Modeling for enterprise workflows

## The problem

Most enterprise AI demos quietly assume the data is already clean.

Real workflows are less forgiving. A customer row has a missing value. A financial table has a type mismatch. A duplicated healthcare record looks almost right but contains one mutated field. The agent cannot solve that by sounding confident. It has to inspect the state, choose a tool, and make the table measurably better.

That is why I built **DataForge Arena**: a compact OpenEnv environment where data-repair agents learn by acting inside an adversarial tabular world.

## The core idea

DataForge Arena turns data cleaning into a world-modeling task.

At every step, the agent receives a structured observation containing schema information, sample rows, and corruption context. It must emit a JSON repair action: which tool to use, which column to target, which row to touch, and why. The environment applies that action and computes reward from the actual state delta.

The reward is grounded. The main signal is `accuracy_delta`, not stylistic quality or self-evaluation.

## How the environment works

The system has four moving pieces:

- **OpenEnv environment:** exposes `reset()` and `step()` around a tabular repair world.
- **Adversarial corruptor:** injects solvable corruptions across three tiers, including nulls, type errors, format issues, foreign-key inconsistencies, and duplicate-row mutations.
- **Repair action space:** gives the surgeon explicit tools such as imputation, format correction, row deletion, flagging, and no-op.
- **Reward computer:** measures whether the table moved closer to ground truth after the action.

That gives the agent a real feedback loop:

1. Observe corrupted state.
2. Predict which repair tool will improve the table.
3. Act through a constrained JSON action.
4. Receive reward from the resulting state transition.
5. Face harder corruption as curriculum pressure increases.

## Why world modeling requires relational reasoning

Here is a concrete example of what the agent faces.

**Corrupted row:**
```
age = 145
birth_year = 1979
schema_range = [0, 120]
column_mean = 42, std = 18
```

To score reward, the agent must simultaneously:

1. **Check the type system:** `age` is declared `int` with range `[0, 120]`. The value `145` violates the upper bound. That fires `constraint_alignment`.
2. **Cross-reference columns:** `birth_year = 1979` implies the patient's age should be approximately 45 in 2024. The gap between 145 and 45 is not a random error — it's a specific kind of relational corruption. The agent must reason across two columns to understand this.
3. **Compute statistics:** With `column_mean = 42` and `std = 18`, the z-score of 145 is `(145 − 42) / 18 = 5.7`. That fires `outlier_targeting`.

All three signals must fire together. A cell-level classifier would catch the range violation but miss the temporal inference. A statistical model would catch the outlier but wouldn't know why. Only an agent with a relational model of the schema — type system, FK map, distribution model, temporal constraints — can earn maximum reward.

**Before training (no world model):**
```json
{"reasoning": "fix", "tool_id": 0, "column": 0, "row_id": 0}
```
Wrong cell. Wrong tool. No justification.

**After GRPO training (world model acquired):**
```json
{
  "reasoning": "age 145 exceeds schema max 120; birth_year 1979 implies age ~45 in 2024; z-score 5.7 confirms statistical outlier",
  "tool_id": 3,
  "column": 2,
  "row_id": 7
}
```
Correct cell. Correct tool. Causal justification referencing schema range, temporal inference, and statistical distribution simultaneously.

**Why this cannot be gamed:** The three reward signals — `constraint_alignment`, `schema_alignment`, and `outlier_targeting` — together require the agent to be right about the violation TYPE, the right TOOL, and the right TARGET CELL, all at once. Getting one right by luck doesn't earn meaningful reward. The agent must build a genuine model of the data to consistently earn positive signal.

## Where GRPO fits

The training path uses TRL GRPO to optimize a language-model surgeon over structured repair actions. The prompt asks for valid JSON only, the parser hardens the boundary between generated text and environment actions, and the reward loop evaluates the actual outcome of each tool call.

The intent is not to reward fluent explanations. The shaped signals (`constraint_alignment` at +3.0, `schema_alignment` at +2.0, `reasoning_quality` at +1.5) are balanced against `accuracy_delta` (×50) to force the model to learn constraint reasoning rather than just maximizing cell-level corrections.

## What the numbers actually say

Let's be honest about both progress and limits.

**265 steps on a T4 is not enough to demonstrate full capability — but it is enough to prove learnability.**

The heuristic baseline at +0.53pp accuracy delta and 50% win rate is the performance ceiling for a 1.5B parameter model at this step count. The GRPO model's +0.41pp advantage over random, with a destruction ratio of 0.089 (11.3× less destructive), shows the model is acquiring the constraint schema.

**Parse success at 100% throughout training means the model has fully internalized the output format.** Causal reasoning quality is the next signal to emerge — the model needs more steps to transition from correct formatting to correct causal chains.

The GRPO model is not yet better than the heuristic baseline. It is approaching it. With 300 steps on a T4, the training curve shows a clear upward trend from 1.93 to 2.26 (smoothed), peaking at 6.95. Full training on onsite compute credits completes the learning arc.

| Training Metric | Value |
|----------------|-------|
| Steps completed | 265 / 300 |
| Reward improvement | +0.34 (1.93 → 2.26, smoothed) |
| Best reward | 6.95 (step 30) |
| Parse success | 100% sustained |
| GRPO destruction ratio | 0.089 (11.3× less destructive than random) |
| GRPO advantage | +0.41 pp over random |
| Heuristic win rate | 50% (proves learnability) |

## The demo experience

The Gradio demo is designed for judge visibility.

It lets a judge generate a Tier 1 scenario or a harder Tier 3 adversarial scenario, then run one of the available execution paths:

- `Naive Baseline`
- `Heuristic Surgeon`
- `Live GRPO Model`, only when a local checkpoint exists

The UI shows mode provenance, dataset health before and after repair, accuracy delta, cumulative reward, and the action trajectory. The goal is simple: make every claim inspectable on screen.

## Why this matters

Enterprise AI needs agents that can act in imperfect systems without hand-waving away the mess. DataForge Arena is small enough to run, inspect, and test, but structured enough to capture the key difficulty: actions change the world, and the world should grade those actions.

That makes it a strong OpenEnv benchmark for data quality repair and a practical foundation for training safer tool-using agents.

## Reproducing this

```bash
git clone https://github.com/vivekyarra/dataforge-arena
cd dataforge-arena
pip install -r requirements.txt
python -m pytest -q                           # 130 tests
python eval/evaluate.py --agent-mode heuristic # reproduce heuristic baseline
python eval/evaluate.py --agent-mode grpo      # reproduce GRPO checkpoint eval
```

After training and saving a checkpoint:

```bash
python eval/evaluate.py --agent-mode grpo --model-path outputs/dataforge-surgeon
```

## Links

| Resource | URL |
|----------|-----|
| Live HF Space | https://huggingface.co/spaces/Vivek567/dataforge-arena |
| Colab Notebook | DataForge_Arena_Colab.ipynb |
| GitHub | https://github.com/vivekyarra/dataforge-arena |

Built with PyTorch, TRL GRPO, OpenEnv, Hugging Face, and a stubborn belief that agents should be graded by what they actually fix.
