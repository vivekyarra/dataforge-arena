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

## Where GRPO fits

The training path uses TRL GRPO to optimize a language-model surgeon over structured repair actions. The prompt asks for valid JSON only, the parser hardens the boundary between generated text and environment actions, and the reward loop evaluates the actual outcome of each tool call.

The intent is not to reward fluent explanations. The efficiency term now adds a small positive signal when a repair tool targets an actually incorrect cell, so target selection is tied to fixing data rather than writing persuasive reasoning.

## What the public repo proves today

This repo is evidence-first. The current committed artifacts show:

| Evidence | Current value |
|----------|---------------|
| OpenEnv-compatible environment | `reset`, `step`, FastAPI endpoints |
| Committed evaluation mode | `grpo` |
| Heuristic surgeon avg accuracy delta | `+0.0010` |
| Heuristic advantage over random | `+0.0053` (`+0.53 pp`) |
| GRPO checkpoint avg accuracy delta | `-0.0004` |
| Matching random baseline avg accuracy delta | `-0.0045` |
| GRPO checkpoint advantage | `+0.0041` (`+0.41 pp`) |
| Logged GRPO curriculum tiers | `1, 2, 3` |
| Mean logged parse success | `40.00%` |
| Parse success first to final | `25% -> 50%` |
| Test suite | `58 passed` via `python -m pytest -q` |

One important note: the public repo does not commit the local checkpoint directory because `outputs/` is ignored. The committed GRPO numbers come from the Colab-trained checkpoint at `outputs/dataforge-surgeon`. The demo and evaluation harness expose live GRPO mode only when that checkpoint exists locally.

## Final Colab evidence

The final short-run training artifact is honest about both progress and limits:

| Artifact | Value |
|----------|-------|
| GPU | Tesla T4 |
| Training target / final logged step | `80` target steps / last logged step `75` |
| First -> final logged reward | `-1.4000 -> -1.4000` |
| Best logged reward | `-0.2000` |
| Smoothed reward, first 3 rows -> last 3 rows | `-1.2000 -> -1.0000` |
| Parse success, first -> final | `25% -> 50%` |
| GRPO advantage over random | `+0.0041` (`+0.41 pp`) |

The trained 1.5B checkpoint does not beat the heuristic surgeon in this short T4 run. The right reading is narrower but still useful: the model learns enough structure to become less destructive than random, while the heuristic surgeon remains the stronger demo policy.

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

## Reproduce it

```bash
git clone https://github.com/vivekyarra/dataforge-arena.git
cd dataforge-arena
pip install -r requirements.txt

python training/generate_data.py
python -m pytest -q
python eval/evaluate.py --agent-mode heuristic --episodes 20 --tier 1 --steps 5 --seed 7
python demo/app.py
```

After training and saving a checkpoint:

```bash
python eval/evaluate.py --agent-mode grpo --model-path outputs/dataforge-surgeon
```

## Links

| Resource | URL |
|----------|-----|
| Live HF Space | https://huggingface.co/spaces/Vivek567/enterprise-data-cleaning-env |
| Colab Notebook | DataForge_Arena_Colab.ipynb |
| GitHub | https://github.com/vivekyarra/dataforge-arena |

Built with PyTorch, TRL GRPO, OpenEnv, Hugging Face, and a stubborn belief that agents should be graded by what they actually fix.
