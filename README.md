---
title: DataForge Arena
emoji: "🔬"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: true
---

## Changelog

### v1.1 (post-audit fixes)
- Fixed the Colab notebook import cell to use live schema exports and removed a stale observation print.
- Logged reward exceptions in `constraint_alignment`, `schema_alignment`, and `outlier_targeting`.
- Changed `null + CORRECT_FORMAT` from zero reward to `-1.5` to break tool-collapse.
- Preserved positive null-imputation rewards by keeping the impute checks ahead of format penalties.
- Added inline anti-collapse penalty for null-cell `CORRECT_FORMAT` actions.
- Lowered corruptor escalation gates to `40/120` with `5`-step escalation and `3`-step de-escalation.
- Aligned `openenv.yaml`, `CITATION.cff`, Docker, and README evidence with the current code and artifacts.

### v1.0 (hackathon submission)
- Initial GRPO environment and training run on Qwen 2.5 1.5B.
- 9.8x less destructive than random baseline in committed GRPO evaluation.
- 100% parse success sustained in the committed training log.

# DataForge Arena

Enterprise data repair as an OpenEnv world-modeling benchmark. The agent observes a corrupted table, infers why a cell is wrong, picks a repair tool, and earns reward only when the repair improves ground-truth accuracy.

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-10b981?style=for-the-badge)](https://github.com/huggingface/openenv)
[![TRL GRPO](https://img.shields.io/badge/TRL-GRPO_Trained-f59e0b?style=for-the-badge)](https://huggingface.co/docs/trl/main/en/grpo)
[![Tests](https://img.shields.io/badge/Tests-127_Passing-10b981?style=for-the-badge)](./tests)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vivekyarra/dataforge-arena/blob/master/DataForge_Arena_Colab.ipynb)

[GitHub](https://github.com/vivekyarra/dataforge-arena) · [HF Space](https://huggingface.co/spaces/Vivek567/dataforge-arena) · [Browser Demo](./artifacts/browser_simulator.html) · [Colab Notebook](./DataForge_Arena_Colab.ipynb)

Built for the Meta x PyTorch x Hugging Face x Scaler OpenEnv Hackathon 2026, Theme 3.1: World Modeling.

## The problem

LLMs can read tabular data, but that is not the same thing as understanding it. A model that sees `age=145`, `birth_year=1979`, and `department_id=500` still needs a causal model of range constraints, temporal consistency, and foreign-key structure before it can repair the table correctly.

DataForge Arena turns that into an RL task. The reward is grounded in cell-level accuracy delta against clean data, and the shaped signals only pay out when the policy identifies the right violation type, chooses the right tool for the column, and targets a real anomaly. There is no LLM judge in the loop.

## Environment

The benchmark ships with two schemas:
- `healthcare`
- `financial`

The corruptor injects seven learnable corruption types across three tiers:
- Tier 1: `inject_null_single`, `inject_type_error`, `enum_substitution`
- Tier 2: `inject_null_cluster`, `swap_date_format`, `inject_out_of_range_age`, `semantic_temporal_drift`, `currency_unit_mismatch`
- Tier 3: `break_foreign_key`, `duplicate_row_mutate`

The agent acts in an 8-tool repair space:
- `IMPUTE_MEDIAN`
- `IMPUTE_MODE`
- `IMPUTE_FORWARD_FILL`
- `CORRECT_FORMAT`
- `DELETE_ROW`
- `MERGE_DUPLICATE`
- `FLAG_UNCERTAIN`
- `NO_OP`

## Reward design

The reward function combines one primary signal with six shaped signals:

| Signal | Meaning | Max |
|---|---|---:|
| `accuracy_delta x 50` | Ground-truth improvement | bounded |
| `constraint_alignment` | Correct violation-type reasoning | +3.0 |
| `schema_alignment` | Tool matches column type | +2.0 |
| `outlier_targeting` | Cell is a real anomaly | +0.5 |
| `reasoning_quality` | Justification names the causal chain | +1.5 |
| `parse_bonus` | Valid structured action | +0.5 |
| `anti_hack` | Penalizes reward hacking | -5.0 |

Constraint alignment carries the largest shaped reward because the environment is trying to train causal diagnosis, not just syntactic cleanup.

## Results

### Committed evidence

All values below are copied from committed artifacts.

| Artifact | Metric | Value |
|---|---|---:|
| `eval/results.json` | GRPO average accuracy delta | `-0.0005` |
| `eval/results.json` | Random average accuracy delta | `-0.0049` |
| `eval/results.json` | GRPO advantage over random | `+0.0044` |
| `eval/results.json` | GRPO win rate | `0.05` |
| `eval/results.json` | GRPO destruction ratio | `0.102` |
| `eval/results.json` | GRPO improvement vs random | `89.8` |
| `eval/heuristic_results.json` | Heuristic average accuracy delta | `-0.000138` |
| `eval/heuristic_results.json` | Random average accuracy delta | `-0.009325` |
| `eval/heuristic_results.json` | Heuristic advantage over random | `+0.009188` |
| `eval/heuristic_results.json` | Heuristic win rate | `0.1` |
| `eval/heuristic_results.json` | Constraint alignment rate | `0.0875` |
| `eval/heuristic_results.json` | Schema alignment rate | `0.105` |
| `eval/heuristic_results.json` | Heuristic destruction ratio | `0.0147` |
| `logs/training_log.csv` | First reward | `1.925` |
| `logs/training_log.csv` | Final reward | `4.475` |
| `logs/training_log.csv` | Best reward | `6.95` |
| `logs/training_log.csv` | Parse success mean | `1.0` |
| `python -m pytest tests/ -x -q` | Test suite | `127 passed` |

The committed training summary shows `+132%` total reward improvement from `1.925` to `4.475`. Note: this increase is driven by parse shaping and contextual bonuses; constraint-grounded signals are pending a full rerun after the v1.1 reward fixes.

### Judge evidence

The current committed GRPO checkpoint is still early. The honest story is:
- it is 9.8x less destructive than random (`destruction_ratio = 0.102`)
- it achieves a `5%` win rate on the committed tier-1 evaluation
- it maintains `100%` parse success across the committed training log

What the training log does support is structured-output stability and total reward improvement. What it does not yet support is a strong committed `constraint_alignment` curve; that signal path is now hardened in code and ready for the next training run.

### Schema generalization

The committed heuristic evaluation runs with `schema = "both"` and includes per-schema breakdowns for healthcare and financial tables in `eval/heuristic_results.json`.

## Why this fits Theme 3.1

This environment is about world models, not surface formatting. To earn reward, the agent must maintain an internal model of:
- column types and nullable constraints
- enum domains
- foreign-key consistency
- temporal implications such as `birth_year -> age`
- statistical distributions and outliers

That is exactly the kind of persistent causal reasoning Theme 3.1 asks for.

## OpenEnv API

```text
GET  /health
GET  /info
POST /reset
POST /step
GET  /metrics
GET  /docs
```

The API server lives in [environment/server.py](/D:/dataforge-arena/environment/server.py).

## Quick start

```bash
git clone https://github.com/vivekyarra/dataforge-arena.git
cd dataforge-arena
pip install -r requirements.txt
python -m pytest tests/ -x -q
python environment/server.py
```

For training on Colab, open [DataForge_Arena_Colab.ipynb](/D:/dataforge-arena/DataForge_Arena_Colab.ipynb).

## Repository map

| Path | Purpose |
|---|---|
| [`environment/`](./environment) | Env, corruptor, reward, tools, FastAPI server |
| [`training/`](./training) | Prompting, parser, model config, GRPO support code |
| [`eval/`](./eval) | Evaluation harness and committed result artifacts |
| [`demo/`](./demo) | Gradio demo implementation |
| [`logs/`](./logs) | Training CSVs, plots, and summaries |
| [`tests/`](./tests) | Regression suite |

## Citation

If you use the environment in research or demos, see [CITATION.cff](/D:/dataforge-arena/CITATION.cff).
