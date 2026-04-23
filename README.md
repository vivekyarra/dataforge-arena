# DataForge Arena

> **The first adversarial RL environment where LLMs learn to diagnose and repair corrupted enterprise data — by playing against themselves.**

Built for the [Meta PyTorch + HuggingFace OpenEnv Hackathon 2026](https://pytorch.org/).

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-10b981?style=for-the-badge)](https://github.com/huggingface/openenv)
[![GRPO](https://img.shields.io/badge/Training-GRPO-f59e0b?style=for-the-badge)](https://arxiv.org/abs/2402.03300)
[![Tests](https://img.shields.io/badge/Tests-28%2F28%20Passing-10b981?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

---

## The Problem Nobody Solved

**$12.9 million per year.** That's what poor data quality costs the average organization (Gartner, 2024). Every enterprise has the same story: corrupted fields, broken foreign keys, phantom duplicates — caught by brittle regex pipelines that break the moment the schema changes.

LLMs can write code, pass bar exams, and generate artwork. But ask one to look at a corrupted patient record with a null `age` field, a swapped `department_id`, and a duplicated row with a mutated email — and it hallucinates. It picks the wrong tool. It doesn't even notice the duplicate.

**No benchmark exists to train this skill.** Until now.

## What DataForge Arena Does

Two adversarial agents. One dataset. An infinite curriculum.

```
    CORRUPTOR                              SURGEON
    (Rule-Based)                           (LLM + GRPO)
         │                                      │
         │   "Break this data."                  │   "Fix it."
         ▼                                      ▼
    ┌─────────────────────────────────────────────────┐
    │              DataForge Environment               │
    │                                                  │
    │   Clean Dataset ──▶ Corrupted ──▶ Repaired?     │
    │                                                  │
    │   6 reward signals   │   Solvability gate        │
    │   Soft-delete invariant │   Dynamic KL beta      │
    └─────────────────────────────────────────────────┘
```

The **CORRUPTOR** uses 7 sabotage tools across 3 difficulty tiers to inject realistic data errors. The **SURGEON** (an LLM fine-tuned with GRPO) diagnoses each corruption and selects from 8 repair tools. As the Surgeon improves, the Corruptor escalates. The environment never runs out of challenge.

## Why It Matters

| What Exists | What We Built |
|-------------|---------------|
| Text benchmarks (GLUE, MMLU) | **Data quality benchmark** — tests reasoning over structured tabular data |
| Static datasets | **Dynamic adversarial curriculum** — difficulty scales with agent capability |
| Single reward signal | **6-signal multi-objective reward** — accuracy, tool logic, reasoning, efficiency, anti-hack |
| LLM-as-judge (slow, expensive) | **Heuristic reward computer** — 45s/step on T4, not 5 min |
| Fixed corruption patterns | **Solvability-gated episodes** — every episode is guaranteed learnable |

## Architecture

### Adversarial Curriculum (3 Tiers)

| Tier | Epochs | What the Corruptor Does | What the Surgeon Must Learn |
|------|--------|------------------------|---------------------------|
| **1** | 0–49 | Single null injection, type errors (`ERR_42`) | Basic imputation, type detection |
| **2** | 50–99 | Null clusters, date format swaps, cross-field inconsistencies | Pattern recognition, multi-cell correlation |
| **3** | 100+ | Foreign key violations, duplicate rows with mutation | Relational reasoning, merge/delete decisions |

Tier transitions use a **10-epoch warmup blend** (30%→100% probability) with **dynamic KL beta** (5× higher during transitions) to prevent catastrophic forgetting.

### Multi-Objective Reward Function

```python
total_reward = (
    accuracy_delta * 20        # Did your fix actually improve the data?
  + accuracy_absolute * 2      # How close to perfect are we?
  + tool_logic                 # Did you pick the right tool for this error type?
  + reasoning_quality          # Did you explain your diagnosis?
  + efficiency                 # Penalty for modifying correct cells
  + anti_hack                  # Penalty for gaming via mass soft-delete
)
```

**No LLM-as-Judge.** Reasoning quality is scored via keyword heuristics — fast enough for RL-scale training.

### Three Design Invariants

1. **Solvability Gate.** Every generated episode is validated. Banned corruptions (full row deletion, entire-column null) are rejected. Episodes retry up to 10× to ensure the Surgeon *can* learn from every training step.

2. **Soft-Delete.** Rows are never physically removed. A `_is_deleted` flag preserves indices, preventing the cascading index drift that silently poisons reward calculations in mutable-length DataFrames.

3. **Independent Rollouts.** Each GRPO rollout (`G` completions per prompt) evaluates on a freshly reset environment. No shared mutable state between rollouts — critical for correct advantage estimation.

## Quick Start

```bash
git clone https://github.com/vivekyarra/dataforge-arena.git
cd dataforge-arena
pip install -r requirements.txt
python training/generate_data.py

# Verify everything works
pytest tests/test_all.py -v    # 28 tests, all green

# Train (auto-detects GPU tier)
python training/train_grpo.py

# Evaluate
python eval/evaluate.py --episodes 20 --tier 1

# Interactive demo
python demo/app.py
```

## GPU Auto-Selection

The training script detects your hardware and selects the optimal model automatically:

| GPU | VRAM | Model | Speed | Training Time |
|-----|------|-------|-------|---------------|
| T4 | 15 GB | Qwen 2.5 1.5B (4-bit) | 45s/step | ~60 min |
| A10G / L4 | 20+ GB | Llama 3.2 3B (4-bit) | 55s/step | ~90 min |
| A100 | 40+ GB | Llama 3.1 8B (4-bit) | 40s/step | ~120 min |

## OpenEnv Compliance

DataForge Arena implements the [OpenEnv](https://github.com/huggingface/openenv) `Env` interface:

```python
class DataForgeEnv(BaseEnv):
    def reset(self) -> DataForgeObservation:
        """Generate a fresh corrupted episode."""
    def step(self, action: SurgeonAction) -> tuple[Observation, dict, bool, dict]:
        """Apply a repair tool and return reward signals."""
```

The environment also exposes a **FastAPI server** for remote interaction:

```
GET  /health              → {"status": "ok", "difficulty": 2, "epoch": 73}
POST /reset               → DataForgeObservation
POST /step  {action}      → {observation, reward, done, info}
```

## Project Structure

```
dataforge-arena/
├── environment/
│   ├── env.py              # DataForgeEnv (OpenEnv BaseEnv interface)
│   ├── corruptor.py        # 3-tier adversarial episode generator
│   ├── reward.py           # 6-signal multi-objective reward computer
│   ├── tools.py            # 8 SURGEON tool implementations
│   ├── schemas.py          # Data schemas + tool definitions
│   └── server.py           # FastAPI server for HF Spaces
├── training/
│   ├── train_grpo.py       # GRPO training loop (TRL + Unsloth)
│   ├── model_config.py     # GPU-aware model auto-selector
│   ├── prompt.py           # System prompt engineering
│   ├── parser.py           # Robust 3-strategy JSON action parser
│   ├── logger.py           # CSV logger + collapse detection
│   └── generate_data.py    # Synthetic healthcare/financial data
├── eval/
│   └── evaluate.py         # Before/after evaluation harness
├── demo/
│   └── app.py              # Gradio tactical demo UI
├── tests/
│   └── test_all.py         # 28 comprehensive tests
├── DataForge_Arena_Colab.ipynb  # One-click Colab training
├── Dockerfile              # HF Spaces deployment
└── requirements.txt
```

## What Makes This Different

Most hackathon projects are wrappers around an API call. DataForge Arena is a **complete system**:

- **Environment**: Adversarial curriculum with solvability guarantees
- **Training**: GRPO with dynamic KL scheduling and collapse detection
- **Evaluation**: Automated before/after accuracy benchmarking
- **Deployment**: FastAPI server + Gradio demo + Docker + Colab notebook
- **Testing**: 28 tests covering every component and every bug fix

The Surgeon doesn't just fix data — it *learns to reason about why data is broken*, selecting the right tool for the right error type, and explaining its diagnosis. That's the skill gap no existing benchmark addresses.

---

> **Built for the Meta PyTorch + HuggingFace OpenEnv Hackathon 2026**
>
> MIT License
