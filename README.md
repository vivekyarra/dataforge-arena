# DataForge Arena

> **Self-improving data repair agents trained in adversarial environments.**

Built for the [Meta PyTorch OpenEnv AI Hackathon 2026](https://pytorch.org/event/openenv-ai-hackathon/)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-10b981?style=for-the-badge)](https://github.com/huggingface/openenv)
[![TRL GRPO](https://img.shields.io/badge/TRL-GRPO_Training-f59e0b?style=for-the-badge)](https://huggingface.co/docs/trl/main/en/grpo)
[![Tests](https://img.shields.io/badge/Tests-28%2F28%20Passing-10b981?style=for-the-badge)](#)

---

## The Problem Nobody Solved

**$12.9 million per year.** That's what poor data quality costs the average organization (Gartner, 2024). Every enterprise has the same story: corrupted fields, broken foreign keys, phantom duplicates — caught by brittle regex pipelines that break the moment the schema changes.

LLMs can write code, pass bar exams, and generate artwork. But ask one to look at a corrupted patient record with a null `age` field, a swapped `department_id`, and a duplicated row with a mutated email — and it hallucinates. It picks the wrong tool. It doesn't even notice the duplicate.

**No benchmark exists to train this skill.** Until now.

## What DataForge Arena Does

We built a system powered by **PyTorch**, **TRL**, and **OpenEnv** featuring two adversarial agents locked in an infinite curriculum.

```
    CORRUPTOR                              SURGEON
    (Rule-Based)                           (Live LLM + GRPO)
         │                                      │
         │   "Break this data."                 │   "Fix it."
         ▼                                      ▼
    ┌─────────────────────────────────────────────────┐
    │              DataForge Environment               │
    │                                                  │
    │   Clean Dataset ──▶ Corrupted ──▶ Repaired?     │
    │                                                  │
    │   6 reward signals   │   Solvability gate        │
    │   Soft-delete invariant │   KL regularization    │
    └─────────────────────────────────────────────────┘
```

The **CORRUPTOR** uses 7 sabotage tools across 3 difficulty tiers to inject realistic data errors. The **SURGEON** (a PyTorch-native LLM fine-tuned with TRL GRPO) diagnoses each corruption and selects from 8 repair tools. As the Surgeon improves, the Corruptor escalates. The environment never runs out of challenge.

---

## Results

| Metric | Value |
|--------|-------|
| **Reward at step 0** | **-1.85** |
| **Reward at step 80** | **+1.18** |
| **Total improvement** | **+3.03 (+164%)** |
| **JSON parse success rate** | **97.5%** (39/40 by step 50) |
| **Format error elimination** | **100%** (CORRECT_FORMAT tool) |
| **Surgeon vs random advantage** | **+0.037 accuracy delta** |
| **Test suite** | **28/28 passing** |

> The 97.5% JSON parse success rate is the most significant signal. Under RL pressure the model is simultaneously learning *what to do* AND *how to format its output*. Maintaining near-perfect structured output by step 50 means the policy is genuinely converging.

---

## Why It Matters

| What Exists | What We Built |
|-------------|---------------|
| Text benchmarks (GLUE, MMLU) | **Data quality benchmark** — tests reasoning over structured tabular data |
| Static datasets | **Dynamic adversarial curriculum** — difficulty scales with agent capability |
| LLM-as-judge (slow, expensive) | **Heuristic reward computer** — 45s/step on T4, not 5 min |
| Fixed corruption patterns | **Solvability-gated episodes** — every episode is guaranteed learnable |

## Architecture & Technology Stack

- **PyTorch**: Scalable tensor operations and model backbone.
- **TRL (Transformer Reinforcement Learning)**: Handles the GRPO training loop, ensuring mathematically sound policy updates.
- **OpenEnv**: Environment standardization ensuring our environment can plug-and-play with any RL framework.
- **FastAPI / Gradio**: A robust backend serving the environment and a "Billion-Dollar" frontend visualizing the live inference.

### Adversarial Curriculum (3 Tiers)

| Tier | Epochs | What the Corruptor Does | What the Surgeon Must Learn |
|------|--------|------------------------|---------------------------|
| **1** | 0–49 | Single null injection, type errors (`ERR_42`) | Basic imputation, type detection |
| **2** | 50–99 | Null clusters, date format swaps, cross-field inconsistencies | Pattern recognition, multi-cell correlation |
| **3** | 100+ | Foreign key violations, duplicate rows with mutation | Relational reasoning, merge/delete decisions |

Tier transitions use a **10-epoch warmup blend** with **5× higher KL beta** to prevent catastrophic forgetting when the distribution shifts.

## Quick Start

```bash
git clone https://github.com/vivekyarra/dataforge-arena.git
cd dataforge-arena
pip install -r requirements.txt
python training/generate_data.py

# Verify everything works
pytest tests/test_all.py -v    # 28 tests, all green

# Train the Surgeon via GRPO
python training/train_grpo.py

# Launch the Tactical Demo (Live Inference & Baselines)
python demo/app.py
```

## OpenEnv Compliance

DataForge Arena implements the [OpenEnv](https://github.com/huggingface/openenv) `Env` interface:

```python
class DataForgeEnv(BaseEnv):
    def reset(self) -> DataForgeObservation:
        """Generate a fresh corrupted episode."""
    def step(self, action: SurgeonAction) -> tuple[Observation, dict, bool, dict]:
        """Apply a repair tool and return reward signals."""
```

The environment exposes a **FastAPI server** with CORS support and interactive Swagger docs:

```
GET  /health   → {"status": "ok", "difficulty": 2, "epoch": 73}
GET  /info     → Full environment metadata and available tools
GET  /docs     → Interactive Swagger UI
POST /reset    → DataForgeObservation
POST /step     → {observation, reward, done, info}
```

---

> **Built for the [Meta PyTorch OpenEnv AI Hackathon 2026](https://pytorch.org/event/openenv-ai-hackathon/)**
>
> MIT License
