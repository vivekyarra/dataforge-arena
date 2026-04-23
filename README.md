# 🏟️ DataForge Arena

**The first adversarial RL environment where LLMs learn to repair corrupted enterprise data through self-play.**

Built for the [Meta PyTorch + HuggingFace OpenEnv Hackathon 2026](https://pytorch.org/).

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-10b981?style=for-the-badge)](https://github.com/huggingface/openenv)
[![GRPO](https://img.shields.io/badge/Training-GRPO-f59e0b?style=for-the-badge)](https://arxiv.org/abs/2402.03300)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

---

## 🎯 Problem

**25% of enterprise data contains quality errors** (Gartner, 2024). Current fix: brittle regex pipelines that break on every schema change. No existing benchmark tests an LLM's ability to *reason about and repair* real-world data corruption patterns.

## 💡 Solution

DataForge Arena is a self-improving RL environment with two adversarial agents:

| Agent | Role | Tools |
|-------|------|-------|
| **CORRUPTOR** 🔴 | Injects realistic data errors across 3 difficulty tiers | 7 sabotage tools (null injection, type errors, FK violations, duplicate rows) |
| **SURGEON** 🟢 | Diagnoses and repairs corrupted cells | 8 repair tools (impute, format, delete, merge, flag, no-op) |

The CORRUPTOR creates progressively harder episodes. The SURGEON learns to fix them via **GRPO** (Group Relative Policy Optimization). As the SURGEON improves, the CORRUPTOR escalates — creating an infinite curriculum.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DataForge Arena                        │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │CORRUPTOR │───▶│ DataForgeEnv │◀───│   SURGEON     │  │
│  │ 3-tier   │    │  (OpenEnv)   │    │  (LLM Agent)  │  │
│  │curriculum│    │              │    │  Qwen/Llama   │  │
│  └──────────┘    │  ┌────────┐  │    └───────────────┘  │
│                  │  │ Reward │  │                        │
│                  │  │Computer│  │    ┌───────────────┐  │
│                  │  │6 signals│ │    │  GRPO Trainer  │  │
│                  │  └────────┘  │    │  (TRL+Unsloth) │  │
│                  └──────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Corruption Tiers (Adversarial Curriculum)
| Tier | Epoch | Corruptions | Difficulty |
|------|-------|-------------|------------|
| 🟢 Tier 1 | 0-49 | Single null, type errors | Warm-up |
| 🟡 Tier 2 | 50-99 | Null clusters, date swaps, cross-field | Intermediate |
| 🔴 Tier 3 | 100+ | FK violations, duplicate rows w/ mutation | Expert |

### Multi-Objective Reward (6 Signals)
```
R = accuracy_delta × 20        # Did the fix improve field-level accuracy?
  + accuracy_absolute × 2      # How close to perfect?
  + tool_logic                  # Is the tool appropriate for this error type?
  + reasoning_quality           # Does the agent explain its diagnosis?
  + efficiency                  # Penalty for modifying correct cells
  + anti_hack                   # Penalty for mass soft-delete gaming
```

## 🔬 Key Design Decisions

1. **Solvability Gate**: Every generated episode is validated — banned tools (whole-row deletion, full-column null) are rejected. Episodes retry up to 10× to ensure the SURGEON *can* learn.

2. **Soft-Delete Invariant**: Rows are never physically removed. A `_is_deleted` flag preserves indices, preventing cascading index drift that poisons reward calculations.

3. **Heuristic Reward (No LLM-as-Judge)**: Reasoning quality is scored via keyword matching, not a second LLM call. This keeps training speed at 45s/step on T4 instead of 5min/step.

4. **Dynamic KL Beta**: During tier transitions (epochs 50-60, 100-110), KL penalty rises 5× to prevent catastrophic forgetting when the distribution shifts.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/vivekyarra/dataforge-arena.git
cd dataforge-arena
pip install -r requirements.txt
python training/generate_data.py
```

### Run Tests
```bash
pytest tests/test_all.py -v  # 28 tests, all pass
```

### Train with GRPO
```bash
# Auto-detects GPU tier and selects appropriate model
python training/train_grpo.py
```

### Deploy Environment (FastAPI)
```bash
python environment/server.py
# Health: GET  /health
# Reset:  POST /reset
# Step:   POST /step  {"reasoning": "...", "tool_id": 0, "column": 1, "row_id": 3}
```

### Interactive Demo
```bash
python demo/app.py  # Gradio UI on port 7860
```

## 📊 GPU Tier Auto-Selection

| GPU | VRAM | Model | Rollouts (G) | Steps | Est. Time |
|-----|------|-------|-------------|-------|-----------|
| T4 | 15GB | Qwen 2.5 1.5B | 4 | 80 | ~60 min |
| A10G / L4 | 20GB+ | Llama 3.2 3B | 6 | 100 | ~90 min |
| A100 | 40GB+ | Llama 3.1 8B | 8 | 150 | ~120 min |

## 📁 Project Structure

```
dataforge-arena/
├── environment/
│   ├── env.py              # DataForgeEnv (OpenEnv BaseEnv)
│   ├── corruptor.py        # 3-tier adversarial episode generator
│   ├── reward.py           # 6-signal multi-objective reward
│   ├── tools.py            # 8 SURGEON tool implementations
│   ├── schemas.py          # Data schemas + tool definitions
│   └── server.py           # FastAPI wrapper for HF Spaces
├── training/
│   ├── train_grpo.py       # Main GRPO training script
│   ├── model_config.py     # GPU-aware model selector
│   ├── prompt.py           # System prompt + one-shot example
│   ├── parser.py           # Robust JSON action parser (3 strategies)
│   ├── logger.py           # CSV training logger + collapse detection
│   └── generate_data.py    # Synthetic dataset generator
├── demo/
│   └── app.py              # Gradio tactical UI
├── tests/
│   └── test_all.py         # 28 tests covering all components
├── eval/
│   └── evaluate.py         # Before/after evaluation harness
├── data/                   # Clean ground truth datasets
├── Dockerfile              # HF Spaces deployment
└── requirements.txt        # Full dependency list
```

## 🔧 OpenEnv Compliance

DataForge Arena implements the [OpenEnv](https://github.com/huggingface/openenv) `Env` interface:

```python
from openenv.env import Env as BaseEnv

class DataForgeEnv(BaseEnv):
    def reset(self) -> DataForgeObservation: ...
    def step(self, action: SurgeonAction) -> tuple[Observation, RewardDict, bool, dict]: ...
```

The environment is fully stateless across episodes and exposes a FastAPI server for remote interaction.

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with 🔥 for the Meta PyTorch + HuggingFace OpenEnv Hackathon 2026</strong>
</p>
