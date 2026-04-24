# DataForge Arena: Teaching LLMs to Fix Broken Enterprise Data

**Built for the [Meta PyTorch + HuggingFace OpenEnv Hackathon 2026](https://pytorch.org/event/openenv-ai-hackathon/)**

## The Problem

25% of enterprise data contains quality errors (Gartner, 2024). Nulls, type mismatches, broken foreign keys, phantom duplicates -- caught today by brittle regex pipelines. No existing benchmark trains LLMs to *reason about and repair* structured data corruption.

## What We Built

**DataForge Arena** is an adversarial RL environment with two agents:

- **CORRUPTOR** (rule-based): Injects realistic errors across 3 difficulty tiers -- from simple nulls to FK violations and mutated duplicates
- **SURGEON** (LLM + GRPO): Learns to diagnose corruptions and select from 8 repair tools

The environment is [OpenEnv](https://github.com/huggingface/openenv)-compliant with a 6-signal multi-objective reward function, solvability-gated episodes, and a soft-delete invariant that prevents index drift.

## Training Results

We trained Qwen 2.5 1.5B (4-bit) on a T4 GPU using GRPO for 80 steps:

| Metric | Value |
|--------|-------|
| **Reward trajectory** | -1.8 to +1.55 over 80 steps |
| **JSON parse success** | 97.5% (robust 3-strategy parser) |
| **Corruption tiers** | 3 (auto-escalating) |
| **Training time** | ~60 min on Colab T4 |

*(Replace with your actual numbers from training_log.csv)*

## Key Technical Decisions

1. **Heuristic reward, not LLM-as-judge** -- keyword-based reasoning scoring keeps training at 45s/step instead of 5 min/step
2. **Independent rollouts** -- each GRPO candidate resets the environment, preventing shared-state contamination
3. **Solvability gate** -- every generated episode retries up to 10x to ensure the corruption is recoverable with available tools

## Links

- **GitHub**: [github.com/vivekyarra/dataforge-arena](https://github.com/vivekyarra/dataforge-arena)
- **Colab**: *(paste your Colab URL here)*
- **HF Space**: *(paste your Space URL here)*
- **Hackathon**: [Meta PyTorch OpenEnv Hackathon](https://pytorch.org/event/openenv-ai-hackathon/)

## Try It

```bash
git clone https://github.com/vivekyarra/dataforge-arena.git
cd dataforge-arena && pip install -r requirements.txt
python training/generate_data.py
pytest tests/test_all.py -v  # 28/28 pass
python demo/app.py           # Interactive demo
```

---

*Built with PyTorch, TRL, Unsloth, and OpenEnv.*
