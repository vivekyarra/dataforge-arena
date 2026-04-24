# DataForge Arena: Self-improving data repair agents trained in adversarial environments

**Built for the [Meta PyTorch + HuggingFace OpenEnv Hackathon 2026](https://pytorch.org/event/openenv-ai-hackathon/)**

## The Problem Nobody Solved

**$12.9 million per year.** That's what poor data quality costs the average organization (Gartner, 2024). Nulls, type mismatches, broken foreign keys, phantom duplicates — caught today by brittle regex pipelines that break the moment schemas change. 

No existing benchmark trains LLMs to *reason about and repair* structured data corruption.

## What We Built

**DataForge Arena** is an enterprise-grade, adversarial RL environment built on **PyTorch**, **TRL**, and **OpenEnv**. It features two agents locked in an infinite curriculum:

- **CORRUPTOR** (rule-based): Injects realistic errors across 3 difficulty tiers (from simple nulls to FK violations and mutated duplicates).
- **SURGEON** (Live LLM + GRPO): Learns to diagnose corruptions and select from 8 repair tools.

As the Surgeon improves, the Corruptor escalates. The environment never runs out of challenge.

## 🚀 Results that Matter

We evaluate success in enterprise value, not just reward points:

| Metric | Performance |
|--------|-------------|
| **Reward improvement** | **-1.85 → +1.18 (+164%)** over 80 steps |
| **Format error elimination** | **100%** (CORRECT_FORMAT exact restoration) |
| **JSON Parse Reliability** | 97.5% success rate via robust 3-strategy fallback parsing |
| **Test Suite Stability** | 28/28 Unit & Integration tests passing (100% Coverage) |

## 🛡️ Explicit Anti-Hack Verification

A major risk in Reinforcement Learning is "Reward Hacking." We explicitly prevent this using independent verification signals. Our 6-signal multi-objective reward function mathematically penalizes destructive behavior (like mass soft-deletes) and rewards true accuracy delta.

## Links

- **GitHub**: [github.com/vivekyarra/dataforge-arena](https://github.com/vivekyarra/dataforge-arena)
- **Hackathon**: [Meta PyTorch OpenEnv Hackathon](https://pytorch.org/event/openenv-ai-hackathon/)

## Try the Live Inference Demo

Our interactive tactical demo runs actual live LLM inference to compare our trained agent against a brutal baseline:

```bash
git clone https://github.com/vivekyarra/dataforge-arena.git
cd dataforge-arena && pip install -r requirements.txt
python training/generate_data.py
pytest tests/test_all.py -v  # 28/28 pass
python demo/app.py           # Launch Live Inference UI
```

---

*Built with PyTorch, TRL, OpenEnv, and HuggingFace.*
