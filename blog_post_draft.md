---
title: "DataForge Arena: Teaching LLMs to Fix Broken Enterprise Data with Adversarial RL"
authors:
  - user: Vivek567
---

**Built for the [Meta PyTorch + HuggingFace OpenEnv Hackathon 2026](https://pytorch.org/event/openenv-ai-hackathon/)**  
**Theme: 3.1 World Modeling — Multi-App RL Environment for Enterprise Workflows (Scaler AI Labs)**

## The $12.9 Million Problem

**$12.9 million per year.** That's what poor data quality costs the average organization, according to Gartner. Every enterprise on earth has the exact same story: corrupted fields, broken foreign keys, phantom duplicates, and inconsistent date formats. Today, these are caught by brittle regex pipelines that shatter the moment the upstream schema changes. 

LLMs can write Python code, pass the bar exam, and generate beautiful artwork. But ask a state-of-the-art model to look at a corrupted patient record with a null `age` field, a swapped `department_id`, and a duplicated row with a mutated email address — and it hallucinates. It picks the wrong tool. It doesn't even notice the duplicate.

**No open-source benchmark exists to train this specific skill.** Until now.

## Introducing DataForge Arena

DataForge Arena is an enterprise-grade, adversarial Reinforcement Learning environment built on **PyTorch**, **TRL**, and **OpenEnv**. It features two agents locked in an infinite, self-improving curriculum.

### The World Modeling Framing

To solve data corruption, an agent must build an internal model of what clean data looks like versus what it observes. This is the essence of **World Modeling**.
* It must model tool-effect relationships (`IMPUTE_MEDIAN` on a null numeric column restores the statistical distribution).
* It must model adversarial dynamics (our Corruptor escalates its tactics as the Surgeon improves).
* It must model multi-schema environments (reasoning across healthcare and financial datasets).

This is a highly structured world with strict rules, complex states, distinct tools, and mathematical consequences — exactly the domain World Modeling RL targets.

## System Architecture

Our environment consists of two primary actors:

1. **The CORRUPTOR (Rule-based, 3 Tiers):** A dynamic difficulty engine that injects realistic errors into pristine data. It monitors the agent's rolling average reward and automatically unlocks harder difficulty tiers when the agent proves competent.
2. **The SURGEON (Qwen 2.5 1.5B + LoRA):** The agent trained via GRPO to diagnose the corruption and select the mathematically optimal repair tool from its arsenal.

### The Adversarial Curriculum

| Tier | Epochs | What the Corruptor Does | What the Surgeon Must Learn |
|------|--------|------------------------|---------------------------|
| **1** | 0–29 | Single null injection, type errors (`ERR_42`) | Basic imputation, type detection |
| **2** | 30–69 | Null clusters, date format swaps, out-of-range bounds | Pattern recognition, multi-cell correlation |
| **3** | 70+ | Foreign key violations, duplicate rows with mutation | Relational reasoning, merge/delete decisions |

### 6-Signal Reward Computer

Instead of using a slow, expensive "LLM-as-a-judge" to evaluate repairs, DataForge uses a deterministic 6-signal reward computer. This allows us to train at 45 seconds per step on a standard T4 GPU instead of 5 minutes.
* **Accuracy Delta:** Did the repair actually move the dataset closer to the ground truth?
* **Tool Logic:** Was the mathematically correct tool chosen for the detected error?
* **Anti-Hack Penalty:** Massive negative rewards for gaming the system (e.g., trying to soft-delete every row to bypass errors).

## 🚀 Results that Matter

We evaluate success in enterprise value, not just abstract reward points. Over an 80-step training run using TRL's `GRPOTrainer`:

| Metric | Performance |
|--------|-------------|
| **Difficulty progression** | **Tier 1 → 2 → 3** (DDA unlocked all tiers over 75 steps) |
| **Format error elimination** | **100%** (CORRECT_FORMAT exact restoration) |
| **JSON Parse Reliability** | 93% success rate via robust 3-strategy fallback parsing |
| **Test Suite Stability** | 28/28 Unit & Integration tests passing (100% Coverage) |

The 93% JSON parse success rate is our most significant signal. Under RL pressure, the model is simultaneously learning *what to do* AND *how to perfectly format its output*.

## Try the Live Inference Demo

We built a "Billion-Dollar" Gradio frontend that runs actual live LLM inference, allowing you to pit our trained Surgeon against a brutal Naive Baseline on Tier 3 adversarial data.

```bash
git clone https://github.com/vivekyarra/dataforge-arena.git
cd dataforge-arena && pip install -r requirements.txt
python training/generate_data.py
python demo/app.py           # Launch Live Inference UI
```

## Links

| Resource | URL |
|----------|-----|
| 🤗 **Live HF Space** | https://huggingface.co/spaces/Vivek567/enterprise-data-cleaning-env |
| 📓 **Colab Notebook** | DataForge_Arena_Colab.ipynb |
| 💻 **GitHub** | https://github.com/vivekyarra/dataforge-arena |

---

*Built with PyTorch, TRL, OpenEnv, and HuggingFace.*
