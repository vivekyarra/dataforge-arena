# DataForge Arena — 3-Minute Pitch Script
*Meta × PyTorch × HuggingFace × Scaler OpenEnv Hackathon 2026*

---

## [0:00–0:30] THE HOOK

Bad data costs enterprises **$12.9 million per year**. Every data pipeline is one corrupted CSV away from failure.

Data engineers spend **3 days per dataset** hunting nulls, type errors, FK violations, and format inconsistencies — manually. Data quality tools report the problem. None of them fix it automatically.

We asked a different question: **What if an LLM could learn to do this in seconds — and explain every decision?**

Not pattern-match. Not classify. Actually *reason* about why a cell is wrong — whether it violates a schema constraint, breaks a foreign key, or is statistically anomalous — and then fix it, correctly, with a causal justification.

That's DataForge Arena.

---

## [0:30–1:15] THE ENVIRONMENT — ONE EPISODE

Let me show you what the agent sees.

**State:** A 15-row healthcare dataset. In row 7, the `age` field shows `145`. The patient's `birth_year` is `1979`.

**What the agent receives:**
```
age=145
schema_range=[0, 120]  →  VIOLATED
column_mean=42, std=18  →  z-score=5.7  →  OUTLIER
birth_year=1979  →  implied_age=45
violation_type: range + temporal
```

**What an untrained model outputs:**
```json
{"reasoning": "fix", "tool_id": 0, "column": 0, "row_id": 0}
```
Wrong cell. Wrong tool. No justification.

**What the GRPO-trained agent outputs:**
```json
{
  "reasoning": "age 145 exceeds schema max 120; birth_year 1979 implies age ~45 in 2024; z-score 5.7 confirms outlier",
  "tool_id": 3,
  "column": 2,
  "row_id": 7
}
```
Correct cell. Correct tool. Causal justification.

**The reward fires:**
- `+2.0` — constraint alignment: agent correctly identified `range_violation`
- `+0.5` — outlier targeting: cell was a genuine 5.7σ outlier
- `+0.8` — reasoning quality: response references column name and violation type
- `accuracy_delta` fires against ground truth

This is world modeling. The agent maintained an internal model of the schema — type system, FK map, statistical distributions, temporal constraints — and reasoned across all of them simultaneously to earn that reward.

---

## [1:15–2:00] THE TRAINING EVIDENCE

Here's what 75 steps of GRPO on a Tesla T4 looks like.

Parse success rate goes from **25% to 50%** — the model is actively learning to produce structured, parseable JSON. That's the first sign of world model acquisition: the agent cannot express causal reasoning until it can reliably output the format.

Total reward climbs from −1.4 toward baseline. At step 75, the GRPO agent is **11.25× less destructive** than random — a statistically meaningful separation with only 20 evaluation episodes.

The heuristic baseline — our hand-coded oracle — achieves a **50% win rate** (random achieves 0%). That proves the environment is learnable. The GRPO model is closing that gap as training continues.

Full 300-step constraint-aware GRPO training is running right now on onsite HF compute credits. As those results commit, the trained surgeon becomes a **deployable data quality microservice** — drop in a corrupted CSV, get back a repair log with a causal justification for every change.

---

## [2:00–2:30] THE INNOVATION

Three things make DataForge Arena genuinely new.

**First: the reward is purely mathematical.** Accuracy delta against ground truth. No LLM-as-judge. No human annotation in the reward loop. Every signal is independently verifiable. You can inspect the math.

**Second: the constraints require relational reasoning.** FK integrity violations and temporal causal constraints — `birth_year ↔ age` — cannot be resolved by looking at a single cell. The agent must maintain a model of the entire schema. The reward function enforces this: `constraint_alignment` only fires when the identified violation type is correct, which requires understanding the schema structure, not just the cell value.

**Third: the world model is inspectable.** The agent's causal justification — the reasoning field — is machine-checkable. We can verify whether the model is citing the right column, the right violation type, the right statistical evidence. The world model is not a black box. It's legible.

---

## [2:30–3:00] THE ASK

We've built the environment, the reward system, the constraint-aware shaping signals, 130 passing tests, and early training evidence.

With the onsite compute credits, we're running the full 300-step GRPO run now. When it completes, we have a trained autonomous data cleaning agent that can:

- Accept any corrupted CSV conforming to healthcare or financial schemas
- Identify every violation type with causal reasoning
- Apply the correct repair tool to every suspect cell
- Return a machine-readable repair log with justifications for every change

That's not a prototype. That's infrastructure.

**DataForge Arena. The environment that trains agents to fix what humans overlook — by teaching them to understand what data means.**

---

*Evidence artifacts: `eval/results.json` · `eval/heuristic_results.json` · `logs/training_log.csv` · `python -m pytest -q`*
