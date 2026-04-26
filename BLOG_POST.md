# DataForge Arena: Training LLMs to Reason About Schema Constraints via GRPO

Enterprise data is broken far more often than most ML benchmarks admit. Industry estimates regularly put serious data-quality defects in the double digits, and one commonly cited figure is that roughly 34% of enterprise data contains material errors. Those errors are not just typos. They are nulls in required fields, values outside schema ranges, enum violations, temporal inconsistencies, and foreign-key mismatches that quietly poison downstream analytics and automation.

Current LLMs can read a table, describe a row, and often guess that something "looks wrong." What they struggle to do reliably is explain why a specific cell is wrong in the context of a schema. That is a world-model problem, not a knowledge problem. To repair `age=145`, the model must understand range constraints. To repair a mismatched `department_id`, it must reason over a relational mapping. To fix a suspicious `amount`, it must place the value inside a statistical distribution rather than treat it as an isolated token.

DataForge Arena is an OpenEnv-compatible reinforcement learning environment built around that gap. The agent receives a corrupted tabular state and must choose a structured repair action: which tool to use, which column to target, which row to target, and a natural-language justification for the decision. The environment currently supports healthcare and financial schemas, and the evaluation harness can run either schema independently or both together.

The corruptor injects seven learnable corruption types across three difficulty tiers. Tier 1 includes single-cell null injection, type-error injection, and enum substitution. Tier 2 adds clustered nulls, date-format swaps, out-of-range ages, temporal drift, and currency-unit mismatch. Tier 3 introduces foreign-key failures and duplicate-row mutation. The agent acts in an eight-tool space: median imputation, mode imputation, forward fill, format correction, delete row, merge duplicate, flag uncertain, and no-op. Every episode is checked against clean ground truth, so reward is fully verifiable and does not rely on an LLM judge.

The reward function combines seven signals. The primary term is ground-truth accuracy delta scaled by 50. On top of that are shaped rewards for constraint alignment, schema alignment, outlier targeting, reasoning quality, parse quality, and anti-hack penalties. The most important shaped term is `constraint_alignment` at +3.0. That weight is deliberate: it forces the policy to learn why the corruption occurred, not just what edit might superficially reduce loss. An agent that picks the wrong tool for a null cell, or applies formatting to a missing numeric field, is penalized directly.

The current committed GRPO checkpoint is still an early training run, and the numbers should be read that way. In the committed evaluation artifact, the GRPO agent is 9.8x less destructive than a random baseline, achieves a 5% win rate on tier-1 episodes, and holds 100% parse success across the training log. Those numbers do not mean the environment is solved. They do show that even a short 300-step run on a T4 can move the policy away from random corruption and toward structured, valid actions. The committed heuristic baseline also shows the task is learnable: it outperforms random by a larger accuracy margin, but without representing learned behavior.

That mix of honesty and verifiability is the point. DataForge Arena is not claiming a finished production model. It is claiming a benchmark where the only path to higher reward is better schema-grounded reasoning, and where every result can be inspected line by line.

Links:

- GitHub: https://github.com/vivekyarra/dataforge-arena
- HF Space: https://huggingface.co/spaces/Vivek567/dataforge-arena
- Demo Video: TBA
- Colab Notebook: https://colab.research.google.com/github/vivekyarra/dataforge-arena/blob/master/DataForge_Arena_Colab.ipynb
