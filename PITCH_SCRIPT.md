[0:00-0:45] Hook + Problem

"Enterprise data is broken in ways that are easy for humans to describe but hard for current LLMs to truly reason about. DataForge Arena is our attempt to turn that gap into a world-modeling benchmark."

[Slide: title slide]

"In real systems, bad rows are not just typos. They are nulls in required columns, values outside schema bounds, enum violations, foreign-key mismatches, and cross-column inconsistencies like a birth year that implies a very different age. A model can read those cells. The hard part is understanding why the value is wrong. That is a world-model problem, not a knowledge problem."

[Switch to architecture slide]

[0:45-1:30] The Environment

"DataForge Arena is an OpenEnv-compatible RL environment for tabular data repair. Each episode starts with a clean healthcare or financial table, then an adversarial corruptor injects one of seven learnable corruption types across three tiers of difficulty."

"The agent sees a structured observation, including corrupted rows, schema metadata, violation summaries, and action history. It then chooses one of eight repair tools: median imputation, mode imputation, forward fill, format correction, delete row, merge duplicate, flag uncertain, or no-op. The key is that reward is grounded in cell-level accuracy delta against clean ground truth, so there is no LLM judge in the loop."

[Switch to reward slide]

"On top of accuracy delta, we shape the policy with six additional signals: constraint alignment, schema alignment, outlier targeting, reasoning quality, parse bonus, and anti-hack penalties. Constraint alignment is worth plus three, which means the agent is rewarded most when it understands the causal violation type correctly."

[1:30-2:15] The Results

[Switch to training curve / evaluation slide]

"The committed GRPO checkpoint is an early run: 300 steps on a T4. At that stage, the model is not solved, and we do not claim that it is. What we can show is measurable movement in the right direction."

"In the committed evaluation artifact, the GRPO agent is 9.8 times less destructive than a random baseline, with a destruction ratio of 0.102. It achieves a 5 percent win rate on tier-1 evaluation episodes, and the training log shows 100 percent parse success sustained across the run. The total reward increases from 1.925 to 4.475, which is a 132 percent gain, although we explicitly note that this is driven largely by parse shaping in the current run."

[Switch to live demo or notebook screenshot]

"The honest read is: the environment is working, the reward is grounded, the agent is learning structured behavior, and the next full rerun should be much more informative now that the reward-path bugs are fixed."

[2:15-3:00] Why it matters + close

[Switch to closing slide]

"This matters because a lot of professional AI work will depend on models acting inside structured systems, not just generating text. If a model cannot maintain a causal understanding of schemas, constraints, and distributions, it cannot be trusted to repair or operate on enterprise data."

"DataForge Arena gives us a benchmark where that capability is measurable, reproducible, and hard to fake. It is an RL environment where the world model is not decorative. It is the only way to earn reward. That is why we think it is a strong fit for Theme 3.1."

Most likely judge questions:

Q: "Why is your win rate only 5%?"
A: "Because this is a short 300-step T4 run near an already high accuracy ceiling; the more informative current metric is that the policy is 9.8x less destructive than random while maintaining valid structured actions."

Q: "How is this different from a rule-based data cleaner?"
A: "Rules can encode repairs, but the benchmark is about learning a reusable policy that chooses tools from observation and reward, including cross-column and distribution-aware reasoning rather than hard-coded case logic."

Q: "What would you do with more compute?"
A: "I would rerun GRPO after the v1.1 reward fixes, expand the tier-2 and financial mix during training, and use the same reward instrumentation to measure whether constraint alignment and win rate rise together rather than only parse quality."
