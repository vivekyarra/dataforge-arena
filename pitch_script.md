# DataForge Arena - 3 Minute Judge Pitch

## 0:00-0:25 - Hook

Enterprise agents do not fail only because they cannot write polished text. They fail because the world underneath them is messy: a missing customer field, a broken date, a duplicated healthcare row, or a foreign-key mismatch that looks almost right.

DataForge Arena turns that mess into an OpenEnv benchmark. The agent is not graded on sounding confident. It is graded on whether the table actually gets better after it acts.

## 0:25-1:05 - World

The world is a tabular enterprise repair environment. On every reset, an adversarial corruptor injects solvable data-quality failures across a curriculum: simple nulls and type errors first, then format issues and out-of-range values, then relational mistakes and duplicate-row mutations.

The observation includes the schema, sampled rows, corruption summary, and recent action history. The action is constrained JSON: reasoning, tool id, column, and row. The repair tools include imputation, format correction, merge duplicate, delete row, flag uncertain, and no-op.

## 1:05-1:45 - Demo Moment

In the demo, I start with a corrupted table and run the naive baseline first. It picks tools without understanding the cell, so the accuracy delta usually stays flat or gets worse.

Then I run the surgeon path. The UI shows the exact action trace: which row, which column, which tool, the reasoning string, and the reward from the environment. The important number is the before-and-after dataset health. If the repair did not improve the table, the reward shows that.

If a trained checkpoint is present locally, the demo exposes Live GRPO Model. If not, that option stays hidden. The interface is deliberately honest about model provenance.

## 1:45-2:30 - Learning Signal

The reward loop is grounded in state change. The primary signal is `accuracy_delta`, and the shaping terms are there to keep the agent pointed at real repair behavior: valid tools, appropriate targets, efficiency, and anti-shortcut checks.

The key fix before final training is that efficiency now gives a positive signal when a repair tool targets an actually incorrect cell. That prevents the model from learning verbose explanations that are disconnected from repairs.

In the final run, the evidence to show is the training curve: accuracy delta should trend upward after the efficiency fix, and the trained checkpoint should be evaluated against the random baseline with the same harness.

## 2:30-3:00 - Close

DataForge Arena is small enough to inspect, fast enough to train for a hackathon, and realistic enough to capture the enterprise problem: agents act inside stateful systems, tools have side effects, and success means the world improved.

That is the benchmark: messy enterprise data in, constrained repair action out, grounded reward from what actually changed.
