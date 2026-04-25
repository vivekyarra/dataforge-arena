# Final Submission Summary

## Evidence Position

Use the heuristic surgeon as the live demo lead. It is the strongest policy in the current artifacts and beats random by `+0.53 pp` accuracy delta on the committed heuristic baseline.

Use the GRPO checkpoint as training evidence. The short Tesla T4 run produced a real checkpoint and a real evaluation, but it does not beat the heuristic. It is still less destructive than random:

| Metric | Value |
|--------|-------|
| GPU | Tesla T4 |
| GRPO eval mode | `grpo` |
| Episodes / tier / steps / seed | `20 / 1 / 5 / 7` |
| GRPO avg accuracy delta | `-0.0004` |
| Random avg accuracy delta | `-0.0045` |
| GRPO advantage over random | `+0.0041` (`+0.41 pp`) |
| GRPO win rate | `0.00%` |
| Random win rate | `0.00%` |

## Training Signal

| Training artifact | Value |
|-------------------|-------|
| Target steps | `80` |
| Last logged step | `75` |
| First -> final reward | `-1.4000 -> -1.4000` |
| Best logged reward | `-0.2000` |
| Smoothed reward, first 3 rows -> last 3 rows | `-1.2000 -> -1.0000` |
| Parse success, first -> final | `25% -> 50%` |
| Mean parse success | `40.00%` |
| Difficulty tiers observed | `1, 2, 3` |

## Judge Narrative

Enterprise data agents fail when they sound confident but make messy state worse. DataForge Arena turns that into an OpenEnv world: observe a corrupted table, choose a constrained repair tool, and get rewarded only by measurable state change.

The strongest current demo is the heuristic surgeon, which beats random and shows inspectable before/after table health. The GRPO checkpoint is the proof that the training path is live: on a small T4 run, the model learned enough output structure to become less destructive than random, with parse success moving from `25%` to `50%`.

The honest conclusion: this is a working benchmark and training loop, not a fully optimized model. Scaling the run and tightening format learning are the obvious next steps.
