"""
Plots per-step accuracy improvement for heuristic surgeon over a single episode.

Run:
    python eval/multi_step_curve.py

Saves:
    logs/multi_step_accuracy.png
"""
from __future__ import annotations

import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.corruptor import Corruptor
from environment.env import DataForgeEnv, SurgeonAction
from environment.reward import RewardComputer
from environment.schemas import HEALTHCARE_SCHEMA


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT_DIR, "logs", "multi_step_accuracy.png")
DATA_PATH = os.path.join(ROOT_DIR, "data", "healthcare_clean.csv")


def heuristic_surgeon(state: pd.DataFrame, gt: pd.DataFrame) -> SurgeonAction:
    cols = [c for c in state.columns if c != "_is_deleted"]
    for row_idx in range(min(len(state), len(gt))):
        for col_idx, col_name in enumerate(cols):
            cell = state.at[row_idx, col_name]
            gt_cell = gt.at[row_idx, col_name]
            if pd.isna(cell) and pd.notna(gt_cell):
                col_type = HEALTHCARE_SCHEMA.get(col_name, {}).get("type", "str")
                tool_id = 0 if col_type in ("int", "float") else 1
                reason = f"Null in '{col_name}'"
                return SurgeonAction(reasoning=reason, tool_id=tool_id, column=col_idx, row_id=row_idx)
            if pd.notna(cell) and pd.notna(gt_cell) and str(cell) != str(gt_cell):
                if str(cell).startswith("ERR_"):
                    col_type = HEALTHCARE_SCHEMA.get(col_name, {}).get("type", "str")
                    tool_id = 0 if col_type in ("int", "float") else 1
                    return SurgeonAction(
                        reasoning=f"Type error in '{col_name}'",
                        tool_id=tool_id,
                        column=col_idx,
                        row_id=row_idx,
                    )
                return SurgeonAction(
                    reasoning=f"Format error in '{col_name}'",
                    tool_id=3,
                    column=col_idx,
                    row_id=row_idx,
                )
    if len(state) > len(gt):
        return SurgeonAction(reasoning="Duplicate row", tool_id=4, column=0, row_id=len(state) - 1)
    return SurgeonAction(reasoning="No errors detected", tool_id=7, column=0, row_id=0)


def main():
    os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)

    random.seed(7)
    np.random.seed(7)

    clean_data = pd.read_csv(DATA_PATH)
    corruptor = Corruptor()
    corruptor.force_tier(1)
    env = DataForgeEnv(corruptor=corruptor, schema=HEALTHCARE_SCHEMA, clean_data=clean_data)
    reward_computer = RewardComputer()

    env.reset()
    accuracies = [reward_computer._field_accuracy(env._state, env._ground_truth)]
    annotations = ["start"]

    cols = [c for c in env._state.columns if c != "_is_deleted"]
    done = False
    for _ in range(10):
        if not done:
            action = heuristic_surgeon(env._state.copy(), env._ground_truth.copy())
            _, _, done, _ = env.step(action)
            col_name = cols[action.column] if action.column < len(cols) else "?"
            action_label = f"{col_name} | tool {action.tool_id} | r{action.row_id}"
            annotations.append(action_label)
            accuracies.append(reward_computer._field_accuracy(env._state, env._ground_truth))
        else:
            annotations.append("done")
            accuracies.append(accuracies[-1])

    steps = list(range(len(accuracies)))

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#111118")
    ax.plot(steps, accuracies, color="#10b981", linewidth=2.8, marker="o", markersize=7)
    ax.fill_between(steps, accuracies, color="#10b981", alpha=0.12)
    ax.set_title(
        "DataForge Arena - Per-Step Accuracy Recovery (Heuristic Surgeon, Tier 1)",
        color="#e2e8f0",
        fontsize=15,
    )
    ax.set_xlabel("Step", color="#94a3b8")
    ax.set_ylabel("Accuracy", color="#94a3b8")
    ax.set_ylim(0.994, 1.001)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.4f}'))
    ax.set_xlim(0, 10)
    ax.grid(color="#1e293b", alpha=0.35, linestyle="--", linewidth=0.8)
    ax.tick_params(colors="#94a3b8")
    ax.spines["bottom"].set_color("#1e293b")
    ax.spines["left"].set_color("#1e293b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for step, accuracy, label in zip(steps[1:], accuracies[1:], annotations[1:]):
        short_label = label if len(label) <= 22 else f"{label[:19]}..."
        ax.annotate(
            short_label,
            xy=(step, accuracy),
            xytext=(0, 10 if step % 2 else -18),
            textcoords="offset points",
            ha="center",
            color="#e2e8f0",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "#0f172a",
                "edgecolor": "#1f2937",
                "alpha": 0.9,
            },
        )

    ax.annotate(
        f"Initial {accuracies[0]:.3f}",
        xy=(0, accuracies[0]),
        xytext=(8, 12),
        textcoords="offset points",
        color="#cbd5e1",
        fontsize=9,
    )
    ax.annotate(
        f"Final {accuracies[-1]:.3f}",
        xy=(steps[-1], accuracies[-1]),
        xytext=(-50, -24),
        textcoords="offset points",
        color="#cbd5e1",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(LOG_PATH, dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    plt.close(fig)

    print(f"Saved: {LOG_PATH}")
    print("Per-step accuracy:", [round(value, 4) for value in accuracies])
    print("Actions:", annotations[1:])


if __name__ == "__main__":
    main()
