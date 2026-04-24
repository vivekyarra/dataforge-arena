"""
Pre-compute 3 episodes for the Gradio demo.
Run in Colab AFTER training, or locally with heuristic fallback.

Usage (Colab, after training):
    python demo/precomputed/generate_episodes.py

Usage (local, heuristic fallback):
    python demo/precomputed/generate_episodes.py --heuristic
"""
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from environment.env import DataForgeEnv, SurgeonAction
from environment.corruptor import Corruptor
from environment.reward import RewardComputer
from environment.schemas import HEALTHCARE_SCHEMA, SURGEON_TOOLS
from environment.tools import apply_tool


def heuristic_action(state, gt, schema):
    """Pick the best heuristic repair action."""
    display_cols = [c for c in state.columns if c != "_is_deleted"]
    for r in range(min(len(state), len(gt))):
        for c_idx, c_name in enumerate(display_cols):
            cell = state.at[r, c_name]
            gt_cell = gt.at[r, c_name]
            if pd.isna(cell) and pd.notna(gt_cell):
                col_type = schema.get(c_name, {}).get("type", "str")
                tool_id = 0 if col_type in ("int", "float") else 1
                return SurgeonAction(
                    reasoning=f"Null in '{c_name}' -> {'IMPUTE_MEDIAN' if tool_id==0 else 'IMPUTE_MODE'}",
                    tool_id=tool_id, column=c_idx, row_id=r)
            elif pd.notna(cell) and pd.notna(gt_cell) and str(cell) != str(gt_cell):
                col_type = schema.get(c_name, {}).get("type", "str")
                tool_id = 0 if col_type in ("int", "float") else 1
                return SurgeonAction(
                    reasoning=f"Type error '{cell}' in '{c_name}' -> impute",
                    tool_id=tool_id, column=c_idx, row_id=r)
    return SurgeonAction(reasoning="no errors detected", tool_id=7, column=0, row_id=0)


def generate(use_heuristic=True):
    clean_data = pd.read_csv("data/healthcare_clean.csv")
    corruptor = Corruptor()
    rc = RewardComputer()
    out_dir = "demo/precomputed"
    os.makedirs(out_dir, exist_ok=True)

    tiers = [
        {"tier": 1, "epoch": 0, "label": "Tier 1 -- single null/type error"},
        {"tier": 2, "epoch": 65, "label": "Tier 2 -- null cluster/date swap"},
        {"tier": 3, "epoch": 115, "label": "Tier 3 -- FK violation/duplicate row"},
    ]

    for ep_num, tier_info in enumerate(tiers, 1):
        corruptor._epoch = tier_info["epoch"]
        n = min(50, len(clean_data))
        sample = clean_data.sample(n=n).reset_index(drop=True)
        dirty, gt, meta = corruptor.generate_episode(sample)

        if meta.get("tool") == "duplicate_row_mutate" and len(dirty) > len(gt):
            src = meta.get("row", 0)
            if src < len(gt):
                gt = pd.concat([gt, gt.iloc[[src]]], ignore_index=True)

        state = dirty.copy()
        acc_before = rc._field_accuracy(state, gt)
        steps = []

        for s in range(5):
            action = heuristic_action(state, gt, HEALTHCARE_SCHEMA)
            prev_acc = rc._field_accuracy(state, gt)
            state = apply_tool(state, action, HEALTHCARE_SCHEMA)
            new_acc = rc._field_accuracy(state, gt)

            steps.append({
                "step": s,
                "action": action.model_dump(),
                "tool_name": SURGEON_TOOLS[action.tool_id]["name"],
                "accuracy_before": round(prev_acc, 4),
                "accuracy_after": round(new_acc, 4),
                "delta": round(new_acc - prev_acc, 4),
            })

        acc_after = rc._field_accuracy(state, gt)
        episode = {
            "episode": ep_num,
            "tier": tier_info["tier"],
            "label": tier_info["label"],
            "corruption": meta["tool"],
            "accuracy_before": round(acc_before, 4),
            "accuracy_after": round(acc_after, 4),
            "steps": steps,
        }

        path = os.path.join(out_dir, f"episode_{ep_num}.json")
        with open(path, "w") as f:
            json.dump(episode, f, indent=2)
        print(f"Episode {ep_num}: {meta['tool']} | {acc_before:.3f} -> {acc_after:.3f} | saved to {path}")

    print("\nDone. 3 episodes saved to demo/precomputed/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic", action="store_true", default=True)
    args = parser.parse_args()
    generate(use_heuristic=args.heuristic)
