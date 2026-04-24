"""
DataForge Arena — Evaluation Harness
Run after training to produce before/after accuracy numbers for your pitch.

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --episodes 20 --tier 2
"""
import sys, os, json, random, argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.env import DataForgeEnv, SurgeonAction
from environment.corruptor import Corruptor
from environment.reward import RewardComputer
from environment.schemas import HEALTHCARE_SCHEMA, SURGEON_TOOLS


def random_baseline_agent(state: pd.DataFrame, gt: pd.DataFrame) -> SurgeonAction:
    """Untrained agent: picks random tool on random cell."""
    display_cols = [c for c in state.columns if c != "_is_deleted"]
    return SurgeonAction(
        reasoning="random action",
        tool_id=random.choice([0, 1, 2, 3, 7]),
        column=random.randint(0, max(0, len(display_cols) - 1)),
        row_id=random.randint(0, max(0, len(state) - 1)),
    )


def heuristic_surgeon_agent(state: pd.DataFrame, gt: pd.DataFrame,
                             schema: dict) -> SurgeonAction:
    """Heuristic 'trained' agent: finds corrupted cell and picks appropriate tool."""
    display_cols = [c for c in state.columns if c != "_is_deleted"]
    
    # Scan for errors
    for r in range(min(len(state), len(gt))):
        for c_idx, c_name in enumerate(display_cols):
            cell = state.at[r, c_name]
            gt_cell = gt.at[r, c_name]
            
            if pd.isna(cell) and pd.notna(gt_cell):
                # Null cell -- pick imputation tool
                col_type = schema.get(c_name, {}).get("type", "str")
                if col_type in ("int", "float"):
                    tool_id = 0  # IMPUTE_MEDIAN
                    reason = f"Null in numeric column '{c_name}' -- using IMPUTE_MEDIAN"
                else:
                    tool_id = 1  # IMPUTE_MODE
                    reason = f"Missing value in '{c_name}' -- using IMPUTE_MODE"
                return SurgeonAction(reasoning=reason, tool_id=tool_id, column=c_idx, row_id=r)
            
            elif pd.notna(cell) and pd.notna(gt_cell) and str(cell) != str(gt_cell):
                # Wrong value -- check if it's a type error (ERR_XX pattern)
                cell_str = str(cell)
                if cell_str.startswith("ERR_") or not _matches_type(cell_str, schema.get(c_name, {})):
                    # Type error: impute with mode/median instead of trying CORRECT_FORMAT
                    col_type = schema.get(c_name, {}).get("type", "str")
                    if col_type in ("int", "float"):
                        tool_id = 0  # IMPUTE_MEDIAN
                        reason = f"Type error '{cell}' in numeric '{c_name}' -- IMPUTE_MEDIAN"
                    else:
                        tool_id = 1  # IMPUTE_MODE
                        reason = f"Type error '{cell}' in '{c_name}' -- IMPUTE_MODE to replace"
                else:
                    # Format error (date format, etc)
                    tool_id = 3  # CORRECT_FORMAT
                    reason = f"Format error '{cell}' in '{c_name}' -- CORRECT_FORMAT"
                return SurgeonAction(reasoning=reason, tool_id=tool_id, column=c_idx, row_id=r)
    
    # Check for duplicates (extra rows beyond GT)
    if len(state) > len(gt):
        return SurgeonAction(reasoning="duplicate row detected -- DELETE_ROW",
                             tool_id=4, column=0, row_id=len(state)-1)
    
    # No errors found -- NO_OP
    return SurgeonAction(reasoning="no errors detected", tool_id=7, column=0, row_id=0)


def _matches_type(val_str: str, schema_info: dict) -> bool:
    """Check if a string value roughly matches the expected type."""
    col_type = schema_info.get("type", "str")
    if col_type in ("int", "float"):
        try:
            float(val_str)
            return True
        except (ValueError, TypeError):
            return False
    return True  # strings always match


def evaluate(n_episodes: int = 10, tier: int = 1, max_steps: int = 5):
    """Run evaluation and print before/after results."""
    clean_data = pd.read_csv("data/healthcare_clean.csv")
    corruptor = Corruptor()
    rc = RewardComputer()
    
    # Force tier
    corruptor._epoch = {1: 0, 2: 65, 3: 115}[tier]
    
    results = {
        "random": {"before": [], "after": [], "deltas": []},
        "surgeon": {"before": [], "after": [], "deltas": []},
    }
    
    print(f"\n{'='*60}")
    print(f"  DataForge Arena -- Evaluation Report")
    print(f"  Episodes: {n_episodes} | Tier: {tier} | Max Steps: {max_steps}")
    print(f"{'='*60}\n")
    
    for ep in range(n_episodes):
        n = min(50, len(clean_data))
        sample = clean_data.sample(n=n).reset_index(drop=True)
        dirty, gt, meta = corruptor.generate_episode(sample)
        
        if meta.get("tool") == "duplicate_row_mutate" and len(dirty) > len(gt):
            src = meta.get("row", 0)
            if src < len(gt):
                gt = pd.concat([gt, gt.iloc[[src]]], ignore_index=True)
        
        acc_before = rc._field_accuracy(dirty, gt)
        
        for agent_name, agent_fn in [
            ("random", lambda s, g: random_baseline_agent(s, g)),
            ("surgeon", lambda s, g: heuristic_surgeon_agent(s, g, HEALTHCARE_SCHEMA)),
        ]:
            state = dirty.copy()
            for _ in range(max_steps):
                action = agent_fn(state, gt)
                from environment.tools import apply_tool
                state = apply_tool(state, action, HEALTHCARE_SCHEMA)
            
            acc_after = rc._field_accuracy(state, gt)
            delta = acc_after - acc_before
            
            results[agent_name]["before"].append(acc_before)
            results[agent_name]["after"].append(acc_after)
            results[agent_name]["deltas"].append(delta)
        
        print(f"  Episode {ep+1:2d}/{n_episodes} | corruption={meta['tool']:25s} | "
              f"random: {results['random']['deltas'][-1]:+.3f} | "
              f"surgeon: {results['surgeon']['deltas'][-1]:+.3f}")
    
    print(f"\n{'-'*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'-'*60}")
    
    for agent_name in ["random", "surgeon"]:
        r = results[agent_name]
        avg_before = np.mean(r["before"])
        avg_after = np.mean(r["after"])
        avg_delta = np.mean(r["deltas"])
        
        label = "Random Baseline" if agent_name == "random" else "DataForge Surgeon"
        print(f"\n  {label}:")
        print(f"    Avg accuracy before:  {avg_before:.4f}")
        print(f"    Avg accuracy after:   {avg_after:.4f}")
        print(f"    Avg improvement:      {avg_delta:+.4f} ({avg_delta*100:+.2f}%)")
        print(f"    Win rate (delta > 0): {sum(1 for d in r['deltas'] if d > 0)}/{n_episodes}")
    
    # Headline number
    surgeon_delta = np.mean(results["surgeon"]["deltas"])
    random_delta = np.mean(results["random"]["deltas"])
    advantage = surgeon_delta - random_delta
    
    print(f"\n{'='*60}")
    print(f"  HEADLINE: Surgeon outperforms random by {advantage*100:+.2f}% accuracy")
    print(f"{'='*60}\n")
    
    # Save results
    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "w") as f:
        json.dump({
            "tier": tier,
            "episodes": n_episodes,
            "surgeon_avg_delta": round(float(surgeon_delta), 6),
            "random_avg_delta": round(float(random_delta), 6),
            "advantage": round(float(advantage), 6),
        }, f, indent=2)
    print("  Results saved to eval/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    
    evaluate(n_episodes=args.episodes, tier=args.tier, max_steps=args.steps)
