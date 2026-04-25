"""
DataForge Arena - Evaluation Harness

Usage:
    python eval/evaluate.py --agent-mode heuristic
    python eval/evaluate.py --agent-mode grpo --model-path outputs/dataforge-surgeon
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.corruptor import Corruptor
from environment.env import DataForgeEnv, SurgeonAction
from environment.reward import RewardComputer
from environment.schemas import HEALTHCARE_SCHEMA
from training.parser import robust_parse_action
from training.prompt import build_prompt


DEFAULT_LOCAL_MODEL_PATH = "outputs/dataforge-surgeon"
llm_pipeline = None


def _preferred_inference_dtype(torch_module, device: int):
    if device < 0:
        return torch_module.float32
    major, _ = torch_module.cuda.get_device_capability(device)
    return torch_module.bfloat16 if major >= 8 else torch_module.float16


def resolve_eval_agent(agent_mode: str, model_path: str | None = None) -> dict:
    if agent_mode == "heuristic":
        return {
            "agent_mode": "heuristic",
            "model_source": "heuristic-rule-based",
            "model_path": None,
            "fallback_used": False,
        }

    resolved_path = model_path or DEFAULT_LOCAL_MODEL_PATH
    if not Path(resolved_path).exists():
        raise FileNotFoundError(
            f"GRPO evaluation requested, but no local checkpoint was found at '{resolved_path}'. "
            "Train the model first or pass --agent-mode heuristic."
        )

    return {
        "agent_mode": "grpo",
        "model_source": resolved_path,
        "model_path": resolved_path,
        "fallback_used": False,
    }


def load_eval_pipeline(model_path: str):
    global llm_pipeline
    if llm_pipeline is not None:
        return llm_pipeline

    from transformers import pipeline
    import torch

    print(f"Loading GRPO checkpoint from {model_path} ...")
    device = 0 if torch.cuda.is_available() else -1
    llm_pipeline = pipeline(
        "text-generation",
        model=model_path,
        device=device,
        torch_dtype=_preferred_inference_dtype(torch, device),
    )
    return llm_pipeline


def random_baseline_agent(state: pd.DataFrame, gt: pd.DataFrame) -> SurgeonAction:
    display_cols = [c for c in state.columns if c != "_is_deleted"]
    return SurgeonAction(
        reasoning="random action",
        tool_id=random.choice([0, 1, 2, 3, 7]),
        column=random.randint(0, max(0, len(display_cols) - 1)),
        row_id=random.randint(0, max(0, len(state) - 1)),
    )


def heuristic_surgeon_agent(state: pd.DataFrame, gt: pd.DataFrame, schema: dict) -> SurgeonAction:
    display_cols = [c for c in state.columns if c != "_is_deleted"]

    for row_idx in range(min(len(state), len(gt))):
        for col_idx, col_name in enumerate(display_cols):
            cell = state.at[row_idx, col_name]
            gt_cell = gt.at[row_idx, col_name]

            if pd.isna(cell) and pd.notna(gt_cell):
                col_type = schema.get(col_name, {}).get("type", "str")
                tool_id = 0 if col_type in ("int", "float") else 1
                reason = (
                    f"Null in numeric column '{col_name}' - IMPUTE_MEDIAN"
                    if tool_id == 0
                    else f"Missing value in '{col_name}' - IMPUTE_MODE"
                )
                return SurgeonAction(reasoning=reason, tool_id=tool_id, column=col_idx, row_id=row_idx)

            if pd.notna(cell) and pd.notna(gt_cell) and str(cell) != str(gt_cell):
                cell_str = str(cell)
                if cell_str.startswith("ERR_") or not _matches_type(cell_str, schema.get(col_name, {})):
                    col_type = schema.get(col_name, {}).get("type", "str")
                    tool_id = 0 if col_type in ("int", "float") else 1
                    return SurgeonAction(
                        reasoning=f"Type error '{cell}' in '{col_name}'",
                        tool_id=tool_id,
                        column=col_idx,
                        row_id=row_idx,
                    )
                return SurgeonAction(
                    reasoning=f"Format or consistency error in '{col_name}'",
                    tool_id=3,
                    column=col_idx,
                    row_id=row_idx,
                )

    if len(state) > len(gt):
        return SurgeonAction(
            reasoning="duplicate row detected - DELETE_ROW",
            tool_id=4,
            column=0,
            row_id=len(state) - 1,
        )

    return SurgeonAction(reasoning="no errors detected", tool_id=7, column=0, row_id=0)


def grpo_surgeon_agent(state: pd.DataFrame, gt: pd.DataFrame, env: DataForgeEnv) -> SurgeonAction:
    obs = env._make_observation()
    prompt = build_prompt(obs)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Observation: {obs.model_dump_json()}\nOutput valid JSON only."},
    ]

    try:
        outputs = llm_pipeline(
            messages,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            num_return_sequences=1,
        )
        generated_text = outputs[0]["generated_text"][-1]["content"]
        return robust_parse_action(generated_text)
    except Exception as exc:
        return SurgeonAction(
            reasoning=f"LLM inference failure: {str(exc)[:48]}",
            tool_id=7,
            column=0,
            row_id=0,
        )


def _matches_type(val_str: str, schema_info: dict) -> bool:
    col_type = schema_info.get("type", "str")
    if col_type in ("int", "float"):
        try:
            float(val_str)
            return True
        except (ValueError, TypeError):
            return False
    return True


def _align_duplicate_ground_truth(dirty: pd.DataFrame, gt: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if meta.get("tool") == "duplicate_row_mutate" and len(dirty) > len(gt):
        src_row = meta.get("row", 0)
        if src_row < len(gt):
            return pd.concat([gt, gt.iloc[[src_row]]], ignore_index=True)
    return gt


def _bootstrap_eval_env(
    clean_data: pd.DataFrame,
    dirty: pd.DataFrame,
    gt: pd.DataFrame,
    tier: int,
    rc: RewardComputer,
) -> DataForgeEnv:
    local_corruptor = Corruptor()
    local_corruptor.force_tier(tier)
    eval_env = DataForgeEnv(local_corruptor, HEALTHCARE_SCHEMA, clean_data)
    starting_acc = rc._field_accuracy(dirty, gt)
    eval_env._state = dirty.copy()
    eval_env._ground_truth = gt.copy()
    eval_env._original_dirty = dirty.copy()
    eval_env._prev_accuracy = starting_acc
    eval_env._starting_accuracy = starting_acc
    eval_env._step_count = 0
    eval_env._action_log = []
    eval_env._episode_rewards = []
    eval_env._episode_start = time.time()
    return eval_env


def _build_results_payload(
    *,
    agent_config: dict,
    tier: int,
    episodes: int,
    max_steps: int,
    seed: int,
    surgeon_delta: float,
    random_delta: float,
    surgeon_win_rate: float,
    random_win_rate: float,
) -> dict:
    advantage = surgeon_delta - random_delta
    payload = {
        "agent_mode": agent_config["agent_mode"],
        "model_source": agent_config["model_source"],
        "fallback_used": agent_config["fallback_used"],
        "tier": tier,
        "episodes": episodes,
        "max_steps": max_steps,
        "seed": seed,
        "surgeon_avg_accuracy_delta": round(float(surgeon_delta), 6),
        "random_avg_accuracy_delta": round(float(random_delta), 6),
        "surgeon_advantage_accuracy_delta": round(float(advantage), 6),
        "surgeon_win_rate": round(float(surgeon_win_rate), 6),
        "random_win_rate": round(float(random_win_rate), 6),
    }
    if agent_config["agent_mode"] == "heuristic":
        payload["note"] = "Heuristic surgeon results. No trained GRPO checkpoint was used."
    return payload


def evaluate(
    n_episodes: int = 10,
    tier: int = 1,
    max_steps: int = 5,
    agent_mode: str = "heuristic",
    model_path: str | None = None,
    seed: int = 7,
) -> dict:
    agent_config = resolve_eval_agent(agent_mode, model_path)
    clean_data = pd.read_csv("data/healthcare_clean.csv")
    corruptor = Corruptor()
    corruptor.force_tier(tier)
    rc = RewardComputer()

    random.seed(seed)
    np.random.seed(seed)

    if agent_config["agent_mode"] == "grpo":
        load_eval_pipeline(agent_config["model_path"])

    results = {
        "random": {"before": [], "after": [], "deltas": []},
        "surgeon": {"before": [], "after": [], "deltas": []},
    }

    surgeon_label = "Heuristic Surgeon" if agent_config["agent_mode"] == "heuristic" else "GRPO Surgeon"
    surgeon_agent_fn = (
        (lambda state, target_gt, eval_env: heuristic_surgeon_agent(state, target_gt, HEALTHCARE_SCHEMA))
        if agent_config["agent_mode"] == "heuristic"
        else (lambda state, target_gt, eval_env: grpo_surgeon_agent(state, target_gt, eval_env))
    )

    print(f"\n{'=' * 60}")
    print("  DataForge Arena - Evaluation Report")
    print(
        f"  Mode: {agent_config['agent_mode']} | Episodes: {n_episodes} | "
        f"Tier: {tier} | Max Steps: {max_steps} | Seed: {seed}"
    )
    print(f"{'=' * 60}\n")

    for episode_idx in range(n_episodes):
        sample = clean_data.sample(
            n=min(50, len(clean_data)),
            random_state=seed + episode_idx,
        ).reset_index(drop=True)
        dirty, gt, meta = corruptor.generate_episode(sample)
        gt = _align_duplicate_ground_truth(dirty, gt, meta)
        acc_before = rc._field_accuracy(dirty, gt)

        agents = [
            ("random", lambda state, target_gt, eval_env: random_baseline_agent(state, target_gt)),
            ("surgeon", surgeon_agent_fn),
        ]

        for agent_name, agent_fn in agents:
            eval_env = _bootstrap_eval_env(clean_data, dirty, gt, tier, rc)
            for _ in range(max_steps):
                action = agent_fn(eval_env._state.copy(), gt, eval_env)
                _, _, done, _ = eval_env.step(action)
                if done:
                    break

            state_after = eval_env._state.copy()
            acc_after = rc._field_accuracy(state_after, gt)
            delta = acc_after - acc_before

            results[agent_name]["before"].append(acc_before)
            results[agent_name]["after"].append(acc_after)
            results[agent_name]["deltas"].append(delta)

        print(
            f"  Episode {episode_idx + 1:2d}/{n_episodes} | corruption={meta['tool']:25s} | "
            f"random: {results['random']['deltas'][-1]:+.3f} | "
            f"{agent_config['agent_mode']}: {results['surgeon']['deltas'][-1]:+.3f}"
        )

    print(f"\n{'-' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'-' * 60}")

    for agent_name, label in [("random", "Random Baseline"), ("surgeon", surgeon_label)]:
        metrics = results[agent_name]
        avg_before = np.mean(metrics["before"])
        avg_after = np.mean(metrics["after"])
        avg_delta = np.mean(metrics["deltas"])
        win_rate = sum(1 for delta in metrics["deltas"] if delta > 0) / max(len(metrics["deltas"]), 1)
        print(f"\n  {label}:")
        print(f"    Avg accuracy before:  {avg_before:.4f}")
        print(f"    Avg accuracy after:   {avg_after:.4f}")
        print(f"    Avg accuracy delta:   {avg_delta:+.4f} ({avg_delta * 100:+.2f}%)")
        print(f"    Win rate (delta > 0): {win_rate:.2%}")

    surgeon_delta = np.mean(results["surgeon"]["deltas"])
    random_delta = np.mean(results["random"]["deltas"])
    surgeon_win_rate = sum(1 for delta in results["surgeon"]["deltas"] if delta > 0) / max(len(results["surgeon"]["deltas"]), 1)
    random_win_rate = sum(1 for delta in results["random"]["deltas"] if delta > 0) / max(len(results["random"]["deltas"]), 1)

    payload = _build_results_payload(
        agent_config=agent_config,
        tier=tier,
        episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        surgeon_delta=surgeon_delta,
        random_delta=random_delta,
        surgeon_win_rate=surgeon_win_rate,
        random_win_rate=random_win_rate,
    )

    print(f"\n{'=' * 60}")
    print(
        "  HEADLINE: "
        f"{surgeon_label} outperforms random by "
        f"{payload['surgeon_advantage_accuracy_delta'] * 100:+.2f}% accuracy delta"
    )
    print(f"{'=' * 60}\n")

    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("  Results saved to eval/results.json")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--agent-mode", type=str, default="heuristic", choices=["heuristic", "grpo"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        evaluate(
            n_episodes=args.episodes,
            tier=args.tier,
            max_steps=args.steps,
            agent_mode=args.agent_mode,
            model_path=args.model_path,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
