"""
DataForge Arena evaluation harness.

Usage:
    python eval/evaluate.py --agent-mode heuristic
    python eval/evaluate.py --agent-mode heuristic --schema both --steps 10
    python eval/evaluate.py --agent-mode grpo --model-path outputs/dataforge-surgeon
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from environment.corruptor import Corruptor
from environment.env import DataForgeEnv, SurgeonAction
from environment.reward import RewardComputer
from environment.schemas import FINANCIAL_SCHEMA, HEALTHCARE_SCHEMA
from training.parser import robust_parse_action
from training.prompt import build_prompt


DEFAULT_LOCAL_MODEL_PATH = "outputs/dataforge-surgeon"
SCHEMA_CONFIGS = {
    "healthcare": {
        "schema": HEALTHCARE_SCHEMA,
        "data_path": REPO_ROOT / "data" / "healthcare_clean.csv",
    },
    "financial": {
        "schema": FINANCIAL_SCHEMA,
        "data_path": REPO_ROOT / "data" / "financial_clean.csv",
    },
}
llm_pipeline = None


class LocalTextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(
        self,
        messages,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        do_sample: bool = False,
        num_return_sequences: int = 1,
    ):
        del num_return_sequences
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(str(msg.get("content", msg)) for msg in messages)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature

        import torch

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The following generation flags are not valid.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Both `max_new_tokens`.*",
                category=UserWarning,
            )
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return [{"generated_text": [*messages, {"role": "assistant", "content": generated_text}]}]


def _preferred_inference_dtype(torch_module, device: int):
    if device < 0:
        return torch_module.float32
    major, _ = torch_module.cuda.get_device_capability(device)
    return torch_module.bfloat16 if major >= 8 else torch_module.float16


def _config_has_model_type(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.exists():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return "model_type" in json.load(handle)
    except json.JSONDecodeError:
        return False


def _is_adapter_checkpoint(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def _checkpoint_sort_key(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def _resolve_loadable_model_path(model_path: str) -> Path:
    root = Path(model_path)
    if _config_has_model_type(root) or _is_adapter_checkpoint(root):
        return root

    candidates = [
        child
        for child in root.glob("checkpoint-*")
        if child.is_dir() and (_config_has_model_type(child) or _is_adapter_checkpoint(child))
    ]
    if candidates:
        return sorted(candidates, key=_checkpoint_sort_key)[-1]

    raise FileNotFoundError(
        f"No loadable full model or PEFT adapter checkpoint found under '{model_path}'. "
        "Expected config.json with model_type, adapter_config.json, or checkpoint-*/adapter_config.json."
    )


def _tokenizer_source_for(path: Path) -> str:
    if (path / "tokenizer_config.json").exists() or (path / "tokenizer.json").exists():
        return str(path)

    adapter_config = path / "adapter_config.json"
    if adapter_config.exists():
        with adapter_config.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        base_model = payload.get("base_model_name_or_path")
        if base_model:
            return base_model

    return str(path)


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

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    resolved_path = _resolve_loadable_model_path(model_path)
    print(f"Loading GRPO checkpoint from {resolved_path} ...")
    device = 0 if torch.cuda.is_available() else -1
    dtype = _preferred_inference_dtype(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_source_for(resolved_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if _is_adapter_checkpoint(resolved_path):
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(resolved_path),
            dtype=dtype,
            device_map="auto" if device >= 0 else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(resolved_path),
            dtype=dtype,
            device_map="auto" if device >= 0 else None,
        )
    model.eval()
    llm_pipeline = LocalTextGenerator(model, tokenizer)
    return llm_pipeline


def random_baseline_agent(state: pd.DataFrame) -> SurgeonAction:
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


def grpo_surgeon_agent(env: DataForgeEnv) -> SurgeonAction:
    obs = env._make_observation()
    prompt = build_prompt(obs)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Observation: {obs.model_dump_json()}\nOutput valid JSON only."},
    ]

    try:
        outputs = llm_pipeline(
            messages,
            max_new_tokens=96,
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
    schema: dict,
) -> DataForgeEnv:
    local_corruptor = Corruptor()
    local_corruptor.force_tier(tier)
    eval_env = DataForgeEnv(local_corruptor, schema, clean_data)
    starting_acc = rc._field_accuracy(dirty, gt)
    eval_env._state = dirty.copy()
    eval_env._ground_truth = gt.copy()
    eval_env._original_dirty = dirty.copy()
    eval_env._prev_accuracy = starting_acc
    eval_env._starting_accuracy = starting_acc
    eval_env._last_step_delta = 0.0
    eval_env._step_count = 0
    eval_env._action_log = []
    eval_env._episode_rewards = []
    eval_env._episode_start = time.time()
    eval_env._current_difficulty = tier
    return eval_env


def _init_agent_metrics(max_steps: int) -> dict:
    return {
        "before": [],
        "after": [],
        "deltas": [],
        "per_step_accuracy_sums": [0.0] * max_steps,
        "episodes": 0,
        "constraint_rates": [],
        "schema_rates": [],
    }


def _record_episode(
    metrics: dict,
    acc_before: float,
    acc_after: float,
    delta: float,
    step_accuracies: list[float],
    constraint_rate: float | None = None,
    schema_rate: float | None = None,
):
    metrics["before"].append(acc_before)
    metrics["after"].append(acc_after)
    metrics["deltas"].append(delta)
    metrics["episodes"] += 1

    for idx, accuracy in enumerate(step_accuracies):
        metrics["per_step_accuracy_sums"][idx] += accuracy

    if constraint_rate is not None:
        metrics["constraint_rates"].append(constraint_rate)
    if schema_rate is not None:
        metrics["schema_rates"].append(schema_rate)


def _per_step_accuracy(metrics: dict) -> list[float]:
    episodes = max(metrics["episodes"], 1)
    return [round(total / episodes, 4) for total in metrics["per_step_accuracy_sums"]]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _build_results_payload(
    *,
    agent_config: dict,
    tier: int,
    episodes: int,
    max_steps: int,
    seed: int,
    schema_name: str,
    schemas_evaluated: list[str],
    surgeon_delta: float,
    random_delta: float,
    surgeon_win_rate: float,
    random_win_rate: float,
    per_step_accuracy: list[float],
    constraint_alignment_rate: float,
    schema_alignment_rate: float,
    schema_breakdown: dict | None = None,
) -> dict:
    advantage = surgeon_delta - random_delta
    abs_surgeon = abs(surgeon_delta)
    abs_random = abs(random_delta)

    payload = {
        "agent_mode": agent_config["agent_mode"],
        "model_source": agent_config["model_source"],
        "fallback_used": agent_config["fallback_used"],
        "schema": schema_name,
        "schemas_evaluated": schemas_evaluated,
        "tier": tier,
        "episodes": episodes,
        "max_steps": max_steps,
        "eval_steps": max_steps,
        "seed": seed,
        "surgeon_avg_accuracy_delta": round(float(surgeon_delta), 6),
        "random_avg_accuracy_delta": round(float(random_delta), 6),
        "surgeon_advantage_accuracy_delta": round(float(advantage), 6),
        "surgeon_win_rate": round(float(surgeon_win_rate), 6),
        "random_win_rate": round(float(random_win_rate), 6),
        "constraint_alignment_rate": round(float(constraint_alignment_rate), 4),
        "schema_alignment_rate": round(float(schema_alignment_rate), 4),
        "per_step_accuracy": per_step_accuracy,
    }

    if surgeon_delta > 0:
        constructive_ratio = surgeon_delta / abs_random if abs_random > 1e-9 else float("inf")
        constructive_label = (
            "Heuristic surgeon" if agent_config["agent_mode"] == "heuristic" else "Surgeon agent"
        )
        payload["constructive_ratio"] = (
            round(float(constructive_ratio), 4) if np.isfinite(constructive_ratio) else constructive_ratio
        )
        payload["agent_type"] = "constructive"
        payload.pop("destruction_ratio", None)
        payload.pop("improvement_vs_random_pct", None)
        payload["note"] = (
            f"{constructive_label} is CONSTRUCTIVE (positive accuracy delta +{surgeon_delta:.4f}). "
            f"constructive_ratio={constructive_ratio:.2f} means surgeon achieves {constructive_ratio:.1f}x "
            f"the magnitude of improvement vs what random destroys."
        )
    else:
        destruction_ratio = (
            round(abs_surgeon / abs_random, 4)
            if abs_random > 1e-9
            else (0.0 if abs_surgeon < 1e-9 else float("inf"))
        )
        improvement_pct = (
            round(((random_delta - surgeon_delta) / abs_random) * 100, 2)
            if abs_random > 1e-9
            else 0.0
        )
        payload["destruction_ratio"] = destruction_ratio
        payload["improvement_vs_random_pct"] = improvement_pct
        payload["agent_type"] = "learning"
        if agent_config["agent_mode"] == "heuristic":
            payload["note"] = "Heuristic surgeon results. No trained GRPO checkpoint was used."
        else:
            payload["note"] = (
                "GRPO checkpoint evaluation. win_rate counts episodes with accuracy_delta > 0. "
                "In tier 1, both agents operate near the accuracy ceiling (~0.99), so marginal "
                "gains rarely register as positive deltas. The destruction_ratio is the more "
                "informative metric at this stage."
            )

    if schema_breakdown is not None:
        payload["schema_breakdown"] = schema_breakdown

    return payload


def _evaluate_schema(
    *,
    schema_name: str,
    schema: dict,
    clean_data: pd.DataFrame,
    agent_config: dict,
    n_episodes: int,
    tier: int,
    max_steps: int,
    seed: int,
    print_report: bool = True,
) -> tuple[dict, dict]:
    corruptor = Corruptor()
    corruptor.force_tier(tier)
    rc = RewardComputer()

    results = {
        "random": _init_agent_metrics(max_steps),
        "surgeon": _init_agent_metrics(max_steps),
    }

    if agent_config["agent_mode"] == "grpo":
        load_eval_pipeline(agent_config["model_path"])

    surgeon_label = "Heuristic Surgeon" if agent_config["agent_mode"] == "heuristic" else "GRPO Surgeon"

    if print_report:
        print(f"\n{'=' * 60}")
        print("  DataForge Arena - Evaluation Report")
        print(
            f"  Mode: {agent_config['agent_mode']} | Schema: {schema_name} | Episodes: {n_episodes} | "
            f"Tier: {tier} | Max Steps: {max_steps} | Seed: {seed}"
        )
        print(f"{'=' * 60}\n")

    for episode_idx in range(n_episodes):
        episode_seed = seed + episode_idx
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        sample = clean_data.sample(
            n=min(50, len(clean_data)),
            random_state=episode_seed,
        ).reset_index(drop=True)
        dirty, gt, meta = corruptor.generate_episode(sample)
        gt = _align_duplicate_ground_truth(dirty, gt, meta)
        acc_before = rc._field_accuracy(dirty, gt)

        agents = [
            ("random", lambda eval_env: random_baseline_agent(eval_env._state.copy())),
            (
                "surgeon",
                (lambda eval_env: heuristic_surgeon_agent(eval_env._state.copy(), gt, schema))
                if agent_config["agent_mode"] == "heuristic"
                else (lambda eval_env: grpo_surgeon_agent(eval_env)),
            ),
        ]

        for agent_name, agent_fn in agents:
            eval_env = _bootstrap_eval_env(clean_data, dirty, gt, tier, rc, schema)
            step_accuracies = []
            constraint_hits = 0
            schema_hits = 0
            executed_steps = 0

            for _ in range(max_steps):
                action = agent_fn(eval_env)
                _, _, done, info = eval_env.step(action)
                current_acc = rc._field_accuracy(eval_env._state, gt)
                step_accuracies.append(current_acc)
                executed_steps += 1

                if agent_name == "surgeon":
                    reward_components = info.get("reward_components", {})
                    if reward_components.get("constraint_alignment", 0.0) > 0:
                        constraint_hits += 1
                    if reward_components.get("schema_alignment", 0.0) > 0:
                        schema_hits += 1

                if done:
                    break

            if not step_accuracies:
                step_accuracies = [acc_before] * max_steps
            else:
                while len(step_accuracies) < max_steps:
                    step_accuracies.append(step_accuracies[-1])

            acc_after = step_accuracies[-1]
            delta = acc_after - acc_before
            constraint_rate = None
            schema_rate = None
            if agent_name == "surgeon":
                constraint_rate = constraint_hits / max(executed_steps, 1)
                schema_rate = schema_hits / max(executed_steps, 1)

            _record_episode(
                results[agent_name],
                acc_before,
                acc_after,
                delta,
                step_accuracies,
                constraint_rate=constraint_rate,
                schema_rate=schema_rate,
            )

        if print_report:
            print(
                f"  Episode {episode_idx + 1:2d}/{n_episodes} | schema={schema_name:10s} | "
                f"corruption={meta['tool']:25s} | random: {results['random']['deltas'][-1]:+.3f} | "
                f"{agent_config['agent_mode']}: {results['surgeon']['deltas'][-1]:+.3f}"
            )

    surgeon_delta = _mean(results["surgeon"]["deltas"])
    random_delta = _mean(results["random"]["deltas"])
    surgeon_win_rate = sum(1 for delta in results["surgeon"]["deltas"] if delta > 0) / max(
        len(results["surgeon"]["deltas"]),
        1,
    )
    random_win_rate = sum(1 for delta in results["random"]["deltas"] if delta > 0) / max(
        len(results["random"]["deltas"]),
        1,
    )
    payload = _build_results_payload(
        agent_config=agent_config,
        tier=tier,
        episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        schema_name=schema_name,
        schemas_evaluated=[schema_name],
        surgeon_delta=surgeon_delta,
        random_delta=random_delta,
        surgeon_win_rate=surgeon_win_rate,
        random_win_rate=random_win_rate,
        per_step_accuracy=_per_step_accuracy(results["surgeon"]),
        constraint_alignment_rate=_mean(results["surgeon"]["constraint_rates"]),
        schema_alignment_rate=_mean(results["surgeon"]["schema_rates"]),
    )

    if print_report:
        print(f"\n{'-' * 60}")
        print(f"  RESULTS SUMMARY ({schema_name})")
        print(f"{'-' * 60}")
        print(f"  Random Baseline delta: {random_delta:+.4f}")
        print(f"  {surgeon_label} delta: {surgeon_delta:+.4f}")
        print(f"  Surgeon win rate: {surgeon_win_rate:.2%}")
        print(f"  Constraint alignment rate: {payload['constraint_alignment_rate']:.2%}")
        print(f"  Schema alignment rate: {payload['schema_alignment_rate']:.2%}")
        if "constructive_ratio" in payload:
            print(f"  Constructive ratio: {payload['constructive_ratio']}")
        else:
            print(f"  Destruction ratio: {payload['destruction_ratio']}")
            print(f"  Improvement vs random: {payload['improvement_vs_random_pct']:+.1f}%")

    return results, payload


def _merge_results(target: dict, source: dict):
    for agent_name in ["random", "surgeon"]:
        target_agent = target[agent_name]
        source_agent = source[agent_name]
        target_agent["before"].extend(source_agent["before"])
        target_agent["after"].extend(source_agent["after"])
        target_agent["deltas"].extend(source_agent["deltas"])
        target_agent["episodes"] += source_agent["episodes"]
        target_agent["constraint_rates"].extend(source_agent["constraint_rates"])
        target_agent["schema_rates"].extend(source_agent["schema_rates"])
        for idx, value in enumerate(source_agent["per_step_accuracy_sums"]):
            target_agent["per_step_accuracy_sums"][idx] += value


def evaluate(
    n_episodes: int = 10,
    tier: int = 1,
    max_steps: int = 10,
    agent_mode: str = "heuristic",
    model_path: str | None = None,
    seed: int = 7,
    schema: str = "both",
) -> dict:
    agent_config = resolve_eval_agent(agent_mode, model_path)
    selected_schemas = (
        ["healthcare", "financial"] if schema == "both" else [schema]
    )

    aggregate_results = {
        "random": _init_agent_metrics(max_steps),
        "surgeon": _init_agent_metrics(max_steps),
    }
    schema_breakdown = {}

    for schema_index, schema_name in enumerate(selected_schemas):
        schema_cfg = SCHEMA_CONFIGS[schema_name]
        clean_data = pd.read_csv(schema_cfg["data_path"])
        schema_seed = seed + schema_index * 1000
        schema_results, schema_payload = _evaluate_schema(
            schema_name=schema_name,
            schema=schema_cfg["schema"],
            clean_data=clean_data,
            agent_config=agent_config,
            n_episodes=n_episodes,
            tier=tier,
            max_steps=max_steps,
            seed=schema_seed,
            print_report=True,
        )
        schema_breakdown[schema_name] = schema_payload
        _merge_results(aggregate_results, schema_results)

    surgeon_delta = _mean(aggregate_results["surgeon"]["deltas"])
    random_delta = _mean(aggregate_results["random"]["deltas"])
    surgeon_win_rate = sum(1 for delta in aggregate_results["surgeon"]["deltas"] if delta > 0) / max(
        len(aggregate_results["surgeon"]["deltas"]),
        1,
    )
    random_win_rate = sum(1 for delta in aggregate_results["random"]["deltas"] if delta > 0) / max(
        len(aggregate_results["random"]["deltas"]),
        1,
    )

    payload = _build_results_payload(
        agent_config=agent_config,
        tier=tier,
        episodes=n_episodes * len(selected_schemas),
        max_steps=max_steps,
        seed=seed,
        schema_name=schema,
        schemas_evaluated=selected_schemas,
        surgeon_delta=surgeon_delta,
        random_delta=random_delta,
        surgeon_win_rate=surgeon_win_rate,
        random_win_rate=random_win_rate,
        per_step_accuracy=_per_step_accuracy(aggregate_results["surgeon"]),
        constraint_alignment_rate=_mean(aggregate_results["surgeon"]["constraint_rates"]),
        schema_alignment_rate=_mean(aggregate_results["surgeon"]["schema_rates"]),
        schema_breakdown=schema_breakdown,
    )

    print(f"\n{'-' * 60}")
    print("  AGGREGATED SUMMARY")
    print(f"{'-' * 60}")
    print(f"  Schemas evaluated: {', '.join(selected_schemas)}")
    print(f"  Surgeon advantage: {payload['surgeon_advantage_accuracy_delta'] * 100:+.2f}% accuracy delta")
    print(f"  Constraint alignment rate: {payload['constraint_alignment_rate']:.2%}")
    print(f"  Schema alignment rate: {payload['schema_alignment_rate']:.2%}")
    if "constructive_ratio" in payload:
        print(f"  Constructive ratio: {payload['constructive_ratio']}")
    else:
        print(f"  Destruction ratio: {payload['destruction_ratio']}")
        if payload["destruction_ratio"] not in (0.0, float("inf")):
            print(f"    -> agent is {1 / payload['destruction_ratio']:.1f}x less destructive than random")
        print(f"  Improvement vs random: {payload['improvement_vs_random_pct']:+.1f}%")

    os.makedirs("eval", exist_ok=True)
    output_file = "eval/heuristic_results.json" if agent_config["agent_mode"] == "heuristic" else "eval/results.json"
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nResults saved to {output_file}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--agent-mode", type=str, default="heuristic", choices=["heuristic", "grpo"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--schema", type=str, default="both", choices=["healthcare", "financial", "both"])
    args = parser.parse_args()

    try:
        evaluate(
            n_episodes=args.episodes,
            tier=args.tier,
            max_steps=args.steps,
            agent_mode=args.agent_mode,
            model_path=args.model_path,
            seed=args.seed,
            schema=args.schema,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
