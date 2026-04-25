"""
DataForge Arena -- GRPO Training Script
Run on campus with HF compute credits.
"""
import json
import os
from pathlib import Path
import random as _random
import re as _re
import sys
import time as _time
import uuid as _uuid
import warnings
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

# Unsloth must patch transformers/TRL before they are imported.
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import pandas as pd
import torch
from datasets import Dataset

# --- HOTFIX FOR TRANSFORMERS HUB COMPAT ---
import transformers.utils.hub
if not hasattr(transformers.utils.hub, "TRANSFORMERS_CACHE"):
    transformers.utils.hub.TRANSFORMERS_CACHE = os.getenv("HF_HOME", "/tmp/hf_cache")
# -----------------------------------------------

from trl import GRPOConfig, GRPOTrainer

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", message=r"Both `max_new_tokens`.*", category=UserWarning)

from environment.corruptor import Corruptor
from environment.env import DataForgeEnv
from environment.schemas import FINANCIAL_SCHEMA, HEALTHCARE_SCHEMA
from training.logger import TrainingLogger
from training.model_config import detect_gpu, select_model, select_precision
from training.parser import robust_parse_action
from training.prompt import build_prompt


gpu_info = detect_gpu()
model_cfg = select_model(gpu_info)
precision_cfg = select_precision(gpu_info)
print(f"\n{'=' * 50}")
print(f"GPU:   {gpu_info['type']} ({gpu_info['vram_gb']}GB)")
print(f"Model: {model_cfg['label']}")
print(f"Steps: {model_cfg['target_steps']}")
print(f"Precision: {precision_cfg['label']}")
print(f"{'=' * 50}\n")

clean_data_hc = pd.read_csv("data/healthcare_clean.csv")
clean_data_fin = pd.read_csv("data/financial_clean.csv")

corruptor = Corruptor()
env = DataForgeEnv(corruptor=corruptor, schema=HEALTHCARE_SCHEMA, clean_data=clean_data_hc)
logger = TrainingLogger(path="logs/training_log.csv")
recent_actions = []

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_cfg["model_name"],
    max_seq_length=model_cfg["max_seq_length"],
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

episode_cache = {}
_EPISODE_KEY_RE = _re.compile(r"EPISODE_CACHE_KEY:\s*([0-9a-f]{12})")


def _tier_for_example(index: int, total_examples: int) -> int:
    progress = index / max(total_examples - 1, 1)
    if model_cfg.get("max_training_tier", 3) <= 2:
        tier2_fraction = float(model_cfg.get("tier2_fraction", 0.10))
        return 1 if progress < (1.0 - tier2_fraction) else 2
    if progress < 0.60:
        return 1
    if progress < 0.85:
        return 2
    return 3


def _attach_episode_key(prompt: str, episode_key: str) -> str:
    return f"EPISODE_CACHE_KEY: {episode_key}\n{prompt}"


def _prompt_to_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        return str(prompt.get("content", prompt))
    if isinstance(prompt, list):
        return "\n".join(_prompt_to_text(part) for part in prompt)
    return str(prompt)


def _extract_episode_key(prompt) -> str | None:
    match = _EPISODE_KEY_RE.search(_prompt_to_text(prompt))
    return match.group(1) if match else None


def _completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion["content"])
        return str(completion)
    if isinstance(completion, list):
        return "\n".join(_completion_to_text(item) for item in completion)
    return str(completion)


def _format_progress_reward(completion) -> float:
    text = _completion_to_text(completion).strip()
    if not text:
        return -1.5

    score = -1.25
    lowered = text.lower()
    if text.startswith("{"):
        score += 0.45
    if "{" in text and "}" in text:
        score += 0.25
    if "```" in text:
        score -= 0.15
    if "\n" not in text:
        score += 0.10
    if len(text) <= 220:
        score += 0.15
    elif len(text) > 400:
        score -= 0.20

    key_groups = (
        ("reasoning",),
        ("tool_id", "tool", "tool_name", "action"),
        ("column", "col", "column_idx", "col_idx"),
        ("row_id", "row_idx", "_row_idx", "row"),
    )
    for aliases in key_groups:
        if any(alias in lowered for alias in aliases):
            score += 0.20

    return max(-1.5, min(score, -0.10))


def _extract_suspect_column_indices(rows_json: str, schema: dict) -> dict[int, list[int]]:
    column_to_idx = {name: idx for idx, name in enumerate(schema.keys())}
    suspect_map = {}

    try:
        rows = json.loads(rows_json)
    except (TypeError, json.JSONDecodeError):
        return suspect_map

    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("_row_idx")
        if row_id is None:
            continue
        suspect_cols = row.get("_suspect_columns", [])
        suspect_map[int(row_id)] = [
            column_to_idx[name] for name in suspect_cols if name in column_to_idx
        ]

    return suspect_map


def _dominant_tool_snapshot(history: list[dict], window: int = 24) -> tuple[int, float]:
    if not history:
        return -1, 0.0

    recent = history[-window:]
    tool_counts = Counter(item.get("tool_id", -1) for item in recent)
    dominant_tool, dominant_count = tool_counts.most_common(1)[0]
    return dominant_tool, dominant_count / max(len(recent), 1)


def _contextual_reward_shaping(action, episode: dict, parse_mode: str) -> float:
    shaping = 0.0

    # Parse quality — strongly reward exact JSON, gently penalize recovered
    if parse_mode == "exact":
        shaping += 1.00
    elif parse_mode == "recovered":
        shaping -= 0.25

    total_errors = int(episode.get("total_errors", 0))
    if total_errors > 0 and action.tool_id == 7:
        shaping -= 1.80  # NO_OP when there are errors: very bad
    elif total_errors > 0 and action.tool_id != 7:
        shaping += 0.15

    # Suspect column targeting — reward precision, penalize random targeting
    row_suspects = episode.get("suspect_column_indices", {}).get(int(action.row_id), [])
    if row_suspects and action.tool_id != 7:
        if int(action.column) in row_suspects:
            shaping += 1.25  # MASSIVE reward for targeting the exactly correct cell
        else:
            shaping -= 0.50  # wrong column in a suspect row
    elif total_errors > 0 and action.tool_id != 7:
        known_rows = episode.get("suspect_column_indices", {})
        if known_rows and int(action.row_id) not in known_rows:
            shaping -= 0.75  # HEAVY penalty for targeting a clean, non-suspect row

    if total_errors == 0 and action.tool_id == 7:
        shaping += 0.40
    elif total_errors == 0 and action.tool_id != 7:
        shaping -= 0.50

    reasoning = getattr(action, "reasoning", "").strip().lower()
    if 2 <= len(reasoning.split()) <= 8:
        shaping += 0.05
    elif len(reasoning.split()) > 10:
        shaping -= 0.10

    # Anti-collapse: brutally penalize tool monoculture to force exploration
    dominant_tool, dominant_rate = _dominant_tool_snapshot(recent_actions)
    if dominant_rate >= 0.40 and action.tool_id == dominant_tool:
        shaping -= min(2.50, 0.50 + (dominant_rate - 0.40) * 5.0)

    return shaping


def build_dataset(n=200) -> Dataset:
    prompts = []
    episode_cache.clear()
    for idx in range(n):
        if _random.random() < float(model_cfg.get("financial_mix_rate", 0.5)):
            env._schema = FINANCIAL_SCHEMA
            env._clean_data = clean_data_fin
        else:
            env._schema = HEALTHCARE_SCHEMA
            env._clean_data = clean_data_hc

        corruptor.force_tier(_tier_for_example(idx, n))
        obs = env.reset()
        episode_key = _uuid.uuid4().hex[:12]
        episode_cache[episode_key] = {
            "state": env._state.copy(),
            "gt": env._ground_truth.copy(),
            "original_dirty": env._original_dirty.copy(),
            "prev_accuracy": env._prev_accuracy,
            "starting_accuracy": env._starting_accuracy,
            "schema": env._schema,
            "difficulty": obs.difficulty,
            "total_errors": obs.total_errors,
            "suspect_column_indices": _extract_suspect_column_indices(obs.rows_json, env._schema),
        }
        prompts.append({"prompt": _attach_episode_key(build_prompt(obs), episode_key)})
    return Dataset.from_list(prompts)


print("Building training dataset...")
train_dataset = build_dataset(model_cfg.get("dataset_size", 400))

current_step = [0]
parse_successes = [0]
parse_recoveries = [0]
invalid_actions = [0]
total_rollouts = [0]
structural_penalty_total = [0.0]


def reward_fn(completions: list, prompts: list, **kwargs) -> list:
    """
    Evaluate each completion by restoring the exact cached episode state from
    the prompt. The reward loop must never blind-reset into a different sample.
    """
    rewards = []
    component_accum = {
        "accuracy_delta": [],
        "tool_logic": [],
        "reasoning": [],
        "efficiency": [],
        "anti_hack": [],
        "policy_shaping": [],
    }
    batch_difficulties = []

    for idx, completion in enumerate(completions):
        total_rollouts[0] += 1
        prompt = prompts[idx] if idx < len(prompts) else ""
        episode_key = _extract_episode_key(prompt)
        episode = episode_cache.get(episode_key) if episode_key else None

        if episode is None:
            rewards.append(-2.5)
            continue

        env._state = episode["state"].copy()
        env._ground_truth = episode["gt"].copy()
        env._original_dirty = episode["original_dirty"].copy()
        env._prev_accuracy = episode["prev_accuracy"]
        env._starting_accuracy = episode["starting_accuracy"]
        env._schema = episode["schema"]
        env._step_count = 0
        env._action_log = []
        env._episode_rewards = []
        env._episode_start = _time.time()
        batch_difficulties.append(episode["difficulty"])

        structural_penalty = 0.0
        parse_mode = "exact"
        try:
            action = robust_parse_action(completion, require_fields=True)
            parse_successes[0] += 1
        except ValueError:
            parse_mode = "recovered"
            structural_penalty = _format_progress_reward(completion)
            try:
                action = robust_parse_action(completion, require_fields=False)
                parse_recoveries[0] += 1
            except ValueError:
                structural_penalty_total[0] += structural_penalty
                rewards.append(structural_penalty)
                continue

        _, reward, _, info = env.step(action)
        policy_shaping = _contextual_reward_shaping(action, episode, parse_mode)
        if info.get("invalid_action"):
            invalid_actions[0] += 1
            policy_shaping -= 0.25

        reward += structural_penalty + policy_shaping
        structural_penalty_total[0] += structural_penalty
        recent_actions.append(action.model_dump())
        if len(recent_actions) > 100:
            recent_actions.pop(0)

        for key in component_accum:
            if key == "policy_shaping":
                component_accum[key].append(policy_shaping)
            else:
                component_accum[key].append(info.get("reward_components", {}).get(key, 0.0))

        rewards.append(reward)

    if current_step[0] % 5 == 0:
        avg_components = {key: sum(values) / max(len(values), 1) for key, values in component_accum.items()}
        avg_reward = sum(rewards) / max(len(rewards), 1)
        logged_difficulty = max(batch_difficulties, default=corruptor.difficulty)
        dominant_tool, dominant_tool_rate = _dominant_tool_snapshot(recent_actions)
        parse_rate = parse_successes[0] / max(total_rollouts[0], 1) * 100
        recovered_rate = parse_recoveries[0] / max(total_rollouts[0], 1) * 100
        invalid_rate = invalid_actions[0] / max(total_rollouts[0], 1) * 100
        logger.log(
            step=current_step[0],
            reward_dict={"total": avg_reward, **avg_components},
            difficulty=logged_difficulty,
            model_label=model_cfg["label"],
            parse_successes=parse_successes[0],
            total_rollouts=total_rollouts[0],
            parse_recoveries=parse_recoveries[0],
            invalid_actions=invalid_actions[0],
            avg_structural_penalty=structural_penalty_total[0] / max(total_rollouts[0], 1),
            dominant_tool=dominant_tool,
            dominant_tool_rate=dominant_tool_rate,
        )
        print(
            f"Step {current_step[0]:3d} | reward={avg_reward:+.3f} | "
            f"difficulty={logged_difficulty} | exact={parse_rate:.0f}% | "
            f"recovered={recovered_rate:.0f}% | invalid={invalid_rate:.0f}% | "
            f"tool={dominant_tool}@{dominant_tool_rate:.0%}"
        )
        parse_successes[0] = 0
        parse_recoveries[0] = 0
        invalid_actions[0] = 0
        total_rollouts[0] = 0
        structural_penalty_total[0] = 0.0
        logger.detect_collapse(recent_actions)

    current_step[0] += 1
    return rewards


print("\nStarting GRPO training...")
print(f"Target: {model_cfg['target_steps']} steps\n")

# TRL GRPOTrainer expects warnings_issued dict on the model object.
# PEFT wrapping breaks __getattr__ resolution so it raises AttributeError.
# Patch it directly before trainer init.
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=GRPOConfig(
        output_dir="outputs/dataforge-surgeon",
        num_generations=model_cfg["num_generations"],
        max_completion_length=model_cfg.get("max_completion_length", 128),
        temperature=model_cfg.get("temperature", 0.5),
        beta=0.01,
        learning_rate=2e-5,
        per_device_train_batch_size=model_cfg["batch_size"],
        gradient_accumulation_steps=model_cfg["grad_accum"],
        num_train_epochs=3,
        logging_steps=5,
        save_steps=25,
        report_to="none",
        max_steps=model_cfg["target_steps"],
        bf16=precision_cfg["bf16"],
        fp16=precision_cfg["fp16"],
    ),
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model("outputs/dataforge-surgeon")
tokenizer.save_pretrained("outputs/dataforge-surgeon")

print("\nTraining complete.")
print("Model saved to: outputs/dataforge-surgeon")
print("Training log:   logs/training_log.csv")
