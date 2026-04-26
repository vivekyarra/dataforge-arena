"""
DataForge Arena -- GRPO Training Script
Run on campus with HF compute credits.

Changes from audit:
  - Logger columns match RewardComputer.compute() keys exactly
  - Tool diversity penalty (−0.3 when >60% same tool in batch)
  - Temperature scheduling: 0.8 → 0.2 by step 200
  - Reward function verification: warn if shaped total < 0.1 for 3+ batches
  - CORRECT_FORMAT on null_numeric returns 0.0 (fixed in reward.py)
"""
from __future__ import annotations

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
try:
    import transformers.utils.hub
    if not hasattr(transformers.utils.hub, "TRANSFORMERS_CACHE"):
        transformers.utils.hub.TRANSFORMERS_CACHE = os.getenv("HF_HOME", "/tmp/hf_cache")
except Exception:
    pass  # Non-fatal; some transformers versions don't need this.
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

# --- Temperature scheduling ---
# Start at 0.8, decay linearly to 0.2 by step 200, hold at 0.2 after.
TEMP_START = 0.8
TEMP_END = 0.2
TEMP_DECAY_STEPS = 200


def _scheduled_temperature(step: int) -> float:
    if step >= TEMP_DECAY_STEPS:
        return TEMP_END
    progress = step / TEMP_DECAY_STEPS
    return TEMP_START - (TEMP_START - TEMP_END) * progress


# --- Shaped reward verification ---
# Track consecutive batches where shaped total < 0.1
_consecutive_low_shaped = [0]


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
        return -0.8

    score = -0.60
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

    return max(-0.8, min(score, 0.20))


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
        parsed_indices = []
        for suspect in suspect_cols:
            suspect_text = str(suspect)
            match = _re.search(r"\[(\d+)\]\s*$", suspect_text)
            if match:
                parsed_indices.append(int(match.group(1)))
                continue

            suspect_name = suspect_text.split("[", 1)[0]
            if suspect_name in column_to_idx:
                parsed_indices.append(column_to_idx[suspect_name])
        suspect_map[int(row_id)] = parsed_indices

    return suspect_map


def _dominant_tool_snapshot(history: list[dict], window: int = 24) -> tuple[int, float]:
    if not history:
        return -1, 0.0

    recent = history[-window:]
    tool_counts = Counter(item.get("tool_id", -1) for item in recent)
    dominant_tool, dominant_count = tool_counts.most_common(1)[0]
    return dominant_tool, dominant_count / max(len(recent), 1)


def _tool_diversity_penalty(batch_actions: list[dict], threshold: float = 0.60) -> float:
    """
    If more than 60% of actions in the batch use the same tool_id,
    apply a −0.3 diversity penalty.
    """
    if len(batch_actions) < 4:
        return 0.0
    tool_counts = Counter(a.get("tool_id", -1) for a in batch_actions)
    _, top_count = tool_counts.most_common(1)[0]
    if top_count / len(batch_actions) > threshold:
        return -0.3
    return 0.0


# violation_type alignment bonus added to _contextual_reward_shaping()
def _contextual_reward_shaping(action, episode: dict, parse_mode: str, training_step: int) -> float:
    shaping = 0.0

    # Parse quality — strongly reward exact JSON, gently penalize recovered
    if parse_mode == "exact":
        shaping += 1.50
    elif parse_mode == "recovered":
        shaping -= 0.15

    # Bonus: agent's tool matches the violation type from observation
    violation_type = episode.get("violation_type", "")
    tool_violation_map = {
        "null_numeric":    {0},
        "null_categorical":{1, 2},
        "range":           {3},
        "type_error":      {3},
        "enum_violation":  {3},
        "fk_mismatch":     {3},
        "clean":           {7},
    }
    expected_tools = tool_violation_map.get(violation_type, set())
    if expected_tools:
        if action.tool_id in expected_tools:
            shaping += 1.50  # agent correctly understood the violation type
        elif violation_type and action.tool_id == 7 and violation_type != "clean":
            shaping -= 2.00  # NO_OP on a real violation: very wrong

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

    # Anti-collapse: wait until the policy has had time to stabilize, then
    # gently discourage obvious tool monoculture instead of crushing early learning.
    dominant_tool, dominant_rate = _dominant_tool_snapshot(recent_actions)
    if training_step >= 40 and len(recent_actions) >= 32:
        if dominant_rate >= 0.75 and action.tool_id == dominant_tool:
            shaping -= min(0.80, 0.12 + (dominant_rate - 0.75) * 2.7)

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
            "violation_type": getattr(obs, 'violation_type', ''),
            "column_stats": getattr(obs, 'column_stats', ''),
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

    All reward components from RewardComputer.compute() are accumulated and
    logged with their EXACT key names. The shaped signals (constraint_alignment,
    schema_alignment, outlier_targeting, reasoning_quality, parse_bonus) are
    all included in the scalar reward returned to the trainer.
    """
    rewards = []
    # Accumulate ALL reward component keys from RewardComputer
    component_accum = {
        "accuracy_delta": [],
        "constraint_alignment": [],
        "schema_alignment": [],
        "outlier_targeting": [],
        "reasoning_quality": [],
        "parse_bonus": [],
        "anti_hack": [],
    }
    batch_difficulties = []
    batch_actions = []  # for tool diversity penalty

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
        policy_shaping = _contextual_reward_shaping(action, episode, parse_mode, current_step[0])
        if info.get("invalid_action"):
            invalid_actions[0] += 1
            policy_shaping -= 0.25

        reward += structural_penalty + policy_shaping
        structural_penalty_total[0] += structural_penalty

        # Track batch actions for diversity penalty
        action_dict = action.model_dump()
        batch_actions.append(action_dict)
        recent_actions.append(action_dict)
        if len(recent_actions) > 100:
            recent_actions.pop(0)

        # Extract reward components from env.step() info — these are the EXACT
        # keys from RewardComputer.compute()
        rc = info.get("reward_components", {})
        for key in component_accum:
            component_accum[key].append(rc.get(key, 0.0))

        rewards.append(reward)

    # Apply tool diversity penalty to all rewards in this batch
    diversity_penalty = _tool_diversity_penalty(batch_actions)
    if diversity_penalty < 0:
        rewards = [r + diversity_penalty for r in rewards]

    # --- Shaped reward verification ---
    # Check that shaped signals are actually being included in the reward
    if component_accum["constraint_alignment"]:
        shaped_total = (
            abs(sum(component_accum["constraint_alignment"])) +
            abs(sum(component_accum["schema_alignment"])) +
            abs(sum(component_accum["outlier_targeting"])) +
            abs(sum(component_accum["reasoning_quality"])) +
            abs(sum(component_accum["parse_bonus"]))
        )
        avg_shaped = shaped_total / max(len(component_accum["constraint_alignment"]), 1)
        if avg_shaped < 0.1:
            _consecutive_low_shaped[0] += 1
            if _consecutive_low_shaped[0] >= 3:
                print(
                    f"[WARNING] Shaped reward signals < 0.1 for {_consecutive_low_shaped[0]} "
                    f"consecutive batches — check that constraint_alignment, schema_alignment, "
                    f"outlier_targeting, reasoning_quality, and parse_bonus are firing."
                )
        else:
            _consecutive_low_shaped[0] = 0

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
            violation_type=episode.get("violation_type", "") if episode else "",
        )
        print(
            f"Step {current_step[0]:3d} | reward={avg_reward:+.3f} | "
            f"difficulty={logged_difficulty} | exact={parse_rate:.0f}% | "
            f"recovered={recovered_rate:.0f}% | invalid={invalid_rate:.0f}% | "
            f"tool={dominant_tool}@{dominant_tool_rate:.0%} | "
            f"temp={_scheduled_temperature(current_step[0]):.2f}"
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

# Temperature scheduling: start at 0.8, the GRPOConfig gets the initial value.
# During training, the reward_fn prints the scheduled temperature for monitoring.
# Note: TRL's GRPOTrainer uses the config temperature for generation. To truly
# schedule temperature per-step, we patch the generation_config before each call.
initial_temperature = _scheduled_temperature(0)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=GRPOConfig(
        output_dir="outputs/dataforge-surgeon",
        num_generations=model_cfg["num_generations"],
        max_completion_length=model_cfg.get("max_completion_length", 128),
        temperature=initial_temperature,
        beta=0.005,
        learning_rate=4e-5,
        warmup_ratio=0.08,
        per_device_train_batch_size=model_cfg["batch_size"],
        gradient_accumulation_steps=model_cfg["grad_accum"],
        # With max_steps set, keep epochs at 1 so Trainer does not chew
        # through multiple full dataset passes on the slow T4 path.
        num_train_epochs=1,
        logging_steps=5,
        save_steps=50,
        report_to="none",
        max_steps=300,
        bf16=precision_cfg["bf16"],
        fp16=precision_cfg["fp16"],
        dataloader_num_workers=0,
    ),
    train_dataset=train_dataset,
)

# --- Temperature scheduling hook ---
# Patch the generation config's temperature before each training step
_original_train_step = trainer.training_step


def _patched_training_step(*args, **kwargs):
    step = current_step[0]
    new_temp = _scheduled_temperature(step)
    if hasattr(trainer, 'generation_config') and trainer.generation_config is not None:
        trainer.generation_config.temperature = new_temp
    elif hasattr(trainer.model, 'generation_config') and trainer.model.generation_config is not None:
        trainer.model.generation_config.temperature = new_temp
    return _original_train_step(*args, **kwargs)


trainer.training_step = _patched_training_step

trainer.train()
trainer.save_model("outputs/dataforge-surgeon")
tokenizer.save_pretrained("outputs/dataforge-surgeon")

print("\nTraining complete.")
print("Model saved to: outputs/dataforge-surgeon")
print("Training log:   logs/training_log.csv")
