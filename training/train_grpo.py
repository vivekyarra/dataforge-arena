"""
DataForge Arena -- GRPO Training Script
Run on campus with HF compute credits.
"""
import os
import random as _random
import re as _re
import sys
import time as _time
import uuid as _uuid
import warnings

import pandas as pd
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

from environment.corruptor import Corruptor
from environment.env import DataForgeEnv
from environment.schemas import FINANCIAL_SCHEMA, HEALTHCARE_SCHEMA
from training.logger import TrainingLogger
from training.model_config import detect_gpu, select_model
from training.parser import robust_parse_action
from training.prompt import build_prompt


gpu_info = detect_gpu()
model_cfg = select_model(gpu_info)
print(f"\n{'=' * 50}")
print(f"GPU:   {gpu_info['type']} ({gpu_info['vram_gb']}GB)")
print(f"Model: {model_cfg['label']}")
print(f"Steps: {model_cfg['target_steps']}")
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
_EPISODE_KEY_RE = _re.compile(r"EPISODE_CACHE_KEY:\s*([0-9a-f]{32})")


def _tier_for_example(index: int, total_examples: int) -> int:
    progress = index / max(total_examples - 1, 1)
    if progress < 0.60:
        return 1
    if progress < 0.85:
        return 2
    return 3


def _attach_episode_key(prompt: str, episode_key: str) -> str:
    return f"{prompt}\nEPISODE_CACHE_KEY: {episode_key}"


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


def build_dataset(n=200) -> Dataset:
    prompts = []
    episode_cache.clear()
    for idx in range(n):
        if _random.random() < 0.5:
            env._schema = HEALTHCARE_SCHEMA
            env._clean_data = clean_data_hc
        else:
            env._schema = FINANCIAL_SCHEMA
            env._clean_data = clean_data_fin

        corruptor.force_tier(_tier_for_example(idx, n))
        obs = env.reset()
        episode_key = _uuid.uuid4().hex
        episode_cache[episode_key] = {
            "state": env._state.copy(),
            "gt": env._ground_truth.copy(),
            "original_dirty": env._original_dirty.copy(),
            "prev_accuracy": env._prev_accuracy,
            "starting_accuracy": env._starting_accuracy,
            "schema": env._schema,
            "difficulty": obs.difficulty,
        }
        prompts.append({"prompt": _attach_episode_key(build_prompt(obs), episode_key)})
    return Dataset.from_list(prompts)


print("Building training dataset...")
train_dataset = build_dataset(400)

current_step = [0]
parse_successes = [0]
total_rollouts = [0]


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

        try:
            action = robust_parse_action(completion)
            parse_successes[0] += 1
        except ValueError:
            rewards.append(-2.0)
            continue

        _, reward, _, info = env.step(action)
        recent_actions.append(action.model_dump())
        if len(recent_actions) > 100:
            recent_actions.pop(0)

        for key in component_accum:
            component_accum[key].append(info.get("reward_components", {}).get(key, 0.0))

        rewards.append(reward)

    if current_step[0] % 5 == 0:
        avg_components = {key: sum(values) / max(len(values), 1) for key, values in component_accum.items()}
        avg_reward = sum(rewards) / max(len(rewards), 1)
        logged_difficulty = max(batch_difficulties, default=corruptor.difficulty)
        logger.log(
            step=current_step[0],
            reward_dict={"total": avg_reward, **avg_components},
            difficulty=logged_difficulty,
            model_label=model_cfg["label"],
            parse_successes=parse_successes[0],
            total_rollouts=total_rollouts[0],
        )
        parse_rate = parse_successes[0] / max(total_rollouts[0], 1) * 100
        print(
            f"Step {current_step[0]:3d} | reward={avg_reward:+.3f} | "
            f"difficulty={logged_difficulty} | parse={parse_rate:.0f}%"
        )
        parse_successes[0] = 0
        total_rollouts[0] = 0
        logger.detect_collapse(recent_actions)

    current_step[0] += 1
    return rewards


print("\nStarting GRPO training...")
print(f"Target: {model_cfg['target_steps']} steps\n")

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=GRPOConfig(
        output_dir="outputs/dataforge-surgeon",
        num_generations=model_cfg["num_generations"],
        max_completion_length=256,
        temperature=0.9,
        beta=0.01,
        learning_rate=1e-5,
        per_device_train_batch_size=model_cfg["batch_size"],
        gradient_accumulation_steps=model_cfg["grad_accum"],
        num_train_epochs=3,
        logging_steps=5,
        save_steps=25,
        report_to="none",
        max_steps=model_cfg["target_steps"],
    ),
    train_dataset=train_dataset,
)

trainer.train()

print("\nTraining complete.")
print("Model saved to: outputs/dataforge-surgeon")
print("Training log:   logs/training_log.csv")
