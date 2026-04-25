"""
DataForge Arena -- GRPO Training Script
Run on campus with HF compute credits.
"""
import os, sys, json, warnings
import torch, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Silence noisy deprecation warnings from transformers/pydantic
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

from training.model_config import detect_gpu, select_model
from training.prompt import build_prompt
from training.parser import robust_parse_action
from training.logger import TrainingLogger
from environment.env import DataForgeEnv, SurgeonAction
from environment.corruptor import Corruptor
from environment.schemas import HEALTHCARE_SCHEMA, FINANCIAL_SCHEMA

# -- Step 1: Detect compute and select model -----------------------
gpu_info = detect_gpu()
model_cfg = select_model(gpu_info)
print(f"\n{'='*50}")
print(f"GPU:   {gpu_info['type']} ({gpu_info['vram_gb']}GB)")
print(f"Model: {model_cfg['label']}")
print(f"Steps: {model_cfg['target_steps']}")
print(f"{'='*50}\n")

# -- Step 2: Load environment ---------------------------------------
clean_data_hc = pd.read_csv("data/healthcare_clean.csv")
clean_data_fin = pd.read_csv("data/financial_clean.csv")

corruptor = Corruptor()
env = DataForgeEnv(corruptor=corruptor,
                   schema=HEALTHCARE_SCHEMA,
                   clean_data=clean_data_hc)
logger = TrainingLogger(path="logs/training_log.csv")
recent_actions = []

# -- Step 3: Load model ---------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_cfg["model_name"],
    max_seq_length=model_cfg["max_seq_length"],
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=16, lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# -- Step 4: Build training dataset --------------------------------
# CRITICAL FIX: Cache full episode state per prompt so reward_fn evaluates
# the model's action on the SAME corrupted data it saw, not a random new one.
import time as _time
import random as _random
episode_cache = {}  # prompt_hash -> episode state

def build_dataset(n=200) -> Dataset:
    prompts = []
    for _ in range(n):
        if _random.random() < 0.5:
            env._schema = HEALTHCARE_SCHEMA
            env._clean_data = clean_data_hc
        else:
            env._schema = FINANCIAL_SCHEMA
            env._clean_data = clean_data_fin
        obs = env.reset()
        prompt = build_prompt(obs)
        # Cache the episode state so reward_fn can restore it
        episode_cache[hash(prompt)] = {
            "state": env._state.copy(),
            "gt": env._ground_truth.copy(),
            "original_dirty": env._original_dirty.copy(),
            "prev_accuracy": env._prev_accuracy,
            "starting_accuracy": env._starting_accuracy,
            "schema": env._schema,
        }
        prompts.append({"prompt": prompt})
    return Dataset.from_list(prompts)

print("Building training dataset...")
train_dataset = build_dataset(200)

# -- Step 5: Reward function ----------------------------------------
current_step = [0]  # mutable for closure
parse_successes = [0]
total_rollouts = [0]

def reward_fn(completions: list, prompts: list, **kwargs) -> list:
    """
    Evaluate each completion by restoring the EXACT episode state from
    the prompt's cached data. This ensures the model is rewarded for
    actions on the same corrupted data it saw in its prompt.
    """
    rewards = []
    component_accum = {
        "accuracy_delta": [], "tool_logic": [],
        "reasoning": [], "efficiency": [], "anti_hack": []
    }

    for i, completion in enumerate(completions):
        total_rollouts[0] += 1

        # Restore the cached episode state matching this prompt
        prompt = prompts[i] if i < len(prompts) else ""
        prompt_key = hash(prompt)
        ep = episode_cache.get(prompt_key)

        if ep is not None:
            env._state = ep["state"].copy()
            env._ground_truth = ep["gt"].copy()
            env._original_dirty = ep["original_dirty"].copy()
            env._prev_accuracy = ep["prev_accuracy"]
            env._starting_accuracy = ep["starting_accuracy"]
            env._schema = ep["schema"]
            env._step_count = 0
            env._action_log = []
            env._episode_rewards = []
            env._episode_start = _time.time()
        else:
            env.reset()

        try:
            action = robust_parse_action(completion)
            parse_successes[0] += 1
        except ValueError:
            rewards.append(-2.0)
            continue

        _, reward, done, info = env.step(action)

        recent_actions.append(action.model_dump())
        if len(recent_actions) > 100:
            recent_actions.pop(0)

        for k in component_accum:
            component_accum[k].append(info.get("reward_components", {}).get(k, 0))

        rewards.append(reward)

    # Advance corruptor epoch every training step
    corruptor.record_episode(sum(rewards) / max(len(rewards), 1))
    
    # Log every 5 steps
    if current_step[0] % 5 == 0:
        avg_components = {k: sum(v)/max(len(v),1) for k, v in component_accum.items()}
        avg_reward = sum(rewards) / max(len(rewards), 1)
        logger.log(
            step=current_step[0],
            reward_dict={"total": avg_reward, **avg_components},
            difficulty=corruptor.difficulty,
            model_label=model_cfg["label"],
            parse_successes=parse_successes[0],
            total_rollouts=total_rollouts[0],
        )
        parse_rate = parse_successes[0] / max(total_rollouts[0], 1) * 100
        print(f"Step {current_step[0]:3d} | reward={avg_reward:+.3f} | "
              f"difficulty={corruptor.difficulty} | "
              f"parse={parse_rate:.0f}%")
        parse_successes[0] = 0
        total_rollouts[0] = 0
        
        # Collapse detection
        logger.detect_collapse(recent_actions)
    
    current_step[0] += 1
    return rewards

# -- Step 6: Train -------------------------------------------------
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
        # NOTE: GRPOConfig beta is set once at init. Dynamic beta would
        # require subclassing GRPOTrainer.compute_loss, which is not worth
        # the complexity. A fixed beta of 0.01 works well for 80-step runs
        # on T4. For longer runs on A100 (150+ steps), consider 0.005.
        beta=0.01,
        learning_rate=5e-6,
        per_device_train_batch_size=model_cfg["batch_size"],
        gradient_accumulation_steps=model_cfg["grad_accum"],
        num_train_epochs=3,
        logging_steps=5,
        save_steps=25,
        report_to="none",     # change to "wandb" if you have account set up
        max_steps=model_cfg["target_steps"],
    ),
    train_dataset=train_dataset,
)

trainer.train()

print("\nTraining complete.")
print(f"Model saved to: outputs/dataforge-surgeon")
print(f"Training log:   logs/training_log.csv")
