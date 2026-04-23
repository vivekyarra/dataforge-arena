"""
DataForge Arena -- GRPO Training Script
Run on campus with HF compute credits.
"""
import os, sys, json, torch, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

from training.model_config import detect_gpu, select_model
from training.prompt import build_prompt
from training.parser import robust_parse_action
from training.logger import TrainingLogger
from environment.env import DataForgeEnv, SurgeonAction
from environment.corruptor import Corruptor
from environment.schemas import HEALTHCARE_SCHEMA

# -- Step 1: Detect compute and select model -----------------------
gpu_info = detect_gpu()
model_cfg = select_model(gpu_info)
print(f"\n{'='*50}")
print(f"GPU:   {gpu_info['type']} ({gpu_info['vram_gb']}GB)")
print(f"Model: {model_cfg['label']}")
print(f"Steps: {model_cfg['target_steps']}")
print(f"{'='*50}\n")

# -- Step 2: Load environment ---------------------------------------
clean_data = pd.read_csv("data/healthcare_clean.csv")
corruptor = Corruptor()
env = DataForgeEnv(corruptor=corruptor,
                   schema=HEALTHCARE_SCHEMA,
                   clean_data=clean_data)
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
def build_dataset(n=200) -> Dataset:
    prompts = []
    for _ in range(n):
        obs = env.reset()
        prompts.append({"prompt": build_prompt(obs)})
    return Dataset.from_list(prompts)

print("Building training dataset...")
train_dataset = build_dataset(200)

# -- Step 5: Reward function with dynamic KL beta -------------------
current_step = [0]  # mutable for closure
parse_successes = [0]
total_rollouts = [0]

def reward_fn(completions: list, prompts: list, **kwargs) -> list:
    rewards = []
    obs = env.reset()
    
    for completion in completions:
        total_rollouts[0] += 1
        try:
            action = robust_parse_action(completion)
            parse_successes[0] += 1
        except ValueError:
            rewards.append(-2.0)
            continue
        
        _, reward_dict, done, info = env.step(action)
        
        # Log action for collapse detection
        recent_actions.append(action.dict())
        if len(recent_actions) > 100:
            recent_actions.pop(0)
        
        rewards.append(reward_dict["total"])
    
    # Log every 5 steps
    if current_step[0] % 5 == 0:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        logger.log(
            step=current_step[0],
            reward_dict={"total": avg_reward,
                         "accuracy_delta": 0, "tool_logic": 0,
                         "reasoning": 0, "efficiency": 0, "anti_hack": 0},
            difficulty=corruptor.difficulty,
            model_label=model_cfg["label"],
            parse_successes=parse_successes[0],
            total_rollouts=total_rollouts[0],
        )
        print(f"Step {current_step[0]:3d} | reward={avg_reward:+.3f} | "
              f"difficulty={corruptor.difficulty} | "
              f"parse_ok={parse_successes[0]}/{total_rollouts[0]}")
        parse_successes[0] = 0
        total_rollouts[0] = 0
        
        # Collapse detection
        logger.detect_collapse(recent_actions)
    
    current_step[0] += 1
    return rewards

# -- Step 6: Safe advantage normalization --------------------------
def safe_advantage_norm(rewards):
    """Handles std=0 -- prevents NaN gradients."""
    R = torch.tensor(rewards, dtype=torch.float32)
    std = R.std()
    if std < 1e-8:
        return [0.0] * len(rewards)  # skip update
    return ((R - R.mean()) / (std + 1e-8)).tolist()

# -- Step 7: Dynamic KL beta (fixes catastrophic forgetting) -------
def get_beta() -> float:
    """
    Raise KL beta during tier transitions to prevent
    policy from updating too aggressively on unfamiliar distributions.
    """
    if corruptor.is_transitioning():
        return 0.05   # 5x higher during transition -- tight leash
    return 0.01       # normal

# -- Step 8: Train -------------------------------------------------
print("\nStarting GRPO training...")
print(f"Target: {model_cfg['target_steps']} steps\n")

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=GRPOConfig(
        output_dir="outputs/dataforge-surgeon",
        num_generations=model_cfg["num_generations"],
        max_new_tokens=256,
        temperature=0.9,
        beta=get_beta(),
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
