"""
DataForge Arena -- Inference Script
Test your newly trained model on a heavily corrupted row!
"""
import pandas as pd
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.prompt import build_prompt
from training.parser import robust_parse_action
from environment.env import DataForgeEnv, SurgeonAction
from environment.schemas import HEALTHCARE_SCHEMA, SURGEON_TOOLS
from environment.corruptor import Corruptor
from environment.reward import RewardComputer

def test_inference():
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. Run: pip install unsloth")
        return

    print("Loading model and LoRA adapters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="outputs/dataforge-surgeon",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print("\nSetting up Environment...")
    clean_data = pd.read_csv("data/healthcare_clean.csv")
    corruptor = Corruptor()
    env = DataForgeEnv(corruptor=corruptor, schema=HEALTHCARE_SCHEMA, clean_data=clean_data)
    rc = RewardComputer()

    # Ramp up difficulty via epoch (difficulty is a dynamic property)
    corruptor._epoch = 115  # tier 3
    obs = env.reset()

    print(f"\n--- Episode Info ---")
    print(f"Rows: {obs.total_rows} | Errors: {obs.total_errors} | "
          f"Error Rate: {obs.error_rate_pct}% | Difficulty: {obs.difficulty}/3")

    print(f"\n--- Corrupted Rows ---")
    rows = json.loads(obs.rows_json)
    for i, row in enumerate(rows[:3]):
        print(f"  Row {i}: {row}")

    prompt = build_prompt(obs)
    print(f"\n--- Prompt Length: {len(prompt)} chars (~{len(prompt)//4} tokens) ---")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print("\nRunning Inference...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.7,
    )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract only the completion part
    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1]

    print("\n--- Raw Model Output ---")
    print(response)

    # Try to parse and execute the action
    try:
        action = robust_parse_action(response)
        print(f"\n--- Parsed Action ---")
        print(f"  Tool: {SURGEON_TOOLS[action.tool_id]['name']} (id={action.tool_id})")
        print(f"  Target: row={action.row_id}, col={action.column}")
        print(f"  Reasoning: {action.reasoning}")

        # Execute
        obs2, reward_dict, done, info = env.step(action)
        print(f"\n--- Result ---")
        print(f"  Total Reward: {reward_dict['total']:+.3f}")
        print(f"  Accuracy Delta: {reward_dict.get('accuracy_delta', 0):+.3f}")
        print(f"  Tool Logic: {reward_dict.get('tool_logic', 0):+.3f}")
        print(f"  Episode Done: {done}")
    except ValueError as e:
        print(f"\n--- Parse Failed ---")
        print(f"  {e}")


if __name__ == "__main__":
    test_inference()
