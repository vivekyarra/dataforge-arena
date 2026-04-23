"""
DataForge Arena -- Inference Script
Test your newly trained model on a heavily corrupted row!
"""
import pandas as pd
from unsloth import FastLanguageModel
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.prompt import build_prompt
from environment.env import DataForgeEnv
from environment.schemas import HEALTHCARE_SCHEMA
from environment.corruptor import Corruptor

def test_inference():
    print("Loading model and LoRA adapters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="outputs/dataforge-surgeon",  # Loads the LoRA adapter we trained
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print("\nSetting up Environment...")
    clean_data = pd.read_csv("data/healthcare_clean.csv")
    corruptor = Corruptor()
    env = DataForgeEnv(corruptor=corruptor, schema=HEALTHCARE_SCHEMA, clean_data=clean_data)
    
    # Intentionally ramp up the difficulty
    corruptor.difficulty = 3 
    obs = env.reset()
    
    print("\n--- Current State (Corrupted!) ---")
    print(obs.head(3).to_markdown())

    prompt = build_prompt(obs)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print("\nRunning Inference...")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        use_cache=True, 
        temperature=0.7
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract only the completion part
    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1]
        
    print("\n--- Agent's Action ---")
    print(response)

if __name__ == "__main__":
    test_inference()
