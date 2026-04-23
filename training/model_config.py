"""
Auto-detects available GPU and selects the appropriate model.
Run this FIRST on campus before anything else.
"""
import subprocess
import torch


def detect_gpu() -> dict:
    """Returns GPU info needed for model selection."""
    if not torch.cuda.is_available():
        return {"type": "none", "vram_gb": 0}
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True
        )
        line = result.stdout.strip().split("\n")[0]
        name = line.split(",")[0].strip()
        vram_mb = int(line.split(",")[1].strip().split(" ")[0])
        vram_gb = vram_mb / 1024
        
        return {"type": name, "vram_gb": round(vram_gb, 1)}
    except Exception:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {"type": "unknown", "vram_gb": round(vram_gb, 1)}


def select_model(gpu_info: dict) -> dict:
    """
    Tier decision tree based on available VRAM.
    A fully trained 1.5B with a clean reward curve BEATS
    a timed-out 8B. Always pick what finishes.
    """
    vram = gpu_info["vram_gb"]
    
    if vram >= 35:  # A100 40GB or H100
        return {
            "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "max_seq_length": 2048,
            "num_generations": 8,
            "batch_size": 1,
            "grad_accum": 4,
            "target_steps": 150,
            "label": "Llama 3.1 8B (full power)",
        }
    
    elif vram >= 14:  # A10G or L40S
        return {
            "model_name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
            "max_seq_length": 2048,
            "num_generations": 6,
            "batch_size": 1,
            "grad_accum": 4,
            "target_steps": 100,
            "label": "Llama 3.2 3B (balanced)",
        }
    
    else:  # T4 (16GB but slow) or L4
        return {
            "model_name": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
            "max_seq_length": 1024,
            "num_generations": 4,
            "batch_size": 1,
            "grad_accum": 2,
            "target_steps": 80,
            "label": "Qwen 2.5 1.5B (speed mode)",
        }


if __name__ == "__main__":
    gpu = detect_gpu()
    model = select_model(gpu)
    print(f"\nGPU detected: {gpu['type']} ({gpu['vram_gb']}GB VRAM)")
    print(f"Selected model: {model['label']}")
    print(f"Model ID: {model['model_name']}")
    print(f"Target steps: {model['target_steps']}")
    print(f"Rollouts (G): {model['num_generations']}")
