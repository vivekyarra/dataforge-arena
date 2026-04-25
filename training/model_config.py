"""
Auto-detects available GPU and selects the appropriate model.
Run this FIRST on campus before anything else.
"""
import subprocess


def detect_gpu() -> dict:
    """Returns GPU info needed for model selection."""
    # BUG 3 FIX: Lazy-import torch so pytest doesn't crash at collection
    try:
        import torch
    except ImportError:
        return {"type": "none", "vram_gb": 0}
    
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
        
        capability = torch.cuda.get_device_capability(0)
        return {
            "type": name,
            "vram_gb": round(vram_gb, 1),
            "capability": f"{capability[0]}.{capability[1]}",
        }
    except Exception:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        capability = torch.cuda.get_device_capability(0)
        return {
            "type": "unknown",
            "vram_gb": round(vram_gb, 1),
            "capability": f"{capability[0]}.{capability[1]}",
        }


def select_precision(gpu_info: dict) -> dict:
    """Select a TrainingArguments precision mode that the detected GPU supports."""
    if gpu_info.get("vram_gb", 0) <= 0:
        return {"bf16": False, "fp16": False, "label": "fp32-cpu"}

    capability = gpu_info.get("capability")
    if capability is not None:
        try:
            major = int(str(capability).split(".")[0])
            use_bf16 = major >= 8
            return {
                "bf16": use_bf16,
                "fp16": not use_bf16,
                "label": "bf16" if use_bf16 else "fp16",
            }
        except (TypeError, ValueError):
            pass

    gpu_name = str(gpu_info.get("type", "")).lower()
    bf16_name_markers = ("a100", "h100", "h200", "l4", "l40", "rtx 30", "rtx 40", "rtx 50")
    use_bf16 = any(marker in gpu_name for marker in bf16_name_markers)
    return {
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "label": "bf16" if use_bf16 else "fp16",
    }


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
            "grad_accum": 8,
            "target_steps": 150,
            "dataset_size": 500,
            "max_completion_length": 128,
            "temperature": 0.5,
            "max_training_tier": 3,
            "label": "Llama 3.1 8B (full power)",
        }
    
    elif vram >= 20:  # A10G or L40S
        return {
            "model_name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
            "max_seq_length": 2048,
            "num_generations": 6,
            "batch_size": 1,
            "grad_accum": 6,
            "target_steps": 100,
            "dataset_size": 400,
            "max_completion_length": 112,
            "temperature": 0.45,
            "max_training_tier": 3,
            "label": "Llama 3.2 3B (balanced)",
        }
    
    else:  # T4 (16GB but slow) or L4
        return {
            "model_name": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
            "max_seq_length": 2048,
            "num_generations": 4,
            "batch_size": 1,
            "grad_accum": 4,
            "target_steps": 80,
            "dataset_size": 320,
            "max_completion_length": 96,
            "temperature": 0.35,
            "max_training_tier": 2,
            "label": "Qwen 2.5 1.5B (speed mode)",
        }


if __name__ == "__main__":
    gpu = detect_gpu()
    model = select_model(gpu)
    precision = select_precision(gpu)
    print(f"\nGPU detected: {gpu['type']} ({gpu['vram_gb']}GB VRAM)")
    print(f"Selected model: {model['label']}")
    print(f"Model ID: {model['model_name']}")
    print(f"Target steps: {model['target_steps']}")
    print(f"Rollouts (G): {model['num_generations']}")
    print(f"Precision: {precision['label']}")
