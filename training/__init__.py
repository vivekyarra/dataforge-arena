"""
DataForge Arena — Training Package
GRPO training pipeline with GPU-aware model selection.
"""

from training.model_config import detect_gpu, select_model
from training.prompt import build_prompt
from training.parser import robust_parse_action
from training.logger import TrainingLogger

__all__ = [
    "detect_gpu",
    "select_model",
    "build_prompt",
    "robust_parse_action",
    "TrainingLogger",
]
