"""
DataForge Arena -- Environment Package
OpenEnv-compliant adversarial data repair environment.

Usage:
    from environment import DataForgeEnv, Corruptor, HEALTHCARE_SCHEMA
"""

# Lazy re-exports: import from submodules only when accessed
# This prevents circular import issues in any Python version
from environment.schemas import HEALTHCARE_SCHEMA, FINANCIAL_SCHEMA, SURGEON_TOOLS, DEPT_MAP
from environment.corruptor import Corruptor
from environment.reward import RewardComputer
from environment.tools import apply_tool
from environment.env import DataForgeEnv, SurgeonAction, DataForgeObservation

__all__ = [
    "DataForgeEnv",
    "SurgeonAction",
    "DataForgeObservation",
    "Corruptor",
    "RewardComputer",
    "HEALTHCARE_SCHEMA",
    "FINANCIAL_SCHEMA",
    "SURGEON_TOOLS",
    "DEPT_MAP",
    "apply_tool",
]
