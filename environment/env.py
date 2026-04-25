import time
import json
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Optional
from openenv.env import Env as BaseEnv

from environment.schemas import SURGEON_TOOLS
from environment.reward import RewardComputer
from environment.tools import apply_tool
from environment.corruptor import Corruptor


class SurgeonAction(BaseModel):
    reasoning: str
    tool_id: int
    column: int
    row_id: int


class DataForgeObservation(BaseModel):
    rows_json: str          # top 5 most corrupted rows as JSON string
    schema_str: str         # schema summary
    step_count: int
    max_steps: int
    difficulty: int
    total_rows: int
    total_errors: int
    error_rate_pct: float
    action_history: list


class DataForgeEnv(BaseEnv):
    MAX_STEPS = 20
    MAX_SECONDS = 30

    def __init__(self, corruptor: Corruptor, schema: dict,
                 clean_data: pd.DataFrame):
        self._corruptor = corruptor
        self._schema = schema
        self._clean_data = clean_data
        self._reward_computer = RewardComputer()
        
        # Episode state
        self._state: pd.DataFrame = None
        self._ground_truth: pd.DataFrame = None
        self._original_dirty: pd.DataFrame = None
        self._prev_accuracy: float = 0.0
        self._starting_accuracy: float = 0.0
        self._episode_start: float = None
        self._step_count: int = 0
        self._action_log: list = []
        self._episode_rewards: list = []

    def reset(self) -> DataForgeObservation:
        # Sample 50 rows for each episode (keeps prompts short)
        n_samples = min(50, len(self._clean_data))
        sample = self._clean_data.sample(n=n_samples, random_state=None).reset_index(drop=True)
        dirty, ground_truth, metadata = self._corruptor.generate_episode(sample)
        
        # BUG 7 FIX: If duplicate_row_mutate added rows, extend ground_truth
        # to match dirty's length so _field_accuracy doesn't get a shape mismatch
        if metadata.get("tool") == "duplicate_row_mutate" and len(dirty) > len(ground_truth):
            # The extra row is a duplicate -- append the original row to ground_truth
            src_row = metadata.get("row", 0)
            if src_row < len(ground_truth):
                extra = ground_truth.iloc[[src_row]].copy()
                ground_truth = pd.concat([ground_truth, extra], ignore_index=True)
        
        self._state = dirty.copy()
        self._ground_truth = ground_truth.copy()
        self._original_dirty = dirty.copy()
        self._prev_accuracy = self._reward_computer._field_accuracy(
            self._state, self._ground_truth
        )
        self._starting_accuracy = self._prev_accuracy
        self._episode_start = time.time()
        self._step_count = 0
        self._action_log = []
        self._episode_rewards = []
        
        return self._make_observation()

    def step(self, action: SurgeonAction) -> tuple[DataForgeObservation, float, bool, dict]:
        # Validate
        if not self._is_valid(action):
            return (self._make_observation(),
                    -0.5,
                    False, {"invalid_action": True})
        
        self._step_count += 1
        self._action_log.append(action.model_dump())
        
        # Apply tool
        self._state = apply_tool(self._state, action, self._schema)
        
        # Compute rewards
        reward_dict = self._reward_computer.compute(
            state=self._state,
            ground_truth=self._ground_truth,
            action=action,
            original_dirty=self._original_dirty,
            prev_accuracy=self._prev_accuracy,
            episode_start=self._episode_start,
            step_count=self._step_count,
            starting_accuracy=self._starting_accuracy,
        )
        
        # Update prev accuracy for next step's delta
        if "_current_accuracy" in reward_dict:
            self._prev_accuracy = reward_dict.pop("_current_accuracy")
        
        self._episode_rewards.append(reward_dict["total"])
        
        done = (
            self._step_count >= self.MAX_STEPS or
            reward_dict.get("episode_complete", False) or
            reward_dict.get("timeout", False)
        )
        
        if done:
            self._corruptor.record_episode(sum(self._episode_rewards))
        
        total_reward = float(reward_dict.pop("total", 0.0))
        # Clean control signals from reward components
        reward_dict.pop("episode_complete", None)
        reward_dict.pop("timeout", None)
        
        return self._make_observation(), total_reward, done, {
            "action_log": self._action_log,
            "step": self._step_count,
            "reward_components": reward_dict,
        }

    def _is_valid(self, action: SurgeonAction) -> bool:
        if action.tool_id not in range(8):
            return False
        if self._state is None:
            return False
        if action.row_id >= len(self._state):
            return False
        # Exclude _is_deleted from valid column targets
        data_cols = [c for c in self._state.columns if c != "_is_deleted"]
        if action.column >= len(data_cols):
            return False
        return True

    def _make_observation(self) -> DataForgeObservation:
        state_clean = self._state.drop(columns=["_is_deleted"], errors="ignore")
        
        # CRITICAL FIX: Score rows by ALL corruption types, not just nulls.
        # Type errors (ERR_XX), format mismatches, and out-of-range values
        # were previously invisible because they're not null.
        corruption_scores = []
        total_errors = 0
        for idx in range(len(state_clean)):
            row = state_clean.iloc[idx]
            score = 0
            for col_name in state_clean.columns:
                val = row[col_name]
                if pd.isna(val):
                    score += 1
                    total_errors += 1
                elif isinstance(val, str) and val.startswith("ERR_"):
                    score += 1
                    total_errors += 1
                elif col_name in self._schema:
                    col_type = self._schema[col_name].get("type", "str")
                    if col_type in ("int", "float"):
                        try:
                            float(val)
                        except (ValueError, TypeError):
                            score += 1
                            total_errors += 1
            corruption_scores.append(score)
        
        # Show top 5 most corrupted rows (by ALL error types)
        scored = sorted(enumerate(corruption_scores), key=lambda x: -x[1])
        top_idx = [i for i, _ in scored[:5]]
        top_rows = state_clean.iloc[top_idx]
        
        # CRITICAL: Include row indices so the model knows which row_id to target.
        # Without this, the model sees rows but can't map them to valid row_id values.
        rows_safe = []
        for orig_idx, (_, row) in zip(top_idx, top_rows.iterrows()):
            record = {"_row_idx": int(orig_idx)}
            for col in top_rows.columns:
                val = row[col]
                record[col] = None if pd.isna(val) else val
            rows_safe.append(record)
        
        total_cells = state_clean.size
        
        schema_str = ", ".join([
            f"{k}:{v['type']}" for k, v in self._schema.items()
        ])
        
        return DataForgeObservation(
            rows_json=json.dumps(rows_safe),
            schema_str=schema_str,
            step_count=self._step_count,
            max_steps=self.MAX_STEPS,
            difficulty=self._corruptor.difficulty,
            total_rows=len(state_clean),
            total_errors=total_errors,
            error_rate_pct=round(100 * total_errors / max(total_cells, 1), 1),
            action_history=self._action_log[-2:],
        )
