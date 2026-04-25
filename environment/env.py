import json
import time

import numpy as np
import pandas as pd
from openenv.env import Env as BaseEnv
from pydantic import BaseModel

from environment.corruptor import Corruptor
from environment.reward import RewardComputer
from environment.tools import apply_tool
from environment.validation import summarize_corruption_details


class SurgeonAction(BaseModel):
    reasoning: str
    tool_id: int
    column: int
    row_id: int


class DataForgeObservation(BaseModel):
    rows_json: str
    schema_str: str
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
    DISPLAY_ROW_LIMIT = 4

    def __init__(self, corruptor: Corruptor, schema: dict, clean_data: pd.DataFrame):
        self._corruptor = corruptor
        self._schema = schema
        self._clean_data = clean_data
        self._reward_computer = RewardComputer()

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
        n_samples = min(50, len(self._clean_data))
        sample = self._clean_data.sample(n=n_samples, random_state=None).reset_index(drop=True)
        dirty, ground_truth, metadata = self._corruptor.generate_episode(sample)

        if metadata.get("tool") == "duplicate_row_mutate" and len(dirty) > len(ground_truth):
            src_row = metadata.get("row", 0)
            if src_row < len(ground_truth):
                extra = ground_truth.iloc[[src_row]].copy()
                ground_truth = pd.concat([ground_truth, extra], ignore_index=True)

        self._state = dirty.copy()
        self._ground_truth = ground_truth.copy()
        self._original_dirty = dirty.copy()
        self._prev_accuracy = self._reward_computer._field_accuracy(self._state, self._ground_truth)
        self._starting_accuracy = self._prev_accuracy
        self._episode_start = time.time()
        self._step_count = 0
        self._action_log = []
        self._episode_rewards = []

        return self._make_observation()

    def step(self, action: SurgeonAction) -> tuple[DataForgeObservation, float, bool, dict]:
        if not self._is_valid(action):
            return self._make_observation(), -0.5, False, {"invalid_action": True}

        self._step_count += 1
        self._action_log.append(action.model_dump())
        previous_state = self._state.copy(deep=True)
        self._state = apply_tool(self._state, action, self._schema)

        reward_dict = self._reward_computer.compute(
            state=self._state,
            ground_truth=self._ground_truth,
            action=action,
            original_dirty=self._original_dirty,
            prev_accuracy=self._prev_accuracy,
            episode_start=self._episode_start,
            step_count=self._step_count,
            starting_accuracy=self._starting_accuracy,
            previous_state=previous_state,
        )

        if "_current_accuracy" in reward_dict:
            self._prev_accuracy = reward_dict.pop("_current_accuracy")

        self._episode_rewards.append(reward_dict["total"])
        done = (
            self._step_count >= self.MAX_STEPS
            or reward_dict.get("episode_complete", False)
            or reward_dict.get("timeout", False)
        )

        if done:
            self._corruptor.record_episode(sum(self._episode_rewards))

        total_reward = float(reward_dict.pop("total", 0.0))
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
        if action.row_id < 0 or action.column < 0:
            return False
        if action.row_id >= len(self._state):
            return False
        if "_is_deleted" in self._state.columns and bool(
            self._state.at[self._state.index[action.row_id], "_is_deleted"]
        ):
            return False

        data_cols = [c for c in self._state.columns if c != "_is_deleted"]
        if action.column >= len(data_cols):
            return False
        return True

    def _make_observation(self) -> DataForgeObservation:
        active_mask = pd.Series(True, index=self._state.index)
        if "_is_deleted" in self._state.columns:
            active_mask = ~self._state["_is_deleted"].fillna(False)
        state_clean = self._state.loc[active_mask].drop(columns=["_is_deleted"], errors="ignore")

        corruption_scores, total_errors, suspect_columns = summarize_corruption_details(
            state_clean,
            self._schema,
            max_issue_columns=3,
        )
        ranked_rows = sorted(
            zip(state_clean.index.tolist(), corruption_scores, suspect_columns),
            key=lambda item: (-item[1], item[0]),
        )
        top_idx = [row_idx for row_idx, _, _ in ranked_rows[: self.DISPLAY_ROW_LIMIT]]
        suspect_by_row = {row_idx: issues for row_idx, _, issues in ranked_rows}
        error_score_by_row = {row_idx: score for row_idx, score, _ in ranked_rows}
        top_rows = state_clean.loc[top_idx] if top_idx else state_clean.iloc[0:0]

        rows_safe = []
        for orig_idx, row in top_rows.iterrows():
            record = {"_row_idx": int(orig_idx)}
            record["_error_score"] = int(error_score_by_row.get(orig_idx, 0))
            record["_suspect_columns"] = suspect_by_row.get(orig_idx, [])
            for col_name in top_rows.columns:
                value = row[col_name]
                if pd.isna(value):
                    record[col_name] = None
                elif isinstance(value, (np.integer, np.floating)):
                    record[col_name] = value.item()
                else:
                    record[col_name] = value
            rows_safe.append(record)

        total_cells = state_clean.size
        schema_str = ", ".join([f"{name}:{info['type']}" for name, info in self._schema.items()])

        return DataForgeObservation(
            rows_json=json.dumps(rows_safe),
            schema_str=schema_str,
            step_count=self._step_count,
            max_steps=self.MAX_STEPS,
            difficulty=self._corruptor.difficulty,
            total_rows=int(len(state_clean)),
            total_errors=total_errors,
            error_rate_pct=round(100 * total_errors / max(total_cells, 1), 1),
            action_history=self._action_log[-2:],
        )
