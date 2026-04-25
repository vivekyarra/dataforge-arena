import logging
import random
import re
from collections import deque

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class Corruptor:
    TIER_EPOCH_GATES = {2: 30, 3: 70}
    TIER_REWARD_GATES = {2: 0.5, 3: 0.9}

    def __init__(self):
        self._epoch = 0
        self._recent_rewards = deque(maxlen=20)
        self._unlocked_tier = 1

    @property
    def difficulty(self) -> int:
        return self._unlocked_tier

    def record_episode(self, surgeon_reward: float):
        self._recent_rewards.append(surgeon_reward)
        self._epoch += 1
        self._update_tier()

    def force_tier(self, tier: int):
        tier = min(max(int(tier), 1), 3)
        self._recent_rewards.clear()
        self._epoch = {1: 0, 2: self.TIER_EPOCH_GATES[2], 3: self.TIER_EPOCH_GATES[3]}[tier]
        self._unlocked_tier = tier

    def _rolling_avg(self) -> float:
        if not self._recent_rewards:
            return -99.0
        return sum(self._recent_rewards) / len(self._recent_rewards)

    def _update_tier(self):
        for candidate_tier in [2, 3]:
            if self._unlocked_tier >= candidate_tier:
                continue
            epoch_ok = self._epoch >= self.TIER_EPOCH_GATES[candidate_tier]
            reward_ok = self._rolling_avg() >= self.TIER_REWARD_GATES[candidate_tier]
            if epoch_ok and reward_ok:
                self._unlocked_tier = candidate_tier
                logger.info(
                    "Corruptor tier %s unlocked: epoch=%s, rolling_avg=%.3f",
                    candidate_tier,
                    self._epoch,
                    self._rolling_avg(),
                )

    def current_tier(self) -> int:
        return self._unlocked_tier

    def is_transitioning(self) -> bool:
        for gate in self.TIER_EPOCH_GATES.values():
            if gate <= self._epoch < gate + 10:
                return True
        return False

    def generate_episode(self, clean_df: pd.DataFrame, max_retries: int = 10) -> tuple:
        """
        Returns (dirty_df, ground_truth, metadata)
        Solvability gate ensures every episode is learnable.
        """
        tier = self.current_tier()

        for _ in range(max_retries):
            dirty, metadata = self._corrupt(clean_df.copy(), tier)
            valid, _ = self._solvability_gate(dirty, clean_df, metadata)
            if valid:
                metadata = dict(metadata)
                metadata["requested_tier"] = tier
                metadata["difficulty"] = tier
                return dirty, clean_df.copy(), metadata

        logger.warning("Corruptor retry limit reached; falling back to tier 1")
        dirty, metadata = self._corrupt_tier1(clean_df.copy())
        metadata = dict(metadata)
        metadata["requested_tier"] = tier
        metadata["difficulty"] = 1
        metadata["fallback_from_tier"] = tier
        return dirty, clean_df.copy(), metadata

    def _corrupt(self, df: pd.DataFrame, tier: int) -> tuple:
        if tier == 1:
            return self._corrupt_tier1(df)
        if tier == 2:
            return self._corrupt_tier2(df)
        return self._corrupt_tier3(df)

    def _corrupt_tier1(self, df: pd.DataFrame) -> tuple:
        tool = random.choice(["inject_null_single", "inject_type_error"])
        col = random.choice(df.columns.tolist())
        row = random.randint(0, len(df) - 1)
        original_val = df.at[row, col]

        if tool == "inject_null_single":
            df.at[row, col] = np.nan
        else:
            df[col] = df[col].astype(object)
            df.at[row, col] = f"ERR_{random.randint(10, 99)}"

        return df, {"tool": tool, "col": col, "row": row, "original": original_val}

    def _corrupt_tier2(self, df: pd.DataFrame) -> tuple:
        tool = random.choice(["inject_null_cluster", "swap_date_format", "inject_out_of_range_age"])
        metadata = {"tool": tool}

        if tool == "inject_null_cluster":
            col = random.choice(df.columns.tolist())
            start = random.randint(0, max(0, len(df) - 5))
            rows = list(range(start, min(start + random.randint(3, 5), len(df))))
            for row in rows:
                df.at[row, col] = np.nan
            metadata.update({"col": col, "rows": rows})

        elif tool == "swap_date_format":
            date_cols = [c for c in df.columns if "date" in c.lower()]
            if not date_cols:
                return self._corrupt_tier1(df)
            col = random.choice(date_cols)
            row = random.randint(0, len(df) - 1)
            value = str(df.at[row, col])
            if re.match(r"\d{4}-\d{2}-\d{2}", value):
                parts = value.split("-")
                df.at[row, col] = f"{parts[1]}/{parts[2]}/{parts[0][2:]}"
            metadata.update({"col": col, "row": row})

        else:
            if "age" not in df.columns or "birth_year" not in df.columns:
                return self._corrupt_tier1(df)
            row = random.randint(0, len(df) - 1)
            df.at[row, "age"] = random.randint(130, 180)
            metadata.update({"col": "age", "row": row})

        return df, metadata

    def _corrupt_tier3(self, df: pd.DataFrame) -> tuple:
        tool = random.choice(["break_foreign_key", "duplicate_row_mutate"])
        metadata = {"tool": tool}

        if tool == "break_foreign_key":
            if "department_id" not in df.columns or "department_name" not in df.columns:
                return self._corrupt_tier2(df)
            row = random.randint(0, len(df) - 1)
            corrupt_col = random.choice(["department_id", "department_name"])
            if corrupt_col == "department_id":
                df.at[row, "department_id"] = random.randint(500, 9999)
            else:
                df.at[row, "department_name"] = "INVALID_DEPT"
            metadata.update(
                {
                    "col": corrupt_col,
                    "row": row,
                    "paired_col": "department_name" if corrupt_col == "department_id" else "department_id",
                }
            )

        else:
            row = random.randint(0, len(df) - 1)
            dup = df.iloc[row].copy()
            col = random.choice(df.columns.tolist())
            dup[col] = np.nan
            df = pd.concat([df, pd.DataFrame([dup])], ignore_index=True)
            metadata.update({"col": col, "row": row})

        return df, metadata

    def _solvability_gate(self, dirty_df: pd.DataFrame, ground_truth: pd.DataFrame, metadata: dict) -> tuple:
        tool = metadata.get("tool", "")

        if tool == "delete_row":
            return False, "row deletion - unrecoverable"

        if tool == "break_foreign_key":
            col = metadata.get("col")
            paired_col = metadata.get("paired_col")
            if not col or not paired_col:
                return False, "foreign-key corruption missing repair metadata"
            if col not in dirty_df.columns or paired_col not in dirty_df.columns:
                return False, "foreign-key corruption missing paired columns"

        for col in dirty_df.columns:
            null_rate = dirty_df[col].isna().mean()
            if null_rate > 0.70:
                return False, f"{col} null rate {null_rate:.0%} > 70%"

        if "col" in metadata:
            col = metadata["col"]
            if col in dirty_df.columns:
                non_null = dirty_df[col].notna().sum()
                if non_null < 3:
                    return False, f"{col} has only {non_null} non-null values"

        return True, "ok"
