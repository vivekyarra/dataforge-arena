import pandas as pd
import numpy as np
import random
import re
from collections import deque
from environment.schemas import CORRUPTOR_TOOLS, DEPT_MAP

class Corruptor:
    TIER_EPOCH_GATES  = {2: 30, 3: 70}
    TIER_REWARD_GATES = {2: 0.5, 3: 0.9}

    def __init__(self):
        self._epoch = 0
        self._recent_rewards = deque(maxlen=20)
        self._unlocked_tier = 1   # highest tier currently unlocked

    @property
    def difficulty(self) -> int:
        return self._unlocked_tier

    def record_episode(self, surgeon_reward: float):
        self._recent_rewards.append(surgeon_reward)
        self._epoch += 1
        self._update_tier()

    def _rolling_avg(self) -> float:
        if not self._recent_rewards:
            return -99.0
        return sum(self._recent_rewards) / len(self._recent_rewards)

    def _update_tier(self):
        for candidate_tier in [2, 3]:
            if self._unlocked_tier >= candidate_tier:
                continue
            epoch_ok  = self._epoch >= self.TIER_EPOCH_GATES[candidate_tier]
            reward_ok = self._rolling_avg() >= self.TIER_REWARD_GATES[candidate_tier]
            if epoch_ok and reward_ok:
                self._unlocked_tier = candidate_tier
                print(f"[CORRUPTOR] Tier {candidate_tier} UNLOCKED — "
                      f"epoch={self._epoch}, rolling_avg={self._rolling_avg():.3f}")

    def current_tier(self) -> int:
        return self._unlocked_tier

    def is_transitioning(self) -> bool:
        for gate in self.TIER_EPOCH_GATES.values():
            if gate <= self._epoch < gate + 10:
                return True
        return False

    def generate_episode(self, clean_df: pd.DataFrame,
                          max_retries: int = 10) -> tuple:
        """
        Returns (dirty_df, ground_truth, metadata)
        Solvability gate ensures every episode is learnable.
        """
        tier = self.current_tier()
        
        for attempt in range(max_retries):
            dirty, metadata = self._corrupt(clean_df.copy(), tier)
            valid, reason = self._solvability_gate(dirty, clean_df, metadata)
            if valid:
                return dirty, clean_df.copy(), metadata
            # else: retry with a new random episode

        # Fallback: guaranteed-valid tier 1 episode
        print(f"[CORRUPTOR] Retry limit -- falling back to tier 1")
        dirty, metadata = self._corrupt_tier1(clean_df.copy())
        return dirty, clean_df.copy(), metadata

    def _corrupt(self, df: pd.DataFrame, tier: int) -> tuple:
        if tier == 1:
            return self._corrupt_tier1(df)
        elif tier == 2:
            return self._corrupt_tier2(df)
        else:
            return self._corrupt_tier3(df)

    # -- Tier 1 tools ----------------------------------------------
    def _corrupt_tier1(self, df: pd.DataFrame) -> tuple:
        tool = random.choice(["inject_null_single", "inject_type_error"])
        col = random.choice(df.columns.tolist())
        row = random.randint(0, len(df) - 1)
        original_val = df.at[row, col]

        if tool == "inject_null_single":
            df.at[row, col] = np.nan
        elif tool == "inject_type_error":
            df[col] = df[col].astype(object)
            df.at[row, col] = f"ERR_{random.randint(10, 99)}"

        return df, {"tool": tool, "col": col, "row": row,
                    "original": original_val}

    # -- Tier 2 tools ----------------------------------------------
    def _corrupt_tier2(self, df: pd.DataFrame) -> tuple:
        tool = random.choice(["inject_null_cluster", "swap_date_format",
                               "inject_out_of_range_age"])
        metadata = {"tool": tool}

        if tool == "inject_null_cluster":
            col = random.choice(df.columns.tolist())
            start = random.randint(0, max(0, len(df) - 5))
            rows = list(range(start, min(start + random.randint(3, 5), len(df))))
            for r in rows:
                df.at[r, col] = np.nan
            metadata.update({"col": col, "rows": rows})

        elif tool == "swap_date_format":
            date_cols = [c for c in df.columns if "date" in c.lower()]
            if date_cols:
                col = random.choice(date_cols)
                row = random.randint(0, len(df) - 1)
                val = str(df.at[row, col])
                if re.match(r"\d{4}-\d{2}-\d{2}", val):
                    parts = val.split("-")
                    df.at[row, col] = f"{parts[1]}/{parts[2]}/{parts[0][2:]}"
                metadata.update({"col": col, "row": row})
            else:
                return self._corrupt_tier1(df)

        elif tool == "inject_out_of_range_age":
            if "age" in df.columns and "birth_year" in df.columns:
                row = random.randint(0, len(df) - 1)
                df.at[row, "age"] = random.randint(80, 120)
                metadata.update({"col": "age", "row": row})
            else:
                return self._corrupt_tier1(df)

        return df, metadata

    # -- Tier 3 tools ----------------------------------------------
    def _corrupt_tier3(self, df: pd.DataFrame) -> tuple:
        tool = random.choice(["break_foreign_key", "duplicate_row_mutate"])
        metadata = {"tool": tool}

        if tool == "break_foreign_key":
            if "department_id" in df.columns:
                row = random.randint(0, len(df) - 1)
                df.at[row, "department_id"] = random.randint(500, 9999)
                if "department_name" in df.columns:
                    df.at[row, "department_name"] = "INVALID_DEPT"
                metadata.update({"col": "department_id", "row": row})
            else:
                return self._corrupt_tier2(df)

        elif tool == "duplicate_row_mutate":
            row = random.randint(0, len(df) - 1)
            dup = df.iloc[row].copy()
            col = random.choice(df.columns.tolist())
            dup[col] = np.nan
            df = pd.concat([df, pd.DataFrame([dup])],
                           ignore_index=True)
            metadata.update({"col": col, "row": row})

        return df, metadata

    # -- Solvability gate ------------------------------------------
    def _solvability_gate(self, dirty_df: pd.DataFrame,
                           ground_truth: pd.DataFrame,
                           metadata: dict) -> tuple:
        tool = metadata.get("tool", "")
        
        # Hard ban: whole row deleted
        if tool == "delete_row":
            return False, "row deletion -- unrecoverable"

        # Column null rate must stay under 70%
        for col in dirty_df.columns:
            null_rate = dirty_df[col].isna().mean()
            if null_rate > 0.70:
                return False, f"{col} null rate {null_rate:.0%} > 70%"

        # At least 3 non-null reference values per affected column
        if "col" in metadata:
            col = metadata["col"]
            if col in dirty_df.columns:
                non_null = dirty_df[col].notna().sum()
                if non_null < 3:
                    return False, f"{col} has only {non_null} non-null values"

        return True, "ok"
    
    def compute_corruptor_reward(self, metadata: dict) -> float:
        tool = metadata.get("tool", "")
        if tool not in CORRUPTOR_TOOLS or CORRUPTOR_TOOLS[tool].get("banned"):
            return -10.0
        return CORRUPTOR_TOOLS[tool]["reward"]
