import logging
import random
import re
from collections import deque

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class Corruptor:
    """
    Adversarial corruptor with adaptive curriculum escalation.

    Tier escalation uses a rolling 5-step average reward, not instantaneous
    reward. Gates:
        Tier 1 -> 2: rolling avg > 2.0 for 5 consecutive steps and step >= 40
        Tier 2 -> 3: rolling avg > 3.0 for 5 consecutive steps and step >= 120

    De-escalation: if rolling avg < 1.0 for 3 consecutive steps, drop one tier.

    Thresholds calibrated for Qwen2.5-1.5B on T4 GPU: tier 2 unlocks when
    5-step rolling avg > 2.0 for 5 consecutive steps after step 40. These
    values reflect the actual reward range observed during training
    (mean 3.45, oscillation range 0.33-6.95).
    """

    TIER_STEP_GATES = {2: 40, 3: 120}
    TIER_REWARD_GATES = {2: 2.0, 3: 3.0}
    ESCALATION_WINDOW = 5
    DEESCALATION_THRESHOLD = 1.0
    DEESCALATION_WINDOW = 3

    def __init__(self):
        self._epoch = 0
        self._recent_rewards = deque(maxlen=50)
        self._unlocked_tier = 1
        self._consecutive_above = 0
        self._consecutive_below = 0

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
        self._epoch = {1: 0, 2: self.TIER_STEP_GATES[2], 3: self.TIER_STEP_GATES[3]}[tier]
        self._unlocked_tier = tier
        self._consecutive_above = 0
        self._consecutive_below = 0

    def _rolling_avg(self, window: int | None = None) -> float:
        if not self._recent_rewards:
            return -99.0
        recent_window = window or self.ESCALATION_WINDOW
        recent = list(self._recent_rewards)[-recent_window:]
        return sum(recent) / len(recent)

    def _update_tier(self):
        rolling = self._rolling_avg(self.ESCALATION_WINDOW)

        if rolling < self.DEESCALATION_THRESHOLD and len(self._recent_rewards) >= self.DEESCALATION_WINDOW:
            self._consecutive_below += 1
            if self._consecutive_below >= self.DEESCALATION_WINDOW and self._unlocked_tier > 1:
                old_tier = self._unlocked_tier
                self._unlocked_tier -= 1
                self._consecutive_below = 0
                self._consecutive_above = 0
                logger.info(
                    "Corruptor de-escalated tier %s -> %s: epoch=%s, rolling_avg=%.3f",
                    old_tier,
                    self._unlocked_tier,
                    self._epoch,
                    rolling,
                )
                return
        else:
            self._consecutive_below = 0

        for candidate_tier in [2, 3]:
            if self._unlocked_tier >= candidate_tier:
                continue
            step_ok = self._epoch >= self.TIER_STEP_GATES[candidate_tier]
            reward_ok = rolling >= self.TIER_REWARD_GATES[candidate_tier]

            if step_ok and reward_ok:
                self._consecutive_above += 1
                if self._consecutive_above >= self.ESCALATION_WINDOW:
                    self._unlocked_tier = candidate_tier
                    self._consecutive_above = 0
                    logger.info(
                        "Corruptor escalated to tier %s: epoch=%s, rolling_avg=%.3f",
                        candidate_tier,
                        self._epoch,
                        rolling,
                    )
            else:
                self._consecutive_above = 0
            break

    def get_escalation_status(self) -> dict:
        rolling = self._rolling_avg(self.ESCALATION_WINDOW)
        next_tier = None
        if self._unlocked_tier < 2:
            next_tier = 2
        elif self._unlocked_tier < 3:
            next_tier = 3

        if next_tier is None:
            steps_to_next_tier = 0
        else:
            steps_to_next_tier = max(self.ESCALATION_WINDOW - self._consecutive_above, 0)

        return {
            "current_tier": int(self._unlocked_tier),
            "rolling_avg": round(float(rolling), 4),
            "steps_above_threshold": int(self._consecutive_above),
            "steps_to_next_tier": int(steps_to_next_tier),
        }

    def current_tier(self) -> int:
        return self._unlocked_tier

    def is_transitioning(self) -> bool:
        transition_window = self.ESCALATION_WINDOW
        for gate in self.TIER_STEP_GATES.values():
            if gate <= self._epoch < gate + transition_window:
                return True
        return False

    def generate_episode(self, clean_df: pd.DataFrame, max_retries: int = 10) -> tuple:
        """
        Returns (dirty_df, ground_truth, metadata).
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
        tool = random.choice(["inject_null_single", "inject_type_error", "enum_substitution"])
        col = random.choice(df.columns.tolist())
        row = random.randint(0, len(df) - 1)
        original_val = df.at[row, col]

        if tool == "inject_null_single":
            df.at[row, col] = np.nan
        elif tool == "inject_type_error":
            df[col] = df[col].astype(object)
            df.at[row, col] = f"ERR_{random.randint(10, 99)}"
        elif tool == "enum_substitution":
            enum_options = {
                "currency": ["CAD", "AUD", "JPY", "CHF", "CNY"],
                "status": ["processing", "cancelled", "refunded", "unknown"],
                "category": ["misc", "other", "unknown"],
            }
            candidates = [c for c in df.columns if c in enum_options]
            if not candidates:
                return self._corrupt_tier1_fallback(df)
            col = random.choice(candidates)
            row = random.randint(0, len(df) - 1)
            original_val = df.at[row, col]
            df.at[row, col] = random.choice(enum_options[col])
            return df, {"tool": tool, "col": col, "row": row, "original": original_val}

        return df, {"tool": tool, "col": col, "row": row, "original": original_val}

    def _corrupt_tier1_fallback(self, df: pd.DataFrame) -> tuple:
        col = random.choice(df.columns.tolist())
        row = random.randint(0, len(df) - 1)
        original_val = df.at[row, col]
        df.at[row, col] = np.nan
        return df, {"tool": "inject_null_single", "col": col, "row": row, "original": original_val}

    def _corrupt_tier2(self, df: pd.DataFrame) -> tuple:
        tool = random.choice(
            [
                "inject_null_cluster",
                "swap_date_format",
                "inject_out_of_range_age",
                "semantic_temporal_drift",
                "currency_unit_mismatch",
            ]
        )
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

        elif tool == "inject_out_of_range_age":
            if "age" not in df.columns or "birth_year" not in df.columns:
                return self._corrupt_tier1(df)
            row = random.randint(0, len(df) - 1)
            df.at[row, "age"] = random.randint(130, 180)
            metadata.update({"col": "age", "row": row})

        elif tool == "semantic_temporal_drift":
            if "age" not in df.columns or "birth_year" not in df.columns:
                return self._corrupt_tier1(df)
            row = random.randint(0, len(df) - 1)
            try:
                birth_year = int(float(str(df.at[row, "birth_year"])))
                correct_age = 2024 - birth_year
                drift = random.randint(18, 30)
                corrupted_age = min(correct_age + drift, 119)
                df.at[row, "age"] = corrupted_age
                metadata.update({"col": "age", "row": row, "original_age": correct_age, "drift": drift})
            except Exception:
                return self._corrupt_tier1(df)

        elif tool == "currency_unit_mismatch":
            if "currency" not in df.columns or "amount" not in df.columns:
                return self._corrupt_tier1(df)
            row = random.randint(0, len(df) - 1)
            try:
                currency = str(df.at[row, "currency"])
                amount = float(df.at[row, "amount"])
                if currency == "INR" and amount < 200:
                    df.at[row, "amount"] = round(amount * 83.0, 2)
                    metadata.update({"col": "amount", "row": row, "original": amount})
                elif currency == "EUR" and amount > 10000:
                    df.at[row, "amount"] = round(amount / 1.08, 2)
                    metadata.update({"col": "amount", "row": row, "original": amount})
                else:
                    return self._corrupt_tier1(df)
            except Exception:
                return self._corrupt_tier1(df)

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
        del ground_truth
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

        if tool == "semantic_temporal_drift":
            if "age" not in dirty_df.columns or "birth_year" not in dirty_df.columns:
                return False, "semantic drift requires age+birth_year columns"
            return True, "ok"

        if tool == "currency_unit_mismatch":
            if "currency" not in dirty_df.columns or "amount" not in dirty_df.columns:
                return False, "currency_unit_mismatch requires currency+amount columns"
            return True, "ok"

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
