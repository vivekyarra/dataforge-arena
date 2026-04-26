import logging
import time

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class RewardComputer:
    """
    Constraint-aware reward computer for DataForge Arena.

    Reward scaling (balanced signals):
        accuracy_delta * 50.0
        constraint_alignment    max +3.0
        schema_alignment        max +2.0
        outlier_targeting       max +0.5
        reasoning_quality       max +1.5
        parse_bonus             max +0.5
        anti_hack               min -5.0

    Total reward range: approximately [-5.0, +8.0]
    """

    def compute(
        self,
        state,
        ground_truth,
        action,
        original_dirty,
        prev_accuracy,
        episode_start,
        step_count,
        starting_accuracy=None,
        previous_state=None,
        schema=None,
    ) -> dict:
        del step_count
        if time.time() - episode_start > 30:
            return {"total": -3.0, "timeout": True}

        rewards = {}
        current_acc = self._field_accuracy(state, ground_truth)
        delta = current_acc - prev_accuracy
        rewards["accuracy_delta"] = delta * 50.0
        rewards["_current_accuracy"] = current_acc

        rewards["constraint_alignment"] = self._score_constraint_alignment(
            action, state, ground_truth, schema
        )
        rewards["schema_alignment"] = self._score_schema_alignment(action, state, schema)
        rewards["outlier_targeting"] = self._score_outlier_targeting(
            action, state, ground_truth, previous_state
        )
        rewards["reasoning_quality"] = self._score_causal_reasoning(action, state, schema)
        rewards["parse_bonus"] = self._score_parse_bonus(action)
        rewards["anti_hack"] = self._detect_shortcuts(state, original_dirty)

        # Anti-collapse: penalize CORRECT_FORMAT on null cells so the policy
        # gets a direct negative signal instead of drifting into tool 3.
        try:
            display_cols = [c for c in state.columns if c != "_is_deleted"]
            if action.column < len(display_cols):
                col_name = display_cols[action.column]
                cell_val = state.iloc[action.row_id][col_name]
                is_null = pd.isna(cell_val) or str(cell_val).strip() in ("", "None", "nan", "NaN")
                tool_name_map = {
                    0: "IMPUTE_MEDIAN",
                    1: "IMPUTE_MODE",
                    2: "IMPUTE_FORWARD_FILL",
                    3: "CORRECT_FORMAT",
                    4: "DELETE_ROW",
                    5: "MERGE_DUPLICATE",
                    6: "FLAG_UNCERTAIN",
                    7: "NO_OP",
                }
                if is_null and tool_name_map.get(action.tool_id) == "CORRECT_FORMAT":
                    rewards["anti_hack"] = min(rewards.get("anti_hack", 0.0), -1.0)
        except Exception:
            pass

        rewards["total"] = (
            rewards["accuracy_delta"]
            + rewards["constraint_alignment"]
            + rewards["schema_alignment"]
            + rewards["outlier_targeting"]
            + rewards["reasoning_quality"]
            + rewards["parse_bonus"]
            + rewards["anti_hack"]
        )

        start_acc = starting_accuracy if starting_accuracy is not None else prev_accuracy
        rewards["episode_complete"] = (current_acc >= 0.999) or ((current_acc - start_acc) > 0.05)
        return rewards

    def _score_parse_bonus(self, action) -> float:
        reasoning = action.reasoning
        if reasoning is None:
            return 0.0
        if len(reasoning) <= 10:
            return 0.0
        if reasoning.startswith("LLM error"):
            return 0.0
        if action.tool_id not in range(8):
            return 0.0
        if action.column < 0:
            return 0.0
        if action.row_id < 0:
            return 0.0
        return 0.5

    def _score_constraint_alignment(self, action, state, ground_truth, schema) -> float:
        del ground_truth
        if schema is None:
            return 0.0
        try:
            display_cols = [c for c in state.columns if c != "_is_deleted"]
            if action.column >= len(display_cols):
                return -0.5
            col_name = display_cols[action.column]
            cell_val = state.iloc[action.row_id][col_name]
            col_schema = schema.get(col_name, {})
            violation = self._detect_constraint_violation(
                cell_val,
                col_name,
                col_schema,
                state,
                action.row_id,
            )

            tool_name_map = {
                0: "IMPUTE_MEDIAN",
                1: "IMPUTE_MODE",
                2: "IMPUTE_FORWARD_FILL",
                3: "CORRECT_FORMAT",
                4: "DELETE_ROW",
                5: "MERGE_DUPLICATE",
                6: "FLAG_UNCERTAIN",
                7: "NO_OP",
            }
            tool = tool_name_map.get(action.tool_id, "NO_OP")

            if violation == "null_numeric" and tool == "IMPUTE_MEDIAN":
                return 3.0
            if violation == "null_categorical" and tool in ("IMPUTE_MODE", "IMPUTE_FORWARD_FILL"):
                return 3.0
            if violation == "null_numeric" and tool == "CORRECT_FORMAT":
                return -1.5
            if violation == "null_categorical" and tool == "CORRECT_FORMAT":
                return -1.5

            if violation is None:
                if tool == "NO_OP":
                    return 0.6
                return -1.0

            if violation == "range" and tool == "CORRECT_FORMAT":
                return 3.0
            if violation == "type_error" and tool == "CORRECT_FORMAT":
                return 3.0
            if violation == "enum_violation" and tool == "CORRECT_FORMAT":
                return 2.5
            if violation == "fk_mismatch" and tool == "CORRECT_FORMAT":
                return 3.0
            if violation is not None and tool == "NO_OP":
                return -2.0
            if violation is not None and tool != "NO_OP":
                return 0.5
            return 0.0
        except Exception as e:
            logger.warning("reward error in constraint_alignment: %s | action=%s", e, action)
            return 0.0

    def _detect_constraint_violation(self, cell_val, col_name, col_schema, state, row_id):
        col_type = col_schema.get("type", "str")

        is_null = pd.isna(cell_val) or str(cell_val).strip() in ("", "None", "nan", "NaN")
        if is_null:
            if col_type in ("int", "float"):
                return "null_numeric"
            return "null_categorical"

        cell_str = str(cell_val)

        if cell_str.startswith("ERR_"):
            return "type_error"
        if col_type in ("int", "float"):
            try:
                numeric_val = float(cell_str)
            except (ValueError, TypeError):
                return "type_error"
            range_constraint = col_schema.get("range")
            if range_constraint and not (range_constraint[0] <= numeric_val <= range_constraint[1]):
                return "range"

        allowed_values = col_schema.get("values")
        if allowed_values and cell_str not in allowed_values:
            return "enum_violation"

        if col_name == "department_id" and "department_name" in state.columns:
            dept_map = {
                1: "Cardiology",
                2: "Neurology",
                3: "Oncology",
                4: "Pediatrics",
                5: "Orthopedics",
                6: "Radiology",
                7: "Emergency",
                8: "Surgery",
                9: "Psychiatry",
                10: "General",
            }
            try:
                dept_id = int(float(cell_str))
                expected_name = dept_map.get(dept_id)
                actual_name = str(state.at[state.index[row_id], "department_name"])
                if expected_name and expected_name != actual_name:
                    return "fk_mismatch"
            except Exception:
                pass

        return None

    def _score_schema_alignment(self, action, state, schema) -> float:
        if schema is None:
            return 0.0
        try:
            display_cols = [c for c in state.columns if c != "_is_deleted"]
            if action.column >= len(display_cols):
                return -0.5
            col_name = display_cols[action.column]
            col_type = schema.get(col_name, {}).get("type", "str")

            type_to_correct_impute = {
                "int": 0,
                "float": 0,
                "str": 1,
                "email": 1,
                "phone": 1,
                "date": 2,
            }
            correct_impute = type_to_correct_impute.get(col_type, 1)
            impute_tools = {0, 1, 2}

            if action.tool_id in impute_tools:
                if action.tool_id == correct_impute:
                    return 2.0
                return -0.5
            return 0.0
        except Exception as e:
            logger.warning("reward error in schema_alignment: %s | action=%s", e, action)
            return 0.0

    def _score_outlier_targeting(self, action, state, ground_truth, previous_state) -> float:
        del ground_truth
        target_state = previous_state if previous_state is not None else state
        try:
            display_cols = [c for c in target_state.columns if c != "_is_deleted"]
            if action.column >= len(display_cols):
                return 0.0
            col_name = display_cols[action.column]
            col_data = pd.to_numeric(target_state[col_name], errors="coerce").dropna()
            if len(col_data) < 4:
                return 0.0
            mean = col_data.mean()
            std = col_data.std()
            if std < 1e-9:
                return 0.0
            cell_val = pd.to_numeric(target_state.iloc[action.row_id][col_name], errors="coerce")
            if pd.isna(cell_val):
                return 0.0
            z_score = abs(cell_val - mean) / std
            if z_score > 3.0:
                return 0.5
            if z_score > 2.0:
                return 0.2
            if z_score < 0.5 and action.tool_id not in (6, 7):
                return -0.3
            return 0.0
        except Exception as e:
            logger.warning("reward error in outlier_targeting: %s | action=%s", e, action)
            return 0.0

    def _score_causal_reasoning(self, action, state, schema) -> float:
        del schema
        reasoning = (action.reasoning or "").strip().lower()

        if len(reasoning) < 5:
            return -0.3
        if len(reasoning) > 200:
            return -0.1

        score = 0.0

        try:
            display_cols = [c for c in state.columns if c != "_is_deleted"]
            if action.column < len(display_cols):
                col_name = display_cols[action.column].lower()
                if col_name in reasoning or col_name.replace("_", " ") in reasoning:
                    score += 0.4
        except Exception:
            pass

        violation_keywords = {
            "null": 0.25,
            "missing": 0.25,
            "none": 0.10,
            "nan": 0.10,
            "range": 0.25,
            "exceed": 0.25,
            "above": 0.15,
            "below": 0.15,
            "max": 0.15,
            "min": 0.15,
            "constraint": 0.20,
            "type": 0.15,
            "err_": 0.20,
            "format": 0.15,
            "invalid": 0.20,
            "outlier": 0.25,
            "anomaly": 0.20,
            "inconsistent": 0.25,
            "foreign": 0.25,
            "fk": 0.25,
            "mismatch": 0.25,
            "department": 0.15,
            "duplicate": 0.20,
            "birth": 0.15,
            "implies": 0.25,
            "therefore": 0.20,
            "schema": 0.20,
            "z-score": 0.30,
            "zscore": 0.30,
            "sigma": 0.20,
        }
        keyword_score = 0.0
        keyword_hits = 0
        for keyword, bonus in violation_keywords.items():
            if keyword in reasoning and keyword_hits < 3:
                keyword_score += bonus
                keyword_hits += 1
        score += min(keyword_score, 0.60)

        causal_connectors = ["because", "since", "implies", "therefore", "so", "hence", "means"]
        if any(connector in reasoning for connector in causal_connectors):
            score += 0.3

        if score >= 0.9:
            score += 0.20

        return min(score, 1.5)

    def _field_accuracy(self, state: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
        state_vals = state.drop(columns=["_is_deleted"], errors="ignore")
        gt_vals = ground_truth.copy()

        min_rows = min(len(state_vals), len(gt_vals))
        state_vals = state_vals.iloc[:min_rows]
        gt_vals = gt_vals.iloc[:min_rows]

        if state_vals.shape[1] != gt_vals.shape[1]:
            return 0.0

        extra_rows = len(state) - len(ground_truth)
        extra_penalty = 0.0
        if extra_rows > 0:
            total_cells_gt = len(ground_truth) * len(ground_truth.columns)
            extra_penalty = (extra_rows * len(ground_truth.columns)) / max(total_cells_gt, 1)

        try:
            matches = state_vals.values == gt_vals.values
            null_mask = pd.isna(state_vals.values)
            gt_null_mask = pd.isna(gt_vals.values)
            null_matches = null_mask & gt_null_mask
            total_matches = matches | null_matches
            base_acc = float(total_matches.sum()) / total_matches.size
            return max(0.0, base_acc - extra_penalty)
        except Exception:
            return 0.0

    @staticmethod
    def _values_match(cell_val, gt_val) -> bool:
        if pd.isna(cell_val) and pd.isna(gt_val):
            return True
        if pd.isna(cell_val) or pd.isna(gt_val):
            return False
        try:
            return bool(cell_val == gt_val)
        except Exception:
            return str(cell_val) == str(gt_val)

    def _detect_shortcuts(self, state, original_dirty) -> float:
        del original_dirty
        if "_is_deleted" in state.columns:
            deleted_rate = state["_is_deleted"].mean()
            if deleted_rate > 0.25:
                return -5.0
        return 0.0
