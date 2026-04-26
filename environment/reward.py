import time
import pandas as pd
import numpy as np


class RewardComputer:
    """
    Constraint-aware reward computer for DataForge Arena.

    Reward scaling (v2 — balanced signals):
        accuracy_delta × 50     — bounded, no longer dominates
        constraint_alignment    — max +3.0 (was +2.0)
        schema_alignment        — max +2.0 (was +1.0)
        outlier_targeting       — max +0.5
        reasoning_quality       — max +1.5 (was +0.8)
        parse_bonus             — max +0.5
        anti_hack               — min −5.0

    Total reward range: approximately [−5.0, +8.0]
    """

    def compute(self, state, ground_truth, action, original_dirty, prev_accuracy,
                episode_start, step_count, starting_accuracy=None, previous_state=None,
                schema=None) -> dict:

        if time.time() - episode_start > 30:
            return {"total": -3.0, "timeout": True}

        rewards = {}
        current_acc = self._field_accuracy(state, ground_truth)
        delta = current_acc - prev_accuracy
        # Reduced multiplier from 250 to 50 so shaped signals are competitive
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

        # parse_bonus: award when the model produced a clean, valid JSON action
        # (no longer depends on "EXACT_PARSE:" prefix which was never reliably set)
        rewards["parse_bonus"] = self._score_parse_bonus(action)

        rewards["anti_hack"] = self._detect_shortcuts(state, original_dirty)

        rewards["total"] = (
            rewards["accuracy_delta"] +
            rewards["constraint_alignment"] +
            rewards["schema_alignment"] +
            rewards["outlier_targeting"] +
            rewards["reasoning_quality"] +
            rewards["parse_bonus"] +
            rewards["anti_hack"]
        )

        start_acc = starting_accuracy if starting_accuracy is not None else prev_accuracy
        rewards["episode_complete"] = (current_acc >= 0.999) or ((current_acc - start_acc) > 0.05)
        return rewards

    def _score_parse_bonus(self, action) -> float:
        """
        Award +0.5 when the model produced a clean, structurally valid action.

        Criteria (all must be true):
            - action.reasoning is not None
            - len(action.reasoning) > 10
            - reasoning does not start with "LLM error"
            - tool_id is in valid range [0, 7]
            - column >= 0
            - row_id >= 0
        """
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
        if schema is None:
            return 0.0
        try:
            display_cols = [c for c in state.columns if c != "_is_deleted"]
            if action.column >= len(display_cols):
                return -0.5
            col_name = display_cols[action.column]
            cell_val = state.iloc[action.row_id][col_name]
            col_schema = schema.get(col_name, {})
            violation = self._detect_constraint_violation(cell_val, col_name, col_schema, state, action.row_id)

            tool_name_map = {
                0: "IMPUTE_MEDIAN", 1: "IMPUTE_MODE", 2: "IMPUTE_FORWARD_FILL",
                3: "CORRECT_FORMAT", 4: "DELETE_ROW", 5: "MERGE_DUPLICATE",
                6: "FLAG_UNCERTAIN", 7: "NO_OP",
            }
            tool = tool_name_map.get(action.tool_id, "NO_OP")

            if violation is None:
                if tool == "NO_OP":
                    return +0.6
                return -1.0

            # Correct tool-violation pairings (max +3.0)
            if violation == "null_numeric" and tool == "IMPUTE_MEDIAN":
                return +3.0
            if violation == "null_categorical" and tool in ("IMPUTE_MODE", "IMPUTE_FORWARD_FILL"):
                return +3.0

            # CRITICAL FIX: CORRECT_FORMAT on null_numeric is WRONG — should
            # return 0.0, not +3.0. This was causing tool-3 collapse because
            # the model earned high reward using CORRECT_FORMAT on null cells.
            if violation == "null_numeric" and tool == "CORRECT_FORMAT":
                return 0.0
            if violation == "null_categorical" and tool == "CORRECT_FORMAT":
                return 0.0

            if violation == "range" and tool == "CORRECT_FORMAT":
                return +3.0
            if violation == "type_error" and tool == "CORRECT_FORMAT":
                return +3.0
            if violation == "enum_violation" and tool == "CORRECT_FORMAT":
                return +2.5
            if violation == "fk_mismatch" and tool == "CORRECT_FORMAT":
                return +3.0
            if violation is not None and tool == "NO_OP":
                return -2.0
            if violation is not None and tool != "NO_OP":
                return +0.5
            return 0.0
        except Exception:
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
            DEPT_MAP = {
                1: "Cardiology", 2: "Neurology", 3: "Oncology", 4: "Pediatrics",
                5: "Orthopedics", 6: "Radiology", 7: "Emergency", 8: "Surgery",
                9: "Psychiatry", 10: "General",
            }
            try:
                dept_id = int(float(cell_str))
                expected_name = DEPT_MAP.get(dept_id)
                actual_name = str(state.at[state.index[row_id], "department_name"])
                if expected_name and expected_name != actual_name:
                    return "fk_mismatch"
            except Exception:
                pass

        return None

    def _score_schema_alignment(self, action, state, schema) -> float:
        """
        Schema alignment: max +2.0 (was +1.0).
        """
        if schema is None:
            return 0.0
        try:
            display_cols = [c for c in state.columns if c != "_is_deleted"]
            if action.column >= len(display_cols):
                return -0.5
            col_name = display_cols[action.column]
            col_type = schema.get(col_name, {}).get("type", "str")

            type_to_correct_impute = {
                "int": 0, "float": 0,
                "str": 1, "email": 1, "phone": 1, "date": 2,
            }
            correct_impute = type_to_correct_impute.get(col_type, 1)
            impute_tools = {0, 1, 2}

            if action.tool_id in impute_tools:
                if action.tool_id == correct_impute:
                    return +2.0
                return -0.5
            return 0.0
        except Exception:
            return 0.0

    def _score_outlier_targeting(self, action, state, ground_truth, previous_state) -> float:
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
                return +0.5
            if z_score > 2.0:
                return +0.2
            if z_score < 0.5 and action.tool_id not in (6, 7):
                return -0.3
            return 0.0
        except Exception:
            return 0.0

    def _score_causal_reasoning(self, action, state, schema) -> float:
        """
        Reasoning quality: max +1.5 (was +0.8).
        """
        reasoning = action.reasoning
        reasoning = reasoning.strip().lower()

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
            "null": 0.3, "missing": 0.3, "none": 0.1, "nan": 0.1,
            "range": 0.3, "exceed": 0.3, "above": 0.2, "below": 0.2,
            "max": 0.2, "min": 0.2, "constraint": 0.2,
            "type": 0.2, "err_": 0.2, "format": 0.2, "invalid": 0.2,
            "outlier": 0.3, "anomaly": 0.2, "inconsistent": 0.3,
            "foreign": 0.3, "fk": 0.3, "mismatch": 0.3, "department": 0.2,
            "duplicate": 0.2, "birth": 0.2, "implies": 0.3, "therefore": 0.2,
        }
        for kw, bonus in violation_keywords.items():
            if kw in reasoning:
                score += bonus
                break

        causal_connectors = ["because", "since", "implies", "therefore", "so", "hence", "means"]
        if any(c in reasoning for c in causal_connectors):
            score += 0.3

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
            matches = (state_vals.values == gt_vals.values)
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
        if "_is_deleted" in state.columns:
            deleted_rate = state["_is_deleted"].mean()
            if deleted_rate > 0.25:
                return -5.0
        return 0.0
