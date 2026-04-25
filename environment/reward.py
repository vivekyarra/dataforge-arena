import pandas as pd
import numpy as np
import time
import re
from environment.schemas import SURGEON_TOOLS

class RewardComputer:
    
    def compute(self, state: pd.DataFrame, ground_truth: pd.DataFrame,
                action, original_dirty: pd.DataFrame,
                prev_accuracy: float, episode_start: float,
                step_count: int, starting_accuracy: float = None) -> dict:
        
        rewards = {}
        
        # Timeout hard stop
        elapsed = time.time() - episode_start
        if elapsed > 30:
            return {"total": -3.0, "timeout": True}
        
        # R1: ACCURACY DELTA -- primary learning signal
        # Delta x 20 means: fixing one cell in a 100-cell dataset
        # = +0.01 delta x 20 = +0.2 reward per correct fix
        current_acc = self._field_accuracy(state, ground_truth)
        delta = current_acc - prev_accuracy
        rewards["accuracy_delta"] = delta * 20.0
        rewards["_current_accuracy"] = current_acc  # stored for next step

        # R2: TOOL LOGIC -- heuristic process supervision, no LLM
        rewards["tool_logic"] = self._score_tool_logic(action, state, ground_truth)

        # R3: REASONING QUALITY -- keyword heuristic, no LLM
        rewards["reasoning"] = self._score_reasoning(action, state)

        # R4: EFFICIENCY -- penalize WRONG actions, not all actions
        rewards["efficiency"] = self._score_efficiency(action, state, ground_truth)

        # R5: ANTI-HACK -- soft-delete rate check
        rewards["anti_hack"] = self._detect_shortcuts(state, original_dirty)

        total = (
            rewards["accuracy_delta"] +
            rewards["tool_logic"] +
            rewards["reasoning"] +
            rewards["efficiency"] +
            rewards["anti_hack"]
        )
        rewards["total"] = total
        
        # NOTE: improvement > 0.05 rarely triggers for single cell fixes.
        # It functionally acts as an early termination gate for massive structural
        # fixes (like duplicate_row_mutate) that instantly restore >5% accuracy.
        start_acc = starting_accuracy if starting_accuracy is not None else prev_accuracy
        improvement = current_acc - start_acc
        rewards["episode_complete"] = (current_acc >= 0.999) or (improvement > 0.05)
        
        return rewards

    def _field_accuracy(self, state: pd.DataFrame,
                         ground_truth: pd.DataFrame) -> float:
        """
        Soft-delete aware accuracy.
        '_is_deleted' rows are counted as wrong on all fields.
        Handles shape mismatch from duplicate_row_mutate by trimming.
        """
        # Remove internal bookkeeping columns
        state_vals = state.drop(columns=["_is_deleted"], errors="ignore")
        gt_vals = ground_truth.copy()
        
        # BUG 7 FIX: Handle shape mismatch from duplicate_row_mutate
        # Trim both to common length so extra rows don't poison accuracy to 51%
        min_rows = min(len(state_vals), len(gt_vals))
        state_vals = state_vals.iloc[:min_rows]
        gt_vals = gt_vals.iloc[:min_rows]
        
        if state_vals.shape[1] != gt_vals.shape[1]:
            # Column count mismatch -- something very wrong
            return 0.0
        
        # Penalize extra rows in state (duplicates the agent hasn't merged/deleted)
        extra_rows = len(state) - len(ground_truth)
        extra_penalty = 0.0
        if extra_rows > 0:
            total_cells_gt = len(ground_truth) * len(ground_truth.columns)
            extra_penalty = (extra_rows * len(ground_truth.columns)) / max(total_cells_gt, 1)
        
        # Element-wise comparison (handles NaN: NaN != anything = wrong)
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

    def _score_tool_logic(self, action, state, ground_truth) -> float:
        try:
            row_data = state.iloc[action.row_id].drop(["_is_deleted"], errors="ignore")
            gt_row = ground_truth.iloc[action.row_id] if action.row_id < len(ground_truth) else None
        except IndexError:
            return -1.0
        
        if gt_row is None:
            # Row exists in state but not in ground_truth (duplicate)
            tool_name = SURGEON_TOOLS[action.tool_id]["name"]
            if tool_name in ("DELETE_ROW", "MERGE_DUPLICATE"):
                return +1.5  # correct to delete/merge a duplicate row
            return -0.5
        
        cell_val = row_data.iloc[action.column]
        gt_val = gt_row.iloc[action.column]
        
        is_null = pd.isna(cell_val)
        is_correct = (not is_null) and (cell_val == gt_val)
        
        # NaN-safe comparison: treat NaN != anything as an error
        try:
            mismatches = sum(1 for a, b in zip(row_data.values, gt_row.values)
                            if pd.isna(a) or pd.isna(b) or str(a) != str(b))
            error_rate_row = mismatches / max(len(row_data), 1)
        except Exception:
            error_rate_row = 0.0
        
        null_rate_col = state.iloc[:, action.column].isna().mean()
        
        tool_name = SURGEON_TOOLS[action.tool_id]["name"]
        
        # Scoring matrix
        if tool_name in ("IMPUTE_MEDIAN", "IMPUTE_MODE", "IMPUTE_FORWARD_FILL"):
            if is_null:       return +1.0
            if is_correct:    return -1.0  # don't impute correct cells
            return -0.5
        
        if tool_name == "CORRECT_FORMAT":
            if is_null:       return -0.5  # can't format a null
            if not is_correct: return +1.0
            return -0.5
        
        if tool_name == "DELETE_ROW":
            if error_rate_row > 0.60: return +1.5   # heavily corrupted row
            if error_rate_row < 0.30: return -2.0   # mostly good row
            return 0.0
        
        if tool_name == "NO_OP":
            if is_correct:    return +0.5   # correct to skip a good cell
            return -0.5                      # wrong to skip a bad cell
        
        if tool_name == "FLAG_UNCERTAIN":
            if null_rate_col > 0.50: return +0.3
            return 0.0
        
        return 0.0

    def _score_reasoning(self, action, state) -> float:
        """Fast keyword heuristic -- NOT LLM-as-judge."""
        reasoning = action.reasoning.strip().lower()
        
        # Anti-hallucination gate: empty/trivial reasoning gets small penalty
        if len(reasoning) < 10:
            return -0.1
            
        bonus = 0.0
        
        try:
            cell_val = state.iloc[action.row_id, action.column]
            is_null = pd.isna(cell_val)
        except IndexError:
            return 0.0
        
        if is_null and any(kw in reasoning for kw in
                           ["null", "missing", "empty", "none", "nan"]):
            bonus += 0.3
        
        if not is_null and any(kw in reasoning for kw in
                               ["format", "type", "invalid", "incorrect",
                                "wrong", "error"]):
            bonus += 0.3
        
        if any(kw in reasoning for kw in
               ["because", "therefore", "since", "indicates", "suggests"]):
            bonus += 0.1
        
        return bonus

    def _score_efficiency(self, action, state, ground_truth) -> float:
        """
        Penalize wrong actions, NOT all actions.
        Key insight: agent must be incentivized to TRY things.
        Wrong action = -1. Right action direction = 0 (outcome handles it).
        """
        try:
            if action.row_id >= len(ground_truth):
                return 0.0  # extra row from duplicate -- no penalty for acting on it
            cell_val = state.iloc[action.row_id, action.column]
            gt_val = ground_truth.iloc[action.row_id, action.column]
            is_correct = (cell_val == gt_val) and pd.notna(cell_val)
        except IndexError:
            return -0.5
        
        tool_name = SURGEON_TOOLS[action.tool_id]["name"]
        
        # Penalize modifying a correct cell
        if is_correct and tool_name not in ("NO_OP", "FLAG_UNCERTAIN"):
            return -1.0
        
        return 0.0

    def _detect_shortcuts(self, state, original_dirty) -> float:
        """
        Anti-hack: soft-delete rate.
        If agent has soft-deleted >25% of rows, it's gaming completeness.
        """
        if "_is_deleted" in state.columns:
            deleted_rate = state["_is_deleted"].mean()
            if deleted_rate > 0.25:
                return -5.0
        return 0.0
