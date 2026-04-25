import re

import numpy as np
import pandas as pd

from environment.schemas import SURGEON_TOOLS
from environment.validation import (
    expected_department_id,
    expected_department_name,
    infer_reference_year,
)


def apply_tool(state: pd.DataFrame, action, schema: dict) -> pd.DataFrame:
    """
    Apply SURGEON tool to state DataFrame.
    Returns modified DataFrame.
    Uses soft-delete -- never physically removes rows.
    """
    tool_name = SURGEON_TOOLS[action.tool_id]["name"]
    row = action.row_id
    col = action.column

    display_cols = [c for c in state.columns if c != "_is_deleted"]
    if row < 0 or col < 0 or row >= len(state) or col >= len(display_cols):
        return state

    if "_is_deleted" in state.columns and bool(state.at[state.index[row], "_is_deleted"]):
        return state

    col_name = display_cols[col]

    if state[col_name].dtype != object:
        state[col_name] = state[col_name].astype(object)

    if tool_name == "IMPUTE_MEDIAN":
        col_data = state[col_name]
        numeric = pd.to_numeric(col_data, errors="coerce")
        median = numeric.median()
        if pd.notna(median):
            state.at[state.index[row], col_name] = median

    elif tool_name == "IMPUTE_MODE":
        col_data = state[col_name].dropna()
        if len(col_data) > 0:
            counts = col_data.value_counts()
            if len(counts) > 0:
                state.at[state.index[row], col_name] = counts.index[0]

    elif tool_name == "IMPUTE_FORWARD_FILL":
        prev_val = _previous_active_value(state, row, col_name)
        if pd.notna(prev_val):
            state.at[state.index[row], col_name] = prev_val

    elif tool_name == "CORRECT_FORMAT":
        state.at[state.index[row], col_name] = _correct_format(state, row, col_name, schema)

    elif tool_name == "DELETE_ROW":
        if "_is_deleted" not in state.columns:
            state["_is_deleted"] = False
        state.at[state.index[row], "_is_deleted"] = True

    elif tool_name == "MERGE_DUPLICATE":
        _merge_duplicate(state, row)

    elif tool_name in ("FLAG_UNCERTAIN", "NO_OP"):
        pass

    return state


def _previous_active_value(state: pd.DataFrame, row: int, col_name: str):
    for prev_row in range(row - 1, -1, -1):
        if "_is_deleted" in state.columns and bool(state.at[state.index[prev_row], "_is_deleted"]):
            continue
        prev_val = state.at[state.index[prev_row], col_name]
        if pd.notna(prev_val):
            return prev_val
    return np.nan


def _correct_format(state: pd.DataFrame, row: int, col_name: str, schema: dict):
    val = state.at[state.index[row], col_name]
    col_lower = col_name.lower()

    if col_name == "department_name" and "department_id" in state.columns:
        repaired_name = expected_department_name(state.at[state.index[row], "department_id"])
        if repaired_name is not None:
            return repaired_name

    if col_name == "department_id" and "department_name" in state.columns:
        repaired_id = expected_department_id(state.at[state.index[row], "department_name"])
        if repaired_id is not None:
            return repaired_id

    if "email" in col_lower:
        val_str = str(val)
        if re.match(r"^[\w.\-+]+@[\w.\-]+\.\w{2,}$", val_str):
            return val_str
        return val

    if "phone" in col_lower:
        digits = re.sub(r"\D", "", str(val))
        if len(digits) >= 10:
            return digits[:10]
        return val

    if "date" in col_lower:
        try:
            return pd.to_datetime(str(val)).strftime("%Y-%m-%d")
        except Exception:
            return val

    if "age" in col_lower:
        reference_year = infer_reference_year(state.drop(columns=["_is_deleted"], errors="ignore"))
        if reference_year is not None and "birth_year" in state.columns:
            try:
                birth_year = int(float(str(state.at[state.index[row], "birth_year"])))
                inferred_age = int(reference_year - birth_year)
                if 0 < inferred_age < 150:
                    return inferred_age
            except Exception:
                pass
        try:
            candidate = int(float(str(val)))
            return candidate if 0 < candidate < 150 else val
        except Exception:
            return val

    return val


def _merge_duplicate(state: pd.DataFrame, row: int):
    """Find nearest near-duplicate row and merge, preferring non-null values."""
    compare_cols = [c for c in state.columns if c != "_is_deleted"]
    if not compare_cols:
        return

    if "_is_deleted" in state.columns and bool(state.at[state.index[row], "_is_deleted"]):
        return

    active_mask = pd.Series(True, index=state.index)
    if "_is_deleted" in state.columns:
        active_mask = ~state["_is_deleted"].fillna(False)

    candidate_frame = state.loc[active_mask, compare_cols]
    target_idx = state.index[row]
    if target_idx not in candidate_frame.index or len(candidate_frame) < 2:
        return

    target_row = candidate_frame.loc[target_idx]
    overlaps = candidate_frame.eq(target_row).sum(axis=1).drop(index=target_idx, errors="ignore")
    if overlaps.empty:
        return

    best_match_idx = overlaps.idxmax()
    best_overlap = int(overlaps.max())
    if best_overlap <= len(compare_cols) * 0.7:
        return

    for col_name in compare_cols:
        if pd.isna(state.at[target_idx, col_name]):
            state.at[target_idx, col_name] = state.at[best_match_idx, col_name]

    if "_is_deleted" not in state.columns:
        state["_is_deleted"] = False
    state.at[best_match_idx, "_is_deleted"] = True
