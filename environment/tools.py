import pandas as pd
import numpy as np
import re
from environment.schemas import SURGEON_TOOLS

def apply_tool(state: pd.DataFrame, action,
               schema: dict) -> pd.DataFrame:
    """
    Apply SURGEON tool to state DataFrame.
    Returns modified DataFrame.
    Uses soft-delete -- never physically removes rows.
    """
    tool_name = SURGEON_TOOLS[action.tool_id]["name"]
    row = action.row_id
    col = action.column
    
    if row >= len(state) or col >= len(state.columns):
        return state  # out of bounds -- no change
    
    col_name = state.columns[col]
    
    # Upcast column to object to prevent pandas dtype warnings
    if col_name != "_is_deleted" and state[col_name].dtype != object:
        state[col_name] = state[col_name].astype(object)
    
    if tool_name == "IMPUTE_MEDIAN":
        col_data = state.iloc[:, col]
        numeric = pd.to_numeric(col_data, errors="coerce")
        median = numeric.median()
        if pd.notna(median):
            state.iloc[row, col] = median
    
    elif tool_name == "IMPUTE_MODE":
        col_data = state.iloc[:, col].dropna()
        if len(col_data) > 0:
            mode = col_data.mode()
            if len(mode) > 0:
                state.iloc[row, col] = mode.iloc[0]
    
    elif tool_name == "IMPUTE_FORWARD_FILL":
        if row > 0:
            prev_val = state.iloc[row - 1, col]
            if pd.notna(prev_val):
                state.iloc[row, col] = prev_val
    
    elif tool_name == "CORRECT_FORMAT":
        state.iloc[row, col] = _correct_format(
            state.iloc[row, col], col_name, schema
        )
    
    elif tool_name == "DELETE_ROW":
        # Soft delete -- never physically remove
        if "_is_deleted" not in state.columns:
            state["_is_deleted"] = False
        state.at[state.index[row], "_is_deleted"] = True
    
    elif tool_name == "MERGE_DUPLICATE":
        _merge_duplicate(state, row, col)
    
    elif tool_name in ("FLAG_UNCERTAIN", "NO_OP"):
        pass  # no state change
    
    return state


def _correct_format(val, col_name: str, schema: dict):
    col_lower = col_name.lower()
    
    if "email" in col_lower:
        val_str = str(val)
        if re.match(r'^[\w.\-+]+@[\w.\-]+\.\w{2,}$', val_str):
            return val_str
        return val  # return original instead of None — agent should use FLAG_UNCERTAIN
    
    if "phone" in col_lower:
        digits = re.sub(r'\D', '', str(val))
        if len(digits) >= 10:
            return digits[:10]
        return None
    
    if "date" in col_lower:
        try:
            return pd.to_datetime(str(val)).strftime('%Y-%m-%d')
        except Exception:
            return None
    
    if "age" in col_lower:
        try:
            v = int(float(str(val)))
            return v if 0 < v < 150 else None
        except Exception:
            return None
    
    return val  # no rule -- return unchanged


def _merge_duplicate(state: pd.DataFrame, row: int, col: int):
    """Find nearest near-duplicate row and merge, preferring non-null values."""
    # BUG 8 FIX: Exclude _is_deleted from overlap comparison
    compare_cols = [c for c in state.columns if c != "_is_deleted"]
    target_row = state.iloc[row][compare_cols]
    best_match_idx = None
    best_overlap = 0
    
    for i, other_row in state.iterrows():
        if i == state.index[row]:
            continue
        other_vals = other_row[compare_cols]
        overlap = (target_row == other_vals).sum()
        if overlap > best_overlap:
            best_overlap = overlap
            best_match_idx = i
    
    if best_match_idx is not None and best_overlap > len(compare_cols) * 0.7:
        # Merge: fill nulls in target from match
        for c in compare_cols:
            if pd.isna(state.at[state.index[row], c]):
                state.at[state.index[row], c] = state.at[best_match_idx, c]
        # Soft-delete the duplicate
        if "_is_deleted" not in state.columns:
            state["_is_deleted"] = False
        state.at[best_match_idx, "_is_deleted"] = True
