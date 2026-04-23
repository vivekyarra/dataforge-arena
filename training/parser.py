import re
import json
from environment.env import SurgeonAction


def _clamp_action(reasoning: str, tool_id: int, column: int, row_id: int) -> SurgeonAction:
    """Construct a SurgeonAction with clamped values. BUG 4 FIX."""
    return SurgeonAction(
        reasoning=reasoning,
        tool_id=min(max(int(tool_id), 0), 7),
        column=max(int(column), 0),
        row_id=max(int(row_id), 0),
    )


def robust_parse_action(completion: str) -> SurgeonAction:
    """
    Never crash on malformed model output.
    Try 3 strategies before giving up.
    All strategies clamp tool_id to 0-7.
    """
    text = completion.strip()
    
    # Strategy 1: direct JSON parse
    try:
        d = json.loads(text)
        return _clamp_action(
            str(d.get("reasoning", "")),
            d.get("tool_id", 7),
            d.get("column", 0),
            d.get("row_id", 0),
        )
    except Exception:
        pass
    
    # Strategy 2: extract JSON object from surrounding text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        candidate = match.group()
        # Fix common model mistakes
        candidate = re.sub(r',\s*}', '}', candidate)   # trailing comma
        candidate = re.sub(r"'", '"', candidate)         # single quotes
        candidate = re.sub(r'(\w+):', r'"\1":', candidate)  # unquoted keys
        candidate = re.sub(r'"{2,}', '"', candidate)    # double-double quotes
        try:
            d = json.loads(candidate)
            return _clamp_action(
                str(d.get("reasoning", "")),
                d.get("tool_id", 7),
                d.get("column", 0),
                d.get("row_id", 0),
            )
        except Exception:
            pass
    
    # Strategy 3: field-by-field regex extraction
    try:
        reasoning_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        tool_m      = re.search(r'"tool_id"\s*:\s*(\d+)', text)
        col_m       = re.search(r'"column"\s*:\s*(\d+)', text)
        row_m       = re.search(r'"row_id"\s*:\s*(\d+)', text)
        
        if all([reasoning_m, tool_m, col_m, row_m]):
            return _clamp_action(
                reasoning_m.group(1),
                int(tool_m.group(1)),
                int(col_m.group(1)),
                int(row_m.group(1)),
            )
    except Exception:
        pass
    
    raise ValueError(f"Cannot parse: {text[:80]}")
