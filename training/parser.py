import re
import json
from environment.env import SurgeonAction


def robust_parse_action(completion: str) -> SurgeonAction:
    """
    Never crash on malformed model output.
    Try 3 strategies before giving up.
    """
    text = completion.strip()
    
    # Strategy 1: direct JSON parse
    try:
        return SurgeonAction(**json.loads(text))
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
            return SurgeonAction(**json.loads(candidate))
        except Exception:
            pass
    
    # Strategy 3: field-by-field regex extraction
    try:
        reasoning_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        tool_m      = re.search(r'"tool_id"\s*:\s*(\d+)', text)
        col_m       = re.search(r'"column"\s*:\s*(\d+)', text)
        row_m       = re.search(r'"row_id"\s*:\s*(\d+)', text)
        
        if all([reasoning_m, tool_m, col_m, row_m]):
            return SurgeonAction(
                reasoning=reasoning_m.group(1),
                tool_id=min(int(tool_m.group(1)), 7),
                column=int(col_m.group(1)),
                row_id=int(row_m.group(1)),
            )
    except Exception:
        pass
    
    raise ValueError(f"Cannot parse: {text[:80]}")
