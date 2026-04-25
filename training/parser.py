import json
import re

from environment.env import SurgeonAction


def _completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion["content"])
        return json.dumps(completion)
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def _coerce_int(value, default: int) -> int:
    try:
        if value is None:
            return default
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _clamp_action(reasoning: str, tool_id: int, column: int, row_id: int) -> SurgeonAction:
    return SurgeonAction(
        reasoning=reasoning,
        tool_id=min(max(_coerce_int(tool_id, 7), 0), 7),
        column=max(_coerce_int(column, 0), 0),
        row_id=max(_coerce_int(row_id, 0), 0),
    )


def _extract_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        start = text.find("{", start + 1)
    return None


def robust_parse_action(completion) -> SurgeonAction:
    """
    Never crash on malformed model output.
    Try 3 strategies before giving up.
    All strategies clamp tool_id to 0-7.
    """
    text = _completion_to_text(completion).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("expected object")
        return _clamp_action(
            str(parsed.get("reasoning", "")),
            parsed.get("tool_id", 7),
            parsed.get("column", 0),
            parsed.get("row_id", 0),
        )
    except Exception:
        pass

    candidate = _extract_balanced_json_object(text)
    if candidate:
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r"'", '"', candidate)
        candidate = re.sub(r"(\w+):", r'"\1":', candidate)
        candidate = re.sub(r'"{2,}', '"', candidate)
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise ValueError("expected object")
            return _clamp_action(
                str(parsed.get("reasoning", "")),
                parsed.get("tool_id", 7),
                parsed.get("column", 0),
                parsed.get("row_id", 0),
            )
        except Exception:
            pass

    try:
        reasoning_match = re.search(r'["\']reasoning["\']\s*:\s*["\']([^"\']*)["\']', text)
        tool_match = re.search(r'["\']tool_id["\']\s*:\s*(-?\d+(?:\.\d+)?)', text)
        column_match = re.search(r'["\']column["\']\s*:\s*(-?\d+(?:\.\d+)?)', text)
        row_match = re.search(r'["\']row_id["\']\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if all([reasoning_match, tool_match, column_match, row_match]):
            return _clamp_action(
                reasoning_match.group(1),
                tool_match.group(1),
                column_match.group(1),
                row_match.group(1),
            )
    except Exception:
        pass

    raise ValueError(f"Cannot parse: {text[:80]}")
