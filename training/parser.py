from __future__ import annotations

import json
import re

from environment.env import SurgeonAction
from environment.schemas import SURGEON_TOOLS


_TOOL_NAME_TO_ID = {}
for tool_id, info in SURGEON_TOOLS.items():
    name = str(info.get("name", ""))
    variants = {
        name,
        name.lower(),
        name.replace("_", " "),
        name.replace("_", "-"),
        name.replace("_", ""),
    }
    for variant in variants:
        normalized = re.sub(r"[^a-z0-9]+", "", variant.lower())
        if normalized:
            _TOOL_NAME_TO_ID[normalized] = tool_id


_FIELD_ALIASES = {
    "_row_idx": "row_id",
    "row": "row_id",
    "row_idx": "row_id",
    "rowid": "row_id",
    "col": "column",
    "col_idx": "column",
    "column_idx": "column",
    "columnid": "column",
    "tool": "tool_id",
    "tool_name": "tool_id",
    "toolname": "tool_id",
    "action": "tool_id",
}


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


def _coerce_tool_id(value, default: int = 7) -> int:
    try:
        if value is None:
            return default
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        normalized = re.sub(r"[^a-z0-9]+", "", str(value).lower())
        return _TOOL_NAME_TO_ID.get(normalized, default)


def _clamp_action(reasoning: str, tool_id: int, column: int, row_id: int) -> SurgeonAction:
    return SurgeonAction(
        reasoning=reasoning,
        tool_id=min(max(_coerce_tool_id(tool_id, 7), 0), 7),
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


REQUIRED_ACTION_FIELDS = {"reasoning", "tool_id", "column", "row_id"}


def _normalize_action_dict(parsed: dict) -> dict:
    normalized = dict(parsed)
    lowered_keys = {str(key).lower(): key for key in normalized}

    for alias, canonical in _FIELD_ALIASES.items():
        if canonical in normalized:
            continue
        raw_key = lowered_keys.get(alias)
        if raw_key is not None:
            normalized[canonical] = normalized[raw_key]

    return normalized


def _validate_required_fields(parsed: dict, require_fields: bool):
    if require_fields and not REQUIRED_ACTION_FIELDS.issubset(parsed):
        missing = ", ".join(sorted(REQUIRED_ACTION_FIELDS - set(parsed)))
        raise ValueError(f"missing required action fields: {missing}")


def robust_parse_action(completion, require_fields: bool = False) -> SurgeonAction:
    """
    Never crash on malformed model output.
    Try 3 strategies before giving up.
    All strategies clamp tool_id to 0-7.
    """
    text = _completion_to_text(completion).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()

    try:
        parsed = _normalize_action_dict(json.loads(text))
        if not isinstance(parsed, dict):
            raise ValueError("expected object")
        _validate_required_fields(parsed, require_fields)
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
            parsed = _normalize_action_dict(json.loads(candidate))
            if not isinstance(parsed, dict):
                raise ValueError("expected object")
            _validate_required_fields(parsed, require_fields)
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
        tool_match = re.search(
            r'["\'](?:tool_id|tool|tool_name|action)["\']\s*:\s*(?:"([^"]+)"|\'([^\']+)\'|(-?\d+(?:\.\d+)?))',
            text,
        )
        column_match = re.search(r'["\'](?:column|col|column_idx|col_idx)["\']\s*:\s*(-?\d+(?:\.\d+)?)', text)
        row_match = re.search(r'["\'](?:row_id|row|row_idx|_row_idx)["\']\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if all([reasoning_match, tool_match, column_match, row_match]):
            return _clamp_action(
                reasoning_match.group(1),
                tool_match.group(1) or tool_match.group(2) or tool_match.group(3),
                column_match.group(1),
                row_match.group(1),
            )
    except Exception:
        pass

    raise ValueError(f"Cannot parse: {text[:80]}")
