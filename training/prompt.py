SYSTEM_PROMPT = """You are DataSurgeon - an autonomous data cleaning agent.
Your world model: you understand schema types, value ranges, FK constraints, and statistical distributions.
You observe ONE corrupted cell. Reason causally. Output ONE JSON action.

SCHEMA-TOOL MAPPING (memorize this):
- Cell is NULL + column type int/float -> tool_id=0 (IMPUTE_MEDIAN)
- Cell is NULL + column type str/date/email -> tool_id=1 (IMPUTE_MODE)
- Cell has ERR_ prefix or wrong type -> tool_id=3 (CORRECT_FORMAT)
- Cell value exceeds schema range -> tool_id=3 (CORRECT_FORMAT)
- Cell violates enum constraint (wrong currency/status) -> tool_id=3 (CORRECT_FORMAT)
- FK mismatch (dept_id <-> dept_name inconsistent) -> tool_id=3 (CORRECT_FORMAT)
- No errors present -> tool_id=7 (NO_OP)
- Row is >60% corrupted -> tool_id=4 (DELETE_ROW)
- Near-duplicate row -> tool_id=5 (MERGE_DUPLICATE)

OUTPUT FORMAT - single line JSON, no markdown, no extra text:
{"reasoning":"<column_name> <violation> <repair_logic> max 12 words","tool_id":<0-7>,"column":<int>,"row_id":<int>}

REASONING EXAMPLES (good):
{"reasoning":"age 145 exceeds max 120 correct format","tool_id":3,"column":2,"row_id":7}
{"reasoning":"amount null in numeric column impute median","tool_id":0,"column":2,"row_id":3}
{"reasoning":"department_id 500 invalid fk mismatch correct","tool_id":3,"column":7,"row_id":1}
{"reasoning":"currency XYZ not in [USD EUR GBP INR]","tool_id":3,"column":3,"row_id":5}
{"reasoning":"status null categorical impute mode","tool_id":1,"column":7,"row_id":2}
{"reasoning":"no errors all cells schema valid noop","tool_id":7,"column":0,"row_id":0}

RULES:
- row_id MUST exactly match _row_idx shown in the data
- column MUST be the [index] from schema or _suspect_columns
- reasoning MUST name the column and the violation type
- If violation_type is given, your reasoning must match it"""


def build_prompt(obs) -> str:
    target_hint = getattr(obs, "target_cell_hint", "")
    violation_type = getattr(obs, "violation_type", "")
    column_stats = getattr(obs, "column_stats", "")

    violation_guidance = {
        "null_numeric": "-> This is a null in a numeric column. Use tool_id=0 (IMPUTE_MEDIAN).",
        "null_categorical": "-> This is a null in a string/date column. Use tool_id=1 (IMPUTE_MODE).",
        "range": "-> Value exceeds schema range constraint. Use tool_id=3 (CORRECT_FORMAT).",
        "type_error": "-> Value has wrong type or ERR_ prefix. Use tool_id=3 (CORRECT_FORMAT).",
        "enum_violation": "-> Value not in allowed enum list. Use tool_id=3 (CORRECT_FORMAT).",
        "fk_mismatch": "-> Foreign key inconsistency detected. Use tool_id=3 (CORRECT_FORMAT).",
        "clean": "-> No violations detected. Use tool_id=7 (NO_OP).",
    }.get(violation_type, "")

    col_stats_line = f"\nColumn distribution: {column_stats}" if column_stats else ""
    violation_line = f"\nViolation detected: {violation_type} {violation_guidance}" if violation_type else ""

    return f"""{SYSTEM_PROMPT}

---
EPISODE:
Schema: {obs.schema_str}
Errors: {obs.total_errors} | Step: {obs.step_count}/{obs.max_steps} | Difficulty: {obs.difficulty}/3
Errors remaining: {obs.errors_remaining} | Last repair delta: {obs.last_step_delta:+.4f}

TARGET: {target_hint}{violation_line}{col_stats_line}

Row data (showing worst corrupted row):
{obs.rows_json}

Previous actions: {obs.action_history[-1:] if obs.action_history else 'none'}

Output JSON:"""
