# All data schemas and tool definitions in one place

HEALTHCARE_SCHEMA = {
    "patient_id":     {"type": "int",    "nullable": False, "range": (1, 99999)},
    "name":           {"type": "str",    "nullable": False},
    "age":            {"type": "int",    "nullable": False, "range": (0, 120)},
    "birth_year":     {"type": "int",    "nullable": False, "range": (1900, 2024)},
    "email":          {"type": "email",  "nullable": True},
    "phone":          {"type": "phone",  "nullable": True},
    "diagnosis":      {"type": "str",    "nullable": False},
    "department_id":  {"type": "int",    "nullable": False, "range": (1, 10)},
    "department_name":{"type": "str",    "nullable": False},
    "admission_date": {"type": "date",   "nullable": False},
}

FINANCIAL_SCHEMA = {
    "transaction_id": {"type": "int",    "nullable": False},
    "account_id":     {"type": "int",    "nullable": False, "range": (1000, 9999)},
    "amount":         {"type": "float",  "nullable": False, "range": (0.01, 100000)},
    "currency":       {"type": "str",    "nullable": False, "values": ["USD","EUR","GBP","INR"]},
    "transaction_date":{"type": "date",  "nullable": False},
    "merchant":       {"type": "str",    "nullable": False},
    "category":       {"type": "str",    "nullable": False},
    "status":         {"type": "str",    "nullable": False, "values": ["completed","pending","failed"]},
}

DEPT_MAP = {
    1: "Cardiology", 2: "Neurology", 3: "Oncology",
    4: "Pediatrics", 5: "Orthopedics", 6: "Radiology",
    7: "Emergency",  8: "Surgery",    9: "Psychiatry", 10: "General",
}

# SURGEON tools — discrete integers 0-7 only
SURGEON_TOOLS = {
    0: {"name": "IMPUTE_MEDIAN",       "applies_to": ["int", "float"]},
    1: {"name": "IMPUTE_MODE",         "applies_to": ["str", "int", "float"]},
    2: {"name": "IMPUTE_FORWARD_FILL", "applies_to": ["int", "float", "str", "date"]},
    3: {"name": "CORRECT_FORMAT",      "applies_to": ["email", "phone", "date", "str"]},
    4: {"name": "DELETE_ROW",          "applies_to": ["any"]},
    5: {"name": "MERGE_DUPLICATE",     "applies_to": ["any"]},
    6: {"name": "FLAG_UNCERTAIN",      "applies_to": ["any"]},
    7: {"name": "NO_OP",               "applies_to": ["any"]},
}

# CORRUPTOR tools — named sabotage operations
CORRUPTOR_TOOLS = {
    # Tier 1 — always detectable, always recoverable
    "inject_null_single":  {"tier": 1, "reward": 1.0, "recoverable": True},
    "inject_type_error":   {"tier": 1, "reward": 1.5, "recoverable": True},
    # Tier 2 — harder to spot
    "inject_null_cluster": {"tier": 2, "reward": 2.0, "recoverable": True},
    "swap_date_format":    {"tier": 2, "reward": 2.5, "recoverable": True},
    "inject_out_of_range_age":    {"tier": 2, "reward": 3.0, "recoverable": True},
    # Tier 3 — relational reasoning required
    "break_foreign_key":   {"tier": 3, "reward": 4.0, "recoverable": True},
    "duplicate_row_mutate":{"tier": 3, "reward": 4.5, "recoverable": True},
    # BANNED — always rejected by solvability gate
    "delete_row":          {"tier": 0, "reward": 0,   "recoverable": False, "banned": True},
    "null_entire_column":  {"tier": 0, "reward": 0,   "recoverable": False, "banned": True},
    "random_noise":        {"tier": 0, "reward": 0,   "recoverable": False, "banned": True},
}
