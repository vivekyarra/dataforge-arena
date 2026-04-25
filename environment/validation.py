import re
from typing import Optional

import pandas as pd

from environment.schemas import DEPT_MAP


EMAIL_RE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w{2,}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DEPT_NAME_TO_ID = {str(name).strip().lower(): dept_id for dept_id, name in DEPT_MAP.items()}


def _coerce_float(value) -> Optional[float]:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def infer_reference_year(df: pd.DataFrame) -> Optional[int]:
    if not {"age", "birth_year"}.issubset(df.columns):
        return None

    ages = pd.to_numeric(df["age"], errors="coerce")
    birth_years = pd.to_numeric(df["birth_year"], errors="coerce")
    valid = (
        ages.notna()
        & birth_years.notna()
        & ages.between(0, 120)
        & birth_years.between(1900, 2100)
    )
    if not valid.any():
        return None

    reference_years = (ages[valid].round() + birth_years[valid].round()).astype(int)
    modes = reference_years.mode()
    if not modes.empty:
        return int(modes.iloc[0])
    return int(round(reference_years.median()))


def expected_department_name(department_id) -> Optional[str]:
    numeric = _coerce_float(department_id)
    if numeric is None or not float(numeric).is_integer():
        return None
    return DEPT_MAP.get(int(numeric))


def expected_department_id(department_name) -> Optional[int]:
    key = str(department_name).strip().lower()
    if not key:
        return None
    return DEPT_NAME_TO_ID.get(key)


def cell_has_error(
    value,
    col_name: str,
    schema: dict,
    row: Optional[pd.Series] = None,
    reference_year: Optional[int] = None,
) -> bool:
    if pd.isna(value):
        return True

    if isinstance(value, str) and value.startswith("ERR_"):
        return True

    schema_info = schema.get(col_name, {})
    col_type = schema_info.get("type", "str")
    value_str = str(value).strip()

    numeric = None
    if col_type in ("int", "float"):
        numeric = _coerce_float(value)
        if numeric is None:
            return True
        if col_type == "int" and not float(numeric).is_integer():
            return True

    if col_type == "email" and not EMAIL_RE.match(value_str):
        return True

    if col_type == "phone":
        digits = re.sub(r"\D", "", value_str)
        if len(digits) != 10:
            return True

    if col_type == "date":
        if not DATE_RE.match(value_str):
            return True
        if pd.isna(pd.to_datetime(value_str, format="%Y-%m-%d", errors="coerce")):
            return True

    if "values" in schema_info and value_str not in schema_info["values"]:
        return True

    if "range" in schema_info and numeric is not None:
        lo, hi = schema_info["range"]
        if numeric < lo or numeric > hi:
            return True

    if col_name == "age" and row is not None and reference_year is not None and "birth_year" in row.index:
        birth_year = _coerce_float(row["birth_year"])
        current_age = _coerce_float(value)
        if (
            birth_year is not None
            and current_age is not None
            and float(birth_year).is_integer()
        ):
            expected_age = int(reference_year - int(birth_year))
            if 0 <= expected_age <= 120 and int(round(current_age)) != expected_age:
                return True

    if col_name == "department_id":
        expected_name = expected_department_name(value)
        if expected_name is None:
            return True
        if row is not None and "department_name" in row.index:
            other_name = str(row["department_name"]).strip()
            other_expected_id = expected_department_id(other_name)
            if other_expected_id is not None and other_name != expected_name:
                return True

    if col_name == "department_name":
        expected_id = expected_department_id(value)
        if expected_id is None:
            return True
        if row is not None and "department_id" in row.index:
            other_id = _coerce_float(row["department_id"])
            if (
                other_id is not None
                and float(other_id).is_integer()
                and int(other_id) in DEPT_MAP
                and int(other_id) != expected_id
            ):
                return True

    return False


def summarize_corruption(df: pd.DataFrame, schema: dict) -> tuple[list[int], int]:
    row_scores, total_errors, _ = summarize_corruption_details(df, schema)
    return row_scores, total_errors


def summarize_corruption_details(
    df: pd.DataFrame,
    schema: dict,
    max_issue_columns: int | None = None,
) -> tuple[list[int], int, list[list[str]]]:
    if df is None or df.empty:
        return [], 0, []

    reference_year = infer_reference_year(df)
    row_scores = []
    total_errors = 0
    row_issue_columns: list[list[str]] = []

    for _, row in df.iterrows():
        issue_columns = []
        for col_name in df.columns:
            has_error = cell_has_error(
                row[col_name],
                col_name,
                schema=schema,
                row=row,
                reference_year=reference_year,
            )
            if has_error:
                issue_columns.append(col_name)
                total_errors += 1
        full_issue_count = len(issue_columns)
        if max_issue_columns is not None:
            issue_columns = issue_columns[:max_issue_columns]
        row_scores.append(full_issue_count)
        row_issue_columns.append(issue_columns)

    return row_scores, total_errors, row_issue_columns
