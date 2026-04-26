"""
Repair legacy training_log.csv column names from the original training run.

Run:
    python scripts/fix_training_log.py
"""
from __future__ import annotations

import os
import shutil

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT_DIR, "logs", "training_log.csv")
BACKUP_PATH = os.path.join(ROOT_DIR, "logs", "training_log_original_backup.csv")
NOTE_TEXT = "columns renamed from training run; shaped_reward_total = sum of all shaped signals"
RENAME_MAP = {
    "tool_logic": "constraint_alignment",
    "reasoning": "schema_alignment",
    "efficiency": "outlier_targeting",
    "policy_shaping": "shaped_reward_total",
}


def main():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Missing training log: {LOG_PATH}")

    shutil.copyfile(LOG_PATH, BACKUP_PATH)

    df = pd.read_csv(LOG_PATH)
    renamed_count = sum(1 for source in RENAME_MAP if source in df.columns)
    df = df.rename(columns=RENAME_MAP)

    notes = [""] * len(df)
    if len(notes) > 0:
        notes[0] = NOTE_TEXT
    df["note"] = notes

    df.to_csv(LOG_PATH, index=False)

    shaped = pd.to_numeric(df.get("shaped_reward_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    nonzero_rows = int((shaped != 0).sum())
    mean_value = float(shaped.mean()) if len(shaped) else 0.0
    print(
        f"Renamed {renamed_count} columns. "
        f"shaped_reward_total nonzero in {nonzero_rows}/{len(df)} rows, mean={mean_value:.2f}"
    )


if __name__ == "__main__":
    main()
