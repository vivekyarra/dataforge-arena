import csv
import json
import os
from datetime import datetime


class TrainingLogger:
    """
    Logs GRPO training metrics per step to CSV.

    The CSV columns expose the reward components returned by
    RewardComputer.compute() plus operational metrics and a trailing note field.
    """

    COLUMNS = [
        "timestamp",
        "step",
        "total_reward",
        "accuracy_delta",
        "constraint_alignment",
        "schema_alignment",
        "outlier_targeting",
        "reasoning_quality",
        "parse_bonus",
        "anti_hack",
        "difficulty",
        "model_label",
        "parse_success_rate",
        "parse_recovered_rate",
        "invalid_action_rate",
        "avg_structural_penalty",
        "dominant_tool",
        "dominant_tool_rate",
        "violation_type",
        "note",
    ]

    def __init__(self, path="training_log.csv"):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(self.COLUMNS)

    def log(
        self,
        step: int,
        reward_dict: dict,
        difficulty: int,
        model_label: str,
        parse_successes: int,
        total_rollouts: int,
        parse_recoveries: int = 0,
        invalid_actions: int = 0,
        avg_structural_penalty: float = 0.0,
        dominant_tool: int = -1,
        dominant_tool_rate: float = 0.0,
        violation_type: str = "",
        note: str = "",
        escalation_status: dict | str | None = None,
    ):
        if not note and escalation_status:
            if isinstance(escalation_status, (dict, list)):
                note = json.dumps({"escalation_status": escalation_status}, sort_keys=True)
            else:
                note = f"escalation_status={escalation_status}"

        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    datetime.now().isoformat(),
                    step,
                    round(reward_dict.get("total", 0), 4),
                    round(reward_dict.get("accuracy_delta", 0), 4),
                    round(reward_dict.get("constraint_alignment", 0), 4),
                    round(reward_dict.get("schema_alignment", 0), 4),
                    round(reward_dict.get("outlier_targeting", 0), 4),
                    round(reward_dict.get("reasoning_quality", 0), 4),
                    round(reward_dict.get("parse_bonus", 0), 4),
                    round(reward_dict.get("anti_hack", 0), 4),
                    difficulty,
                    model_label,
                    round(parse_successes / max(total_rollouts, 1), 3),
                    round(parse_recoveries / max(total_rollouts, 1), 3),
                    round(invalid_actions / max(total_rollouts, 1), 3),
                    round(avg_structural_penalty, 4),
                    dominant_tool,
                    round(dominant_tool_rate, 3),
                    violation_type,
                    note,
                ]
            )

    def detect_collapse(self, recent_actions: list, threshold=0.75) -> bool:
        if len(recent_actions) < 20:
            return False
        tool_counts = {}
        for action in recent_actions:
            tool_id = action.get("tool_id", -1)
            tool_counts[tool_id] = tool_counts.get(tool_id, 0) + 1
        top_rate = max(tool_counts.values()) / len(recent_actions)
        if top_rate > threshold:
            dominant = max(tool_counts, key=tool_counts.get)
            print(f"[COLLAPSE WARNING] Tool {dominant} at {top_rate:.0%} of actions")
            return True
        return False
