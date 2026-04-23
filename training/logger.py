import csv
import os
from datetime import datetime


class TrainingLogger:
    def __init__(self, path="training_log.csv"):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow([
                "timestamp", "step", "total_reward",
                "accuracy_delta", "tool_logic", "reasoning",
                "efficiency", "anti_hack", "difficulty",
                "model_label", "parse_success_rate",
            ])
    
    def log(self, step: int, reward_dict: dict,
            difficulty: int, model_label: str,
            parse_successes: int, total_rollouts: int):
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                step,
                round(reward_dict.get("total", 0), 4),
                round(reward_dict.get("accuracy_delta", 0), 4),
                round(reward_dict.get("tool_logic", 0), 4),
                round(reward_dict.get("reasoning", 0), 4),
                round(reward_dict.get("efficiency", 0), 4),
                round(reward_dict.get("anti_hack", 0), 4),
                difficulty,
                model_label,
                round(parse_successes / max(total_rollouts, 1), 3),
            ])
    
    def detect_collapse(self, recent_actions: list, threshold=0.75) -> bool:
        """Returns True if agent is collapsing into one tool."""
        if len(recent_actions) < 20:
            return False
        tool_counts = {}
        for a in recent_actions:
            t = a.get("tool_id", -1)
            tool_counts[t] = tool_counts.get(t, 0) + 1
        top_rate = max(tool_counts.values()) / len(recent_actions)
        if top_rate > threshold:
            dominant = max(tool_counts, key=tool_counts.get)
            print(f"[COLLAPSE WARNING] Tool {dominant} at {top_rate:.0%} of actions")
            return True
        return False
