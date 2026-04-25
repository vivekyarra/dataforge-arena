import html
import json
import logging
import os
import sys
import threading
import time
import warnings
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.corruptor import Corruptor
from environment.env import DataForgeEnv, SurgeonAction
from environment.reward import RewardComputer
from environment.schemas import HEALTHCARE_SCHEMA, SURGEON_TOOLS
from environment.validation import summarize_corruption
from eval.evaluate import load_eval_pipeline, _resolve_loadable_model_path
from training.parser import robust_parse_action
from training.prompt import build_prompt


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "healthcare_clean.csv")
LOG_PATH = os.path.join(ROOT_DIR, "logs", "training_log.csv")
EVAL_RESULTS_PATH = os.path.join(ROOT_DIR, "eval", "results.json")
LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, "outputs", "dataforge-surgeon")

clean_data = pd.read_csv(DATA_PATH)
rc = RewardComputer()
llm_pipeline = None
llm_lock = threading.Lock()
logger = logging.getLogger("dataforge.demo")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)
logger.setLevel(os.getenv("DATAFORGE_LOG_LEVEL", "INFO"))
logger.propagate = False


def _preferred_inference_dtype(torch_module, device: int):
    if device < 0:
        return torch_module.float32
    major, _ = torch_module.cuda.get_device_capability(device)
    return torch_module.bfloat16 if major >= 8 else torch_module.float16


def local_model_available(model_path: str = LOCAL_MODEL_PATH) -> bool:
    try:
        _resolve_loadable_model_path(model_path)
        return True
    except FileNotFoundError:
        return False


def available_agent_choices(model_available: bool | None = None) -> list[str]:
    resolved = local_model_available() if model_available is None else model_available
    choices = ["Naive Baseline", "Heuristic Surgeon"]
    if resolved:
        choices.append("Live GRPO Model")
    return choices


def _new_session_state():
    return {"dirty": None, "gt": None, "meta": None, "tier": 1}


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _format_pp(value: float | None) -> str:
    if value is None:
        return "Pending"
    return f"{float(value) * 100:+.2f} pp"


def _status_tone(value: float | None) -> str:
    if value is None:
        return "neutral"
    return "good" if value >= 0 else "bad"


def _align_duplicate_ground_truth(dirty: pd.DataFrame, gt: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if meta.get("tool") == "duplicate_row_mutate" and len(dirty) > len(gt):
        src = meta.get("row", 0)
        if src < len(gt):
            return pd.concat([gt, gt.iloc[[src]]], ignore_index=True)
    return gt


def _build_rollout_env(dirty: pd.DataFrame, gt: pd.DataFrame, tier: int):
    local_corruptor = Corruptor()
    local_corruptor.force_tier(tier)
    env = DataForgeEnv(corruptor=local_corruptor, schema=HEALTHCARE_SCHEMA, clean_data=clean_data)
    starting_acc = rc._field_accuracy(dirty, gt)
    env._state = dirty.copy()
    env._ground_truth = gt.copy()
    env._original_dirty = dirty.copy()
    env._prev_accuracy = starting_acc
    env._starting_accuracy = starting_acc
    env._step_count = 0
    env._action_log = []
    env._episode_rewards = []
    env._episode_start = time.time()
    return env, starting_acc


def load_llm():
    global llm_pipeline
    if not local_model_available():
        return False, (
            "Live GRPO mode is unavailable. No local checkpoint was found at "
            f"{LOCAL_MODEL_PATH}."
        )

    with llm_lock:
        if llm_pipeline is not None:
            return True, "Loaded local GRPO checkpoint."

        try:
            logger.info("Loading live GRPO checkpoint from %s", LOCAL_MODEL_PATH)
            llm_pipeline = load_eval_pipeline(LOCAL_MODEL_PATH)
            logger.info("Live GRPO checkpoint loaded successfully.")
            return True, "Loaded local GRPO checkpoint."
        except Exception as exc:
            llm_pipeline = None
            logger.exception("Error loading local GRPO checkpoint")
            return False, f"Failed to load the local GRPO checkpoint: {exc}"


def _run_llm(messages):
    with llm_lock:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The following generation flags are not valid.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Both `max_new_tokens`.*",
                category=UserWarning,
            )
            return llm_pipeline(
                messages,
                max_new_tokens=96,
                temperature=0.1,
                do_sample=False,
                num_return_sequences=1,
            )


def _agent_provenance(agent_type: str) -> tuple[str, str]:
    if agent_type == "Naive Baseline":
        return "Rule-based baseline", "Simple null and ERR_* scan with no learned policy."
    if agent_type == "Heuristic Surgeon":
        return "Rule-based heuristic", "Deterministic repair policy used for the current committed evaluation evidence."
    return "Local GRPO checkpoint", "Live inference from a local trained checkpoint in outputs/dataforge-surgeon."


def _read_eval_results() -> dict:
    try:
        with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _read_baseline_results() -> dict:
    try:
        with open(os.path.join(ROOT_DIR, "eval", "heuristic_results.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _training_summary() -> dict:
    df = get_training_data()
    summary = {
        "parse_success": None,
        "parse_first": None,
        "parse_last": None,
        "tiers": "Pending",
        "last_step": "Pending",
        "latest_reward": None,
        "best_reward": None,
    }
    if df.empty:
        return summary

    if "parse_success_rate" in df:
        values = pd.to_numeric(df["parse_success_rate"], errors="coerce").dropna()
        if len(values) > 0:
            summary["parse_success"] = float(values.mean() * 100)
            summary["parse_first"] = float(values.iloc[0] * 100)
            summary["parse_last"] = float(values.iloc[-1] * 100)

    if "difficulty" in df:
        tiers = sorted(pd.to_numeric(df["difficulty"], errors="coerce").dropna().astype(int).unique())
        if tiers:
            summary["tiers"] = f"{tiers[0]}-{tiers[-1]}" if len(tiers) > 1 else str(tiers[0])

    if "step" in df:
        steps = pd.to_numeric(df["step"], errors="coerce").dropna()
        if len(steps) > 0:
            summary["last_step"] = str(int(steps.max()))

    if "total_reward" in df:
        rewards = pd.to_numeric(df["total_reward"], errors="coerce").dropna()
        if len(rewards) > 0:
            summary["latest_reward"] = float(rewards.iloc[-1])
            summary["best_reward"] = float(rewards.max())

    return summary


def _metric_card(label: str, value: str, detail: str, tone: str = "neutral") -> str:
    return (
        f"<div class='metric-card metric-{tone}'>"
        f"<div class='metric-label'>{_escape(label)}</div>"
        f"<div class='metric-value'>{_escape(value)}</div>"
        f"<div class='metric-detail'>{_escape(detail)}</div>"
        "</div>"
    )


def _hero_html() -> str:
    return """
    <section class='hero-shell'>
      <div class='hero-copy'>
        <div class='eyebrow'>OpenEnv enterprise benchmark</div>
        <h1>DataForge Arena</h1>
        <p>
          A judge-visible RL arena where data repair agents face adversarial tabular corruption,
          choose grounded tools, and earn reward only when the dataset measurably improves.
        </p>
      </div>
      <div class='hero-steps'>
        <div><span>1</span><strong>Corrupt</strong><small>Generate solvable enterprise data failures.</small></div>
        <div><span>2</span><strong>Repair</strong><small>Run baseline, heuristic, or local GRPO surgeon.</small></div>
        <div><span>3</span><strong>Verify</strong><small>Inspect accuracy deltas and action traces.</small></div>
      </div>
    </section>
    """


def _evidence_snapshot_html() -> str:
    results = _read_eval_results()
    baseline = _read_baseline_results()
    training = _training_summary()
    checkpoint_ready = local_model_available()

    grpo_advantage = results.get("surgeon_advantage_accuracy_delta")
    grpo_delta = results.get("surgeon_avg_accuracy_delta")
    random_delta = results.get("random_avg_accuracy_delta")
    grpo_episodes = results.get("episodes", "Pending")
    heuristic_advantage = baseline.get("surgeon_advantage_accuracy_delta")
    heuristic_episodes = baseline.get("episodes", "Pending")
    parse_success = training["parse_success"]

    cards = [
        _metric_card(
            "Heuristic baseline",
            _format_pp(heuristic_advantage),
            f"Rule-based surgeon over {heuristic_episodes} eval episodes",
            _status_tone(heuristic_advantage),
        ),
        _metric_card(
            "GRPO checkpoint",
            _format_pp(grpo_advantage),
            f"GRPO delta {_format_pp(grpo_delta)} vs random {_format_pp(random_delta)} over {grpo_episodes} episodes",
            _status_tone(grpo_advantage),
        ),
        _metric_card(
            "Parse reliability",
            f"{parse_success:.2f}%" if parse_success is not None else "Pending",
            (
                f"First {training['parse_first']:.1f}% -> last {training['parse_last']:.1f}% | "
                f"tiers {training['tiers']}"
                if training["parse_first"] is not None and training["parse_last"] is not None
                else f"Logged GRPO curriculum through tiers {training['tiers']}"
            ),
            "good" if parse_success and parse_success >= 50 else "neutral",
        ),
        _metric_card(
            "Live GRPO mode",
            "Available" if checkpoint_ready else "Checkpoint gated",
            "Appears only when a loadable checkpoint or adapter exists",
            "good" if checkpoint_ready else "neutral",
        ),
    ]
    return "<section class='evidence-grid'>" + "".join(cards) + "</section>"


def _telemetry_intro_html():
    if local_model_available():
        availability = "Local GRPO checkpoint detected. Live inference is available for this session."
        tone = "good"
    else:
        availability = (
            "No local GRPO checkpoint detected. The demo stays honest and exposes baseline plus heuristic evidence paths."
        )
        tone = "neutral"
    return (
        f"<div class='metric-card metric-{tone} telemetry-card'>"
        "<div class='metric-label'>Mode inventory</div>"
        f"<div class='metric-detail'>{_escape(availability)}</div>"
        "</div>"
    )


def _empty_state_html():
    return (
        "<div class='metric-card telemetry-card'>"
        "<div class='metric-label'>Ready</div>"
        "<div class='metric-value compact'>Awaiting scenario</div>"
        "<div class='metric-detail'>Generate a Tier 1 or Tier 3 corruption episode, then execute an agent path.</div>"
        "</div>"
    )


def _scenario_ready_html(meta: dict, acc_before: float, total_errors: int) -> str:
    return (
        "<div class='metric-card metric-warn telemetry-card'>"
        "<div class='metric-label'>Scenario armed</div>"
        f"<div class='metric-value compact'>{_escape(meta.get('tool', 'unknown'))}</div>"
        f"<div class='metric-detail'>Starting health {_escape(f'{acc_before:.1%}')} with {_escape(total_errors)} visible schema issues.</div>"
        "</div>"
    )


def _score_card(label: str, value: float | None, tone: str = "neutral") -> str:
    display = f"{value:.1%}" if value is not None else "Pending"
    return _metric_card(label, display, "Field-level dataset accuracy", tone)


def _unavailable_html(message: str) -> str:
    return (
        "<div class='metric-card metric-bad telemetry-card'>"
        "<div class='metric-label'>Live model unavailable</div>"
        f"<div class='metric-detail'>{_escape(message)}</div>"
        "</div>"
    )


DARK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
:root {
  --df-bg: #070914;
  --df-panel: rgba(13, 18, 32, 0.86);
  --df-panel-strong: rgba(16, 24, 39, 0.94);
  --df-border: rgba(148, 163, 184, 0.18);
  --df-muted: #94a3b8;
  --df-text: #f8fafc;
  --df-soft: #cbd5e1;
  --df-good: #22c55e;
  --df-warn: #f59e0b;
  --df-bad: #fb7185;
  --df-blue: #38bdf8;
}
body,
.gradio-container {
  background:
    linear-gradient(135deg, rgba(56, 189, 248, 0.10) 0%, transparent 24%),
    linear-gradient(225deg, rgba(34, 197, 94, 0.09) 0%, transparent 28%),
    linear-gradient(180deg, #070914 0%, #0b1020 52%, #070914 100%) !important;
  color: var(--df-text);
  font-family: 'Inter', sans-serif;
}
.gradio-container { max-width: 1500px !important; }
.hero-shell {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(300px, 0.8fr);
  gap: 20px;
  align-items: stretch;
  padding: 28px;
  margin: 8px 0 18px;
  border: 1px solid var(--df-border);
  border-radius: 8px;
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(8, 13, 28, 0.86));
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.36), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.hero-copy h1 {
  margin: 6px 0 8px;
  color: var(--df-text);
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 40px;
  line-height: 1.04;
  letter-spacing: 0;
}
.hero-copy p {
  max-width: 820px;
  margin: 0;
  color: var(--df-soft);
  font-size: 16px;
  line-height: 1.65;
}
.eyebrow,
.metric-label,
.section-label {
  color: var(--df-muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}
.hero-steps {
  display: grid;
  gap: 10px;
}
.hero-steps div {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr);
  column-gap: 12px;
  align-items: center;
  padding: 12px;
  border: 1px solid var(--df-border);
  border-radius: 8px;
  background: rgba(15, 23, 42, 0.66);
}
.hero-steps span {
  grid-row: span 2;
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 8px;
  background: linear-gradient(135deg, rgba(34, 197, 94, 0.22), rgba(56, 189, 248, 0.20));
  color: var(--df-text);
  font: 700 13px 'JetBrains Mono', monospace;
}
.hero-steps strong {
  color: var(--df-text);
  font-size: 14px;
}
.hero-steps small {
  color: var(--df-muted);
  font-size: 12px;
  line-height: 1.35;
}
.evidence-grid,
.scenario-grid,
.rollout-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(150px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
.scenario-grid,
.rollout-metrics {
  grid-template-columns: repeat(3, minmax(130px, 1fr));
}
.rollout-metrics {
  grid-template-columns: repeat(4, minmax(130px, 1fr));
}
.panel {
  background: linear-gradient(180deg, var(--df-panel) 0%, rgba(8, 13, 26, 0.92) 100%);
  backdrop-filter: blur(16px);
  border: 1px solid var(--df-border);
  border-radius: 8px;
  padding: 18px;
  box-shadow: 0 18px 54px rgba(0, 0, 0, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.metric-card {
  background: linear-gradient(180deg, var(--df-panel-strong) 0%, rgba(8, 13, 26, 0.96) 100%);
  border: 1px solid var(--df-border);
  border-radius: 8px;
  padding: 15px;
  text-align: left;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 8px 20px rgba(0,0,0,0.25);
  word-break: break-word;
  min-height: 96px;
}
.metric-card.metric-good { border-color: rgba(34, 197, 94, 0.34); }
.metric-card.metric-warn { border-color: rgba(245, 158, 11, 0.34); }
.metric-card.metric-bad { border-color: rgba(251, 113, 133, 0.38); }
.metric-value {
  margin-top: 7px;
  color: var(--df-text);
  font: 700 28px/1.05 'JetBrains Mono', monospace;
}
.metric-value.compact {
  font-size: 18px;
  line-height: 1.22;
}
.metric-detail {
  margin-top: 8px;
  color: var(--df-soft);
  font-size: 12px;
  line-height: 1.45;
}
.telemetry-card { min-height: 0; margin-bottom: 12px; }
.rollout-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  padding: 11px 13px;
  margin: 7px 0;
  border-radius: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
}
.rollout-winner {
  background: linear-gradient(90deg, rgba(34,197,94,0.18) 0%, rgba(56,189,248,0.08) 100%);
  border-left: 3px solid var(--df-good);
}
.rollout-loser  {
  background: rgba(251,113,133,0.12);
  border-left: 3px solid var(--df-bad);
}
.tag {
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  white-space: nowrap;
}
.tag-null   { background: rgba(251, 113, 133, 0.15); color:#fecdd3; border: 1px solid rgba(251, 113, 133, 0.35); }
.tag-type   { background: rgba(245, 158, 11, 0.15); color:#fde68a; border: 1px solid rgba(245, 158, 11, 0.35); }
.tag-fixed  { background: rgba(34, 197, 94, 0.16); color:#bbf7d0; border: 1px solid rgba(34, 197, 94, 0.35); }
.tag-dup    { background: rgba(56, 189, 248, 0.16); color:#bae6fd; border: 1px solid rgba(56, 189, 248, 0.35); }
.gradio-container button.primary,
.gradio-container button[variant='primary'] {
  background: linear-gradient(135deg, #16a34a 0%, #0284c7 100%) !important;
  border: 0 !important;
  color: white !important;
}
.gradio-container table {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
}
@media (max-width: 980px) {
  .hero-shell { grid-template-columns: 1fr; padding: 20px; }
  .hero-copy h1 { font-size: 32px; }
  .evidence-grid,
  .scenario-grid,
  .rollout-metrics { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
}
@media (max-width: 640px) {
  .hero-copy h1 { font-size: 28px; }
  .evidence-grid,
  .scenario-grid,
  .rollout-metrics { grid-template-columns: 1fr; }
  .panel { padding: 14px; }
}
"""


def get_training_data():
    try:
        df = pd.read_csv(LOG_PATH)
        if len(df) > 0:
            return df
    except Exception:
        pass
    return pd.DataFrame({"step": [0], "total_reward": [0], "difficulty": [1]})


def generate_episode(tier, session_state):
    session_state = dict(session_state or _new_session_state())
    tier = int(tier)

    local_corruptor = Corruptor()
    local_corruptor.force_tier(tier)
    sample = clean_data.sample(n=min(50, len(clean_data))).reset_index(drop=True)
    dirty, gt, meta = local_corruptor.generate_episode(sample)
    gt = _align_duplicate_ground_truth(dirty, gt, meta)

    session_state.update({"dirty": dirty.copy(), "gt": gt.copy(), "meta": meta, "tier": tier})

    display_cols = [c for c in dirty.columns if c != "_is_deleted"]
    display = dirty[display_cols].head(8).copy()
    _, total_errors = summarize_corruption(dirty[display_cols], HEALTHCARE_SCHEMA)
    acc_before = rc._field_accuracy(dirty, gt)

    stats_html = f"""
    <div class='scenario-grid'>
      {_metric_card("Dataset health", f"{acc_before:.1%}", "Before agent repair", "warn" if acc_before < 0.95 else "good")}
      {_metric_card("Visible issues", str(total_errors), "Schema-level corruption signals", "bad" if total_errors else "good")}
      {_metric_card("Corruption type", meta.get("tool", "unknown"), "Adversarial episode generator", "warn")}
    </div>
    """
    return display, stats_html, session_state


def heuristic_surgeon_agent(state: pd.DataFrame, gt: pd.DataFrame) -> SurgeonAction:
    display_cols = [c for c in state.columns if c != "_is_deleted"]

    for row_idx in range(min(len(state), len(gt))):
        for col_idx, col_name in enumerate(display_cols):
            cell = state.at[row_idx, col_name]
            gt_cell = gt.at[row_idx, col_name]

            if pd.isna(cell) and pd.notna(gt_cell):
                col_type = HEALTHCARE_SCHEMA.get(col_name, {}).get("type", "str")
                tool_id = 0 if col_type in ("int", "float") else 1
                reason = (
                    f"Null in numeric column '{col_name}' - IMPUTE_MEDIAN"
                    if tool_id == 0
                    else f"Missing value in '{col_name}' - IMPUTE_MODE"
                )
                return SurgeonAction(reasoning=reason, tool_id=tool_id, column=col_idx, row_id=row_idx)

            if pd.notna(cell) and pd.notna(gt_cell) and str(cell) != str(gt_cell):
                if str(cell).startswith("ERR_"):
                    col_type = HEALTHCARE_SCHEMA.get(col_name, {}).get("type", "str")
                    tool_id = 0 if col_type in ("int", "float") else 1
                    return SurgeonAction(
                        reasoning=f"Type error '{cell}' in '{col_name}'",
                        tool_id=tool_id,
                        column=col_idx,
                        row_id=row_idx,
                    )
                return SurgeonAction(
                    reasoning=f"Format or consistency error in '{col_name}'",
                    tool_id=3,
                    column=col_idx,
                    row_id=row_idx,
                )

    if len(state) > len(gt):
        return SurgeonAction(
            reasoning="duplicate row detected - DELETE_ROW",
            tool_id=4,
            column=0,
            row_id=len(state) - 1,
        )

    return SurgeonAction(reasoning="no errors detected", tool_id=7, column=0, row_id=0)


def render_ui_state(rollouts, current_state, gt, acc_before, agent_type):
    acc_after = rc._field_accuracy(current_state, gt)
    success_rate_improvement = (
        ((acc_after - acc_before) / (1.0 - acc_before)) * 100 if acc_before < 1.0 else 0
    )
    accuracy_delta = acc_after - acc_before
    total_reward = sum(rollout.get("reward", 0) for rollout in rollouts)
    provenance_title, provenance_body = _agent_provenance(agent_type)

    rollout_html = f"""
    <div style='font-family: JetBrains Mono, monospace;'>
      <div class='metric-card telemetry-card'>
        <div class='metric-label'>Execution path</div>
        <div class='metric-value compact'>{_escape(provenance_title)}</div>
        <div class='metric-detail'>{_escape(provenance_body)}</div>
      </div>
      <div class='rollout-metrics'>
        {_metric_card("Accuracy before", f"{acc_before:.1%}", "Initial field-level health", "bad")}
        {_metric_card("Accuracy after", f"{acc_after:.1%}", "Current repaired state", "good" if acc_after >= acc_before else "bad")}
        {_metric_card("Accuracy delta", f"{accuracy_delta * 100:+.2f} pp", f"Recovered share: {success_rate_improvement:+.1f}%", "good" if accuracy_delta >= 0 else "bad")}
        {_metric_card("Tool calls", str(len(rollouts)), f"Cumulative reward {total_reward:+.2f}", "neutral")}
      </div>
      <p class='section-label' style='margin:0 0 8px'>Trajectory log</p>
    """

    for idx, rollout in enumerate(rollouts):
        css = "rollout-winner" if rollout.get("reward", 0) >= 0 else "rollout-loser"
        reasoning_text = str(rollout.get("reasoning", ""))
        if len(reasoning_text) > 55:
            reasoning_text = reasoning_text[:55] + "..."
        reasoning_text = _escape(reasoning_text)
        tool_name = _escape(rollout.get("tool_name", "?"))

        rollout_html += f"""
        <div class='rollout-row {css}'>
          <div style='color:#94a3b8; font-weight:700;'>STEP {idx + 1:02d}</div>
          <div style='color:#cbd5e1; flex:1; min-width:180px; font-style:italic;'>{reasoning_text}</div>
          <div class='tag tag-{"null" if rollout.get("is_baseline") else "fixed"}'>{tool_name}</div>
          <div style='color:#fde68a; font-weight:600; white-space:nowrap;'>Reward={rollout.get("reward", 0):+.2f}</div>
        </div>"""

    rollout_html += "</div>"
    repaired_display = current_state[[c for c in current_state.columns if c != "_is_deleted"]].head(8).copy()
    before_html = _score_card("Before", acc_before, "bad")
    after_html = _score_card("After", acc_after, "good" if acc_after >= acc_before else "bad")
    return rollout_html, repaired_display, before_html, after_html


def simulate_agent(agent_type, session_state):
    session_state = dict(session_state or _new_session_state())
    if session_state.get("dirty") is None:
        yield _empty_state_html(), None, _score_card("Before", None), _score_card("After", None), session_state
        return

    dirty = session_state["dirty"].copy()
    gt = session_state["gt"].copy()
    tier = int(session_state.get("tier", 1))
    env, acc_before = _build_rollout_env(dirty, gt, tier)
    display_cols = [c for c in env._state.columns if c != "_is_deleted"]
    rollouts = []

    for _ in range(5):
        if agent_type == "Naive Baseline":
            target_row = None
            target_col = None
            action_tool = 7
            action_reason = "No errors found."

            for row_idx in range(len(env._state)):
                for col_idx, col_name in enumerate(display_cols):
                    cell = env._state.at[row_idx, col_name]
                    if pd.isna(cell):
                        target_row, target_col = row_idx, col_idx
                        action_tool = 0
                        action_reason = "Naive baseline: null found. Imputing median."
                        break
                    if str(cell).startswith("ERR_"):
                        target_row, target_col = row_idx, col_idx
                        action_tool = 0
                        action_reason = "Naive baseline: type error found. Imputing median."
                        break
                if target_row is not None:
                    break

            action = SurgeonAction(
                reasoning=action_reason,
                tool_id=action_tool,
                column=target_col if target_col is not None else 0,
                row_id=target_row if target_row is not None else 0,
            )
        elif agent_type == "Heuristic Surgeon":
            action = heuristic_surgeon_agent(env._state.copy(), gt)
        else:
            success, message = load_llm()
            if not success:
                yield (
                    _unavailable_html(message),
                    None,
                    _score_card("Before", acc_before, "bad"),
                    _score_card("After", None),
                    session_state,
                )
                return

            obs = env._make_observation()
            prompt = build_prompt(obs)
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Observation: {obs.model_dump_json()}\nOutput valid JSON only."},
            ]

            try:
                outputs = _run_llm(messages)
                generated_text = outputs[0]["generated_text"][-1]["content"]
                action = robust_parse_action(generated_text, require_fields=True)
            except Exception as exc:
                logger.exception("LLM inference failed")
                action = SurgeonAction(
                    reasoning=f"LLM parse failure: {str(exc)[:40]}",
                    tool_id=7,
                    column=0,
                    row_id=0,
                )

        _, total_reward, done, _ = env.step(action)
        rollouts.append(
            {
                "reasoning": action.reasoning,
                "tool_name": SURGEON_TOOLS.get(action.tool_id, {"name": "UNKNOWN"})["name"],
                "reward": total_reward,
                "selected": True,
                "is_baseline": agent_type == "Naive Baseline",
            }
        )

        yield (*render_ui_state(rollouts, env._state, gt, acc_before, agent_type), session_state)
        if done:
            break


def build_demo():
    choices = available_agent_choices()
    default_choice = "Live GRPO Model" if "Live GRPO Model" in choices else "Heuristic Surgeon"

    with gr.Blocks(title="DataForge Arena") as demo:
        session_state = gr.State(_new_session_state())

        gr.HTML(_hero_html())
        gr.HTML(_evidence_snapshot_html())

        with gr.Row():
            with gr.Column(scale=1, elem_classes="panel"):
                gr.HTML("<p class='section-label' style='margin:0 0 10px'>1. Corrupted input</p>")
                with gr.Row():
                    btn_easy = gr.Button("Tier 1 Scenario", variant="secondary")
                    btn_hard = gr.Button("Tier 3 Adversarial", variant="secondary")
                dirty_view = gr.Dataframe(label="", interactive=False)
                error_stats = gr.HTML("")

            with gr.Column(scale=2, elem_classes="panel"):
                gr.HTML("<p class='section-label' style='margin:0 0 10px'>2. Agent telemetry</p>")
                mode_inventory = gr.HTML(_telemetry_intro_html())
                agent_choice = gr.Radio(choices, value=default_choice, label="EXECUTION PATH")
                run_btn = gr.Button("EXECUTE AGENT", variant="primary", size="lg")
                rollout_html = gr.HTML(_empty_state_html())

            with gr.Column(scale=1, elem_classes="panel"):
                gr.HTML("<p class='section-label' style='margin:0 0 10px'>3. Repaired output</p>")
                repaired_view = gr.Dataframe(label="", interactive=False)
                with gr.Row():
                    score_before = gr.HTML(_score_card("Before", None))
                    score_after = gr.HTML(_score_card("After", None))

        with gr.Row(elem_classes="panel"):
            with gr.Column():
                gr.HTML("<p class='section-label' style='margin:0 0 10px'>Training evidence</p>")
                with gr.Row():
                    with gr.Column():
                        reward_plot = gr.LinePlot(
                            x="step",
                            y="total_reward",
                            title="Reward Curve",
                            x_title="Step",
                            y_title="Reward",
                            height=220,
                        )
                    with gr.Column():
                        difficulty_plot = gr.LinePlot(
                            x="step",
                            y="difficulty",
                            title="Difficulty Escalation",
                            x_title="Step",
                            y_title="Tier",
                            height=220,
                        )

        def generate_easy(state):
            display, stats_html, next_state = generate_episode(1, state)
            meta = next_state["meta"]
            acc_before = rc._field_accuracy(next_state["dirty"], next_state["gt"])
            full_display = next_state["dirty"][[c for c in next_state["dirty"].columns if c != "_is_deleted"]]
            _, total_errors = summarize_corruption(full_display, HEALTHCARE_SCHEMA)
            return (
                display,
                stats_html,
                next_state,
                _scenario_ready_html(meta, acc_before, total_errors),
                None,
                _score_card("Before", acc_before, "bad"),
                _score_card("After", None),
            )

        def generate_hard(state):
            display, stats_html, next_state = generate_episode(3, state)
            meta = next_state["meta"]
            acc_before = rc._field_accuracy(next_state["dirty"], next_state["gt"])
            full_display = next_state["dirty"][[c for c in next_state["dirty"].columns if c != "_is_deleted"]]
            _, total_errors = summarize_corruption(full_display, HEALTHCARE_SCHEMA)
            return (
                display,
                stats_html,
                next_state,
                _scenario_ready_html(meta, acc_before, total_errors),
                None,
                _score_card("Before", acc_before, "bad"),
                _score_card("After", None),
            )

        def load_curves():
            df = get_training_data()
            return df, df

        scenario_outputs = [
            dirty_view,
            error_stats,
            session_state,
            rollout_html,
            repaired_view,
            score_before,
            score_after,
        ]
        btn_easy.click(fn=generate_easy, inputs=[session_state], outputs=scenario_outputs)
        btn_hard.click(fn=generate_hard, inputs=[session_state], outputs=scenario_outputs)
        run_btn.click(
            fn=simulate_agent,
            inputs=[agent_choice, session_state],
            outputs=[rollout_html, repaired_view, score_before, score_after, session_state],
        )
        demo.load(fn=load_curves, outputs=[reward_plot, difficulty_plot])

    return demo


demo = build_demo()


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.queue(default_concurrency_limit=8).launch(
        server_name=server_name,
        server_port=server_port,
        css=DARK_CSS,
        theme=gr.themes.Base(),
    )
