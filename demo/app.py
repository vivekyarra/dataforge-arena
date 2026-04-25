from __future__ import annotations

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


def local_model_available(model_path: str | None = None) -> bool:
    if model_path is None:
        model_path = LOCAL_MODEL_PATH
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
        return "—"
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
            warnings.filterwarnings("ignore", message=r"The following generation flags are not valid.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=r"Both `max_new_tokens`.*", category=UserWarning)
            return llm_pipeline(
                messages, max_new_tokens=96, temperature=0.1,
                do_sample=False, num_return_sequences=1,
            )


def _agent_provenance(agent_type: str) -> tuple[str, str]:
    if agent_type == "Naive Baseline":
        return "Naive Baseline", "Simple null and ERR_* scan with no learned policy."
    if agent_type == "Heuristic Surgeon":
        return "Heuristic Surgeon", "Deterministic repair policy — committed evaluation baseline."
    return "Live GRPO Checkpoint", "Live inference from trained local checkpoint in outputs/dataforge-surgeon."


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
        "parse_success": None, "parse_first": None, "parse_last": None,
        "parse_recovered_rate": None, "invalid_action_rate": None,
        "dominant_tool": None, "dominant_tool_rate": None,
        "tiers": "—", "last_step": "—", "latest_reward": None, "best_reward": None,
    }
    if df.empty:
        return summary

    for key, col in [("parse_success", "parse_success_rate"), ("parse_first", None), ("parse_last", None)]:
        if col and col in df:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals):
                summary["parse_success"] = float(vals.mean() * 100)
                summary["parse_first"] = float(vals.iloc[0] * 100)
                summary["parse_last"] = float(vals.iloc[-1] * 100)

    if "parse_success_rate" in df:
        vals = pd.to_numeric(df["parse_success_rate"], errors="coerce").dropna()
        if len(vals):
            summary["parse_success"] = float(vals.mean() * 100)
            summary["parse_first"] = float(vals.iloc[0] * 100)
            summary["parse_last"] = float(vals.iloc[-1] * 100)

    if "parse_recovered_rate" in df:
        vals = pd.to_numeric(df["parse_recovered_rate"], errors="coerce").dropna()
        if len(vals): summary["parse_recovered_rate"] = float(vals.iloc[-1] * 100)

    if "invalid_action_rate" in df:
        vals = pd.to_numeric(df["invalid_action_rate"], errors="coerce").dropna()
        if len(vals): summary["invalid_action_rate"] = float(vals.iloc[-1] * 100)

    if "dominant_tool" in df:
        vals = pd.to_numeric(df["dominant_tool"], errors="coerce").dropna()
        if len(vals): summary["dominant_tool"] = int(vals.iloc[-1])

    if "dominant_tool_rate" in df:
        vals = pd.to_numeric(df["dominant_tool_rate"], errors="coerce").dropna()
        if len(vals): summary["dominant_tool_rate"] = float(vals.iloc[-1] * 100)

    if "difficulty" in df:
        tiers = sorted(pd.to_numeric(df["difficulty"], errors="coerce").dropna().astype(int).unique())
        if tiers:
            summary["tiers"] = f"{tiers[0]}–{tiers[-1]}" if len(tiers) > 1 else str(tiers[0])

    if "step" in df:
        steps = pd.to_numeric(df["step"], errors="coerce").dropna()
        if len(steps): summary["last_step"] = str(int(steps.max()))

    if "total_reward" in df:
        rewards = pd.to_numeric(df["total_reward"], errors="coerce").dropna()
        if len(rewards):
            summary["latest_reward"] = float(rewards.iloc[-1])
            summary["best_reward"] = float(rewards.max())

    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  HTML BUILDERS — PREMIUM REDESIGN
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, sub: str = "", tone: str = "neutral") -> str:
    """A clean KPI stat block."""
    tone_map = {
        "good":    ("var(--g)", "var(--g-dim)", "var(--g-border)"),
        "bad":     ("var(--r)", "var(--r-dim)", "var(--r-border)"),
        "warn":    ("var(--a)", "var(--a-dim)", "var(--a-border)"),
        "neutral": ("var(--t2)", "transparent",  "var(--border)"),
    }
    col, bg, bord = tone_map.get(tone, tone_map["neutral"])
    return f"""
    <div class="kpi" style="border-color:{bord}; background:{bg};">
      <div class="kpi-label">{_escape(label)}</div>
      <div class="kpi-value" style="color:{col};">{_escape(value)}</div>
      {f'<div class="kpi-sub">{_escape(sub)}</div>' if sub else ''}
    </div>"""


def _metric_card(label: str, value: str, detail: str, tone: str = "neutral") -> str:
    """Legacy alias — maps to _kpi for backwards compat."""
    return _kpi(label, value, detail, tone)


def _hero_html() -> str:
    return """
    <div class="hero">
      <div class="hero-bg-grid"></div>
      <div class="hero-glow"></div>
      <div class="hero-inner">
        <div class="hero-left">
          <div class="hero-eyebrow">
            <span class="pulse-dot"></span>
            OpenEnv Enterprise Benchmark
          </div>
          <h1 class="hero-title">DataForge<br><span class="hero-accent">Arena</span></h1>
          <p class="hero-body">
            A judge-visible RL arena where data repair agents face adversarial
            tabular corruption, choose grounded tools, and earn reward only when
            the dataset measurably improves.
          </p>
        </div>
        <div class="hero-steps">
          <div class="hero-step">
            <div class="step-num">01</div>
            <div class="step-body">
              <div class="step-title">Corrupt</div>
              <div class="step-desc">Generate adversarial enterprise data failures across 3 difficulty tiers.</div>
            </div>
          </div>
          <div class="hero-step">
            <div class="step-num">02</div>
            <div class="step-body">
              <div class="step-title">Repair</div>
              <div class="step-desc">Run baseline, heuristic, or trained GRPO surgeon agent.</div>
            </div>
          </div>
          <div class="hero-step">
            <div class="step-num">03</div>
            <div class="step-body">
              <div class="step-title">Verify</div>
              <div class="step-desc">Inspect accuracy deltas, reward DNA, and action traces.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """


def _evidence_snapshot_html() -> str:
    results = _read_eval_results()
    baseline = _read_baseline_results()
    training = _training_summary()
    checkpoint_ready = local_model_available()

    grpo_advantage = results.get("surgeon_advantage_accuracy_delta")
    grpo_delta = results.get("surgeon_avg_accuracy_delta")
    random_delta = results.get("random_avg_accuracy_delta")
    grpo_episodes = results.get("episodes", "—")
    heuristic_advantage = baseline.get("surgeon_advantage_accuracy_delta")
    heuristic_episodes = baseline.get("episodes", "—")
    parse_success = training["parse_success"]

    recovery_detail = ""
    if training["parse_recovered_rate"] is not None:
        recovery_detail = f" · recovered {training['parse_recovered_rate']:.1f}%"

    ps_str = f"{parse_success:.1f}%" if parse_success is not None else "—"
    ps_sub = ""
    if training["parse_first"] is not None:
        ps_sub = (
            f"First {training['parse_first']:.1f}% -> last {training['parse_last']:.1f}%"
            f" · tiers {training['tiers']}{recovery_detail}"
        )

    cards = [
        _kpi("Heuristic baseline", _format_pp(heuristic_advantage),
              f"Rule-based surgeon · {heuristic_episodes} episodes",
              _status_tone(heuristic_advantage)),
        _kpi("GRPO checkpoint", _format_pp(grpo_advantage),
              f"vs random {_format_pp(random_delta)} · {grpo_episodes} episodes",
              _status_tone(grpo_advantage)),
        _kpi("Parse Reliability", ps_str, ps_sub or f"Tiers {training['tiers']}",
              "good" if parse_success and parse_success >= 50 else "neutral"),
        _kpi("Live GRPO", "Ready" if checkpoint_ready else "Checkpoint gated",
              "Checkpoint found" if checkpoint_ready else "No checkpoint at outputs/dataforge-surgeon",
              "good" if checkpoint_ready else "neutral"),
    ]
    return f"<div class='evidence-strip'>{''.join(cards)}</div>"


def _telemetry_intro_html():
    if local_model_available():
        msg = "Local GRPO checkpoint detected — live inference available."
        tone = "good"
    else:
        msg = "No local checkpoint found. Baseline and heuristic paths active."
        tone = "neutral"
    col = "var(--g)" if tone == "good" else "var(--t2)"
    return f"""
    <div class="mode-banner" style="border-color:{'var(--g-border)' if tone=='good' else 'var(--border)'};">
      <span class="mode-dot" style="background:{col};"></span>
      <span class="mode-text">{_escape(msg)}</span>
    </div>"""


def _empty_state_html():
    return """
    <div class="empty-state">
      <div class="empty-icon">◎</div>
      <div class="empty-title">Arena Idle</div>
      <div class="empty-desc">Generate a Tier 1 or Tier 3 scenario, then execute an agent to begin.</div>
    </div>"""


def _scenario_ready_html(meta: dict, acc_before: float, total_errors: int) -> str:
    tool = meta.get("tool", "unknown")
    return f"""
    <div class="scenario-armed">
      <div class="scenario-armed-header">
        <span class="pulse-dot"></span>
        <span>Scenario Armed</span>
      </div>
      <div class="scenario-armed-tool">{_escape(tool)}</div>
      <div class="scenario-armed-stats">
        <span>Health <strong>{acc_before:.1%}</strong></span>
        <span>·</span>
        <span><strong>{total_errors}</strong> visible issues</span>
      </div>
    </div>"""


def _score_card(label: str, value: float | None, tone: str = "neutral") -> str:
    display = f"{value:.1%}" if value is not None else "—"
    return _kpi(label, display, "Field-level accuracy", tone)


def _unavailable_html(message: str) -> str:
    return f"""
    <div class="unavail-banner">
      <div class="unavail-icon">✕</div>
      <div class="unavail-title">Model Unavailable</div>
      <div class="unavail-desc">{_escape(message)}</div>
    </div>"""


def _format_cell_value(value) -> str:
    if pd.isna(value):
        return "<span class='null-tok'>NULL</span>"
    return _escape(value)


def _tool_name(tool_id: int | None) -> str:
    return SURGEON_TOOLS.get(tool_id, {"name": "UNKNOWN"})["name"]


def _progress_html(completed_steps: int, total_steps: int, pending_step: int | None = None) -> str:
    total_steps = max(total_steps, 1)
    visual = completed_steps + (0.4 if pending_step is not None else 0.0)
    pct = min(100.0, max(0.0, visual / total_steps * 100))
    active = pending_step if pending_step is not None else max(completed_steps, 1)
    phase = "Running…" if pending_step is not None else "Complete"
    return f"""
    <div class="rollout-progress">
      <div class="rp-row">
        <div>
          <div class="kpi-label">Rollout Progress</div>
          <div class="rp-step">Step {active} / {total_steps}</div>
        </div>
        <div class="rp-phase">{_escape(phase)}</div>
      </div>
      <div class="rp-track"><span class="rp-fill" style="width:{pct:.1f}%"></span></div>
    </div>"""


def _diff_summary_html(original_state, current_state, gt) -> str:
    if original_state is None or current_state is None or gt is None:
        return """
        <div class="diff-placeholder">
          <div class="kpi-label">Change Audit</div>
          <div class="diff-placeholder-text">Run a scenario to inspect fixed cells, regressions, and remaining issues.</div>
        </div>"""

    display_cols = [c for c in current_state.columns if c != "_is_deleted"]
    row_limit = min(len(current_state), len(gt))
    changed_rows, remaining_rows = [], []
    fixed_count = regressed_count = remaining_count = 0

    for ri in range(row_limit):
        for col in display_cols:
            before = original_state.at[ri, col]
            after = current_state.at[ri, col]
            target = gt.at[ri, col]
            before_ok = rc._values_match(before, target)
            after_ok = rc._values_match(after, target)
            if not after_ok:
                remaining_count += 1
                if len(remaining_rows) < 8:
                    remaining_rows.append((ri, col, _format_cell_value(after), _format_cell_value(target)))
            if not rc._values_match(before, after):
                if not before_ok and after_ok:
                    badge, cls = "Fixed", "badge-fixed"
                    fixed_count += 1
                elif before_ok and not after_ok:
                    badge, cls = "Regressed", "badge-regressed"
                    regressed_count += 1
                else:
                    badge, cls = "Shifted", "badge-shifted"
                if len(changed_rows) < 8:
                    changed_rows.append((badge, cls, ri, col,
                                         _format_cell_value(before), _format_cell_value(after), _format_cell_value(target)))

    def _rows(rows, changed):
        if not rows:
            txt = "No edits yet." if changed else "All cells aligned."
            return f"<tr><td colspan='6' class='empty-row'>{txt}</td></tr>"
        out = []
        for row in rows:
            if changed:
                badge, cls, ri, col, before, after, target = row
                out.append(f"<tr><td><span class='dbadge {cls}'>{badge}</span></td>"
                            f"<td class='mono'>r{ri}</td><td class='mono'>{_escape(col)}</td>"
                            f"<td>{before}</td><td>{after}</td><td>{target}</td></tr>")
            else:
                ri, col, after, target = row
                out.append(f"<tr><td class='mono'>r{ri}</td><td class='mono'>{_escape(col)}</td>"
                            f"<td>{after}</td><td>{target}</td></tr>")
        return "".join(out)

    fc_tone = "good" if fixed_count else "neutral"
    rc_tone = "bad" if regressed_count else "good"
    rem_tone = "warn" if remaining_count else "good"

    return f"""
    <div class="diff-root">
      <div class="diff-kpis">
        {_kpi("Cells Fixed", str(fixed_count), "Edits matching ground truth", fc_tone)}
        {_kpi("Regressions", str(regressed_count), "Edits away from ground truth", rc_tone)}
        {_kpi("Remaining", str(remaining_count), "Still unaligned", rem_tone)}
      </div>
      <div class="diff-tables">
        <div class="diff-panel">
          <div class="diff-panel-title">Changed Cells</div>
          <div class="table-scroll">
            <table class="dt">
              <thead><tr><th>Status</th><th>Row</th><th>Column</th><th>Before</th><th>After</th><th>Target</th></tr></thead>
              <tbody>{_rows(changed_rows, True)}</tbody>
            </table>
          </div>
        </div>
        <div class="diff-panel">
          <div class="diff-panel-title">Still Broken</div>
          <div class="table-scroll">
            <table class="dt">
              <thead><tr><th>Row</th><th>Column</th><th>Current</th><th>Target</th></tr></thead>
              <tbody>{_rows(remaining_rows, False)}</tbody>
            </table>
          </div>
        </div>
      </div>
    </div>"""


def _benchmark_race_html() -> str:
    results = _read_eval_results()
    baseline = _read_baseline_results()
    training = _training_summary()
    lanes = [
        ("Heuristic Surgeon", baseline.get("surgeon_advantage_accuracy_delta"), "Committed baseline over random"),
        ("GRPO Checkpoint",   results.get("surgeon_advantage_accuracy_delta"),  "Trained checkpoint over random"),
    ]
    scale = max([abs(v) for _, v, _ in lanes if v is not None] + [0.01])

    html_lanes = []
    for label, value, detail in lanes:
        if value is None:
            pct, vtxt, tone = 20, "Pending", "neutral"
        else:
            pct = 20 + abs(value) / scale * 80
            vtxt = _format_pp(value)
            tone = "good" if value >= 0 else "bad"
        html_lanes.append(f"""
        <div class="race-lane">
          <div class="race-meta">
            <span class="race-name">{_escape(label)}</span>
            <span class="race-val race-{tone}">{_escape(vtxt)}</span>
          </div>
          <div class="race-track"><span class="race-bar race-bar-{tone}" style="width:{pct:.1f}%"></span></div>
          <div class="race-detail">{_escape(detail)}</div>
        </div>""")

    lr = training["latest_reward"]
    lr_txt = f"{lr:+.2f}" if lr is not None else "—"
    ia_txt = f"{training['invalid_action_rate']:.1f}%" if training["invalid_action_rate"] is not None else "—"
    return f"""
    <div class="benchmark-root">
      <div class="bench-header">Benchmark Race</div>
      {''.join(html_lanes)}
      <div class="bench-foot">
        <span>Latest reward <strong>{lr_txt}</strong></span>
        <span>·</span>
        <span>Invalid actions <strong>{ia_txt}</strong></span>
      </div>
    </div>"""


def _architecture_html() -> str:
    items = [
        ("Observe", "↗", "Schema, suspect rows, and recent actions packed into a structured prompt."),
        ("Act",     "⚡", "The surgeon emits one constrained JSON repair action per step."),
        ("Score",   "◈", "Reward grounded in accuracy delta, tool logic, efficiency, anti-shortcut."),
        ("Escalate","↑", "Corruptor advances from simple nulls to hard relational failures."),
    ]
    cards = []
    for title, icon, desc in items:
        cards.append(f"""
        <div class="arch-card">
          <div class="arch-icon">{icon}</div>
          <div class="arch-title">{_escape(title)}</div>
          <div class="arch-desc">{_escape(desc)}</div>
        </div>""")
    return f"<div class='arch-grid'>{''.join(cards)}</div>"


def _reward_dna_html(components: dict) -> str:
    if not components:
        return ""
    labels = {
        "accuracy_delta":     "Accuracy Δ",
        "constraint_alignment": "Constraint",
        "schema_alignment":   "Schema",
        "outlier_targeting":  "Outlier",
        "reasoning_quality":  "Reasoning",
        "parse_bonus":        "Parse",
        "anti_hack":          "Anti-Hack",
    }
    scale = max([abs(float(v)) for v in components.values() if isinstance(v, (int, float))] + [0.01])
    rows = []
    for key, label in labels.items():
        val = float(components.get(key, 0.0))
        pct = min(100.0, abs(val) / scale * 100.0)
        pos = val >= 0
        rows.append(f"""
        <div class="dna-row">
          <div class="dna-label">{_escape(label)}</div>
          <div class="dna-track"><span class="dna-bar {'dna-pos' if pos else 'dna-neg'}" style="width:{pct:.1f}%"></span></div>
          <div class="dna-val {'dna-val-pos' if pos else 'dna-val-neg'}">{val:+.2f}</div>
        </div>""")
    return f"""
    <div class="reward-dna">
      <div class="dna-header">Reward DNA — Last Step</div>
      {''.join(rows)}
    </div>"""


def _tier_badge_html(tier: int) -> str:
    labels = {1: "Tier 1 · Simple Nulls", 2: "Tier 2 · Cluster Corruption", 3: "Tier 3 · Relational Failures"}
    colors = {1: "var(--g)", 2: "var(--a)", 3: "var(--r)"}
    col = colors.get(tier, "var(--t2)")
    label = labels.get(tier, f"Tier {tier}")
    return f"""
    <div class="tier-badge" style="color:{col}; border-color:{col}33;">
      <span class="tier-dot" style="background:{col};"></span>
      {_escape(label)}
    </div>"""


def get_training_data():
    try:
        df = pd.read_csv(LOG_PATH)
        if len(df) > 0:
            return df
    except Exception:
        pass
    return pd.DataFrame({"step": [0], "total_reward": [0], "difficulty": [1]})


# ─────────────────────────────────────────────────────────────────────────────
#  CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────

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

    tool_short = meta.get("tool", "unknown")
    stats_html = f"""
    <div class="scenario-kpis">
      {_kpi("Dataset Health", f"{acc_before:.1%}", "Before repair", "warn" if acc_before < 0.95 else "good")}
      {_kpi("Visible Issues", str(total_errors), "Schema-level signals", "bad" if total_errors else "good")}
      {_kpi("Corruption Type", tool_short, "Adversarial generator", "warn")}
    </div>
    {_tier_badge_html(tier)}
    """
    return display, stats_html, session_state


def heuristic_surgeon_agent(state: pd.DataFrame, gt: pd.DataFrame) -> SurgeonAction:
    display_cols = [c for c in state.columns if c != "_is_deleted"]
    for ri in range(min(len(state), len(gt))):
        for ci, col in enumerate(display_cols):
            cell = state.at[ri, col]
            gt_cell = gt.at[ri, col]
            if pd.isna(cell) and pd.notna(gt_cell):
                col_type = HEALTHCARE_SCHEMA.get(col, {}).get("type", "str")
                tool_id = 0 if col_type in ("int", "float") else 1
                reason = (f"Null in numeric '{col}' — IMPUTE_MEDIAN" if tool_id == 0
                           else f"Missing value in '{col}' — IMPUTE_MODE")
                return SurgeonAction(reasoning=reason, tool_id=tool_id, column=ci, row_id=ri)
            if pd.notna(cell) and pd.notna(gt_cell) and str(cell) != str(gt_cell):
                if str(cell).startswith("ERR_"):
                    col_type = HEALTHCARE_SCHEMA.get(col, {}).get("type", "str")
                    tool_id = 0 if col_type in ("int", "float") else 1
                    return SurgeonAction(reasoning=f"Type error '{cell}' in '{col}'", tool_id=tool_id, column=ci, row_id=ri)
                return SurgeonAction(reasoning=f"Format/consistency error in '{col}'", tool_id=3, column=ci, row_id=ri)
    if len(state) > len(gt):
        return SurgeonAction(reasoning="Duplicate row detected — DELETE_ROW", tool_id=4, column=0, row_id=len(state) - 1)
    return SurgeonAction(reasoning="No errors detected", tool_id=7, column=0, row_id=0)


def render_ui_state(rollouts, original_state, current_state, gt, acc_before, agent_type,
                    total_steps: int = 5, pending_step: int | None = None):
    acc_after = rc._field_accuracy(current_state, gt)
    delta = acc_after - acc_before
    recovery = ((acc_after - acc_before) / (1.0 - acc_before)) * 100 if acc_before < 1.0 else 0
    total_reward = sum(r.get("reward", 0) for r in rollouts)
    provenance_title, provenance_body = _agent_provenance(agent_type)
    diff_html = _diff_summary_html(original_state, current_state, gt)

    last_reasoning = last_violation = last_tool = ""
    if rollouts:
        last = rollouts[-1]
        last_reasoning = last.get("reasoning", "")
        last_violation = last.get("violation_type", "")
        last_tool = last.get("tool_name", "")

    violation_colors = {
        "null_numeric": "#ff4d6a", "null_categorical": "#ff4d6a",
        "range": "#ffb020", "type_error": "#ffb020", "enum_violation": "#3b9eff",
        "fk_mismatch": "#ff4d6a", "semantic_temporal_drift": "#ffb020",
        "currency_unit_mismatch": "#ffb020", "clean": "#00e87a",
    }
    vc = violation_colors.get(last_violation, "rgba(255,255,255,0.5)")

    causal_html = ""
    if last_reasoning:
        causal_html = f"""
        <div class="causal-block">
          <div class="causal-header">Agent Reasoning — Last Action</div>
          <div class="causal-quote">"{_escape(last_reasoning)}"</div>
          <div class="causal-tags">
            <span class="ctag ctag-dim">Violation</span>
            <span class="ctag" style="color:{vc}; border-color:{vc}33;">{_escape(last_violation) if last_violation else "none"}</span>
            <span class="ctag ctag-dim" style="margin-left:8px;">Tool</span>
            <span class="ctag ctag-green">{_escape(last_tool)}</span>
          </div>
        </div>"""

    # Build trajectory rows
    traj_rows = ""
    for idx, rollout in enumerate(rollouts):
        rew = rollout.get("reward", 0)
        is_win = rew >= 0
        reasoning = str(rollout.get("reasoning", ""))
        if len(reasoning) > 60:
            reasoning = reasoning[:60] + "…"
        tool_name = rollout.get("tool_name", "?")
        row_id = rollout.get("row_id", "?")
        col_name = rollout.get("column_name", "?")
        hint = rollout.get("target_cell_hint", "")
        loc = f"r{row_id} / {col_name}" + (f" → {hint[:40]}" if hint else "")
        comps = rollout.get("components", {})
        acc_c = comps.get("accuracy_delta", 0.0)
        eff_c = comps.get("efficiency", 0.0)
        vtype = rollout.get("violation_type", "")

        traj_rows += f"""
        <div class="traj-row {'traj-win' if is_win else 'traj-loss'}">
          <div class="traj-step">{'✓' if is_win else '✕'} {idx+1:02d}</div>
          <div class="traj-reasoning">{_escape(reasoning)}</div>
          <div class="traj-tags">
            <span class="ttag">{_escape(tool_name)}</span>
            {f'<span class="ttag ttag-viol">{_escape(vtype)}</span>' if vtype else ''}
          </div>
          <div class="traj-loc mono">{_escape(loc)}</div>
          <div class="traj-reward {'rew-pos' if is_win else 'rew-neg'}">{rew:+.2f}</div>
        </div>"""

    rollout_html = f"""
    <div class="rollout-root">
      {causal_html}
      {_progress_html(len(rollouts), total_steps, pending_step)}
      <div class="agent-pill">
        <span class="agent-pill-dot"></span>
        <span class="agent-pill-name">{_escape(provenance_title)}</span>
        <span class="agent-pill-desc">{_escape(provenance_body)}</span>
      </div>
      <div class="rollout-kpis">
        {_kpi("Before", f"{acc_before:.1%}", "Initial accuracy", "neutral")}
        {_kpi("After",  f"{acc_after:.1%}",  "Repaired accuracy", "good" if acc_after >= acc_before else "bad")}
        {_kpi("Delta",  f"{delta*100:+.2f} pp", f"Recovery {recovery:+.1f}%", "good" if delta >= 0 else "bad")}
        {_kpi("Reward", f"{total_reward:+.2f}", f"{len(rollouts)} tool calls", "good" if total_reward >= 0 else "bad")}
      </div>
      <div class="traj-header">Trajectory Log</div>
      <div class="traj-list">
        {traj_rows if traj_rows else '<div class="empty-traj">No steps yet.</div>'}
      </div>
      {_reward_dna_html(rollouts[-1].get("components", {})) if rollouts else ''}
    </div>"""

    repaired_display = current_state[[c for c in current_state.columns if c != "_is_deleted"]].head(8).copy()
    before_html = _score_card("Before", acc_before, "neutral")
    after_html = _score_card("After", acc_after, "good" if acc_after >= acc_before else "bad")
    return rollout_html, repaired_display, before_html, after_html, diff_html


def simulate_agent(agent_type, session_state):
    session_state = dict(session_state or _new_session_state())
    if session_state.get("dirty") is None:
        yield (_empty_state_html(), None, _score_card("Before", None), _score_card("After", None),
               _diff_summary_html(None, None, None), session_state)
        return

    dirty = session_state["dirty"].copy()
    gt = session_state["gt"].copy()
    tier = int(session_state.get("tier", 1))
    env, acc_before = _build_rollout_env(dirty, gt, tier)
    display_cols = [c for c in env._state.columns if c != "_is_deleted"]
    rollouts = []
    max_rollout_steps = 5

    for step_idx in range(max_rollout_steps):
        yield (*render_ui_state(rollouts, dirty, env._state, gt, acc_before, agent_type,
                                total_steps=max_rollout_steps, pending_step=step_idx + 1), session_state)

        if agent_type == "Naive Baseline":
            target_row = target_col = None
            action_tool = 7
            action_reason = "No errors found."
            for ri in range(len(env._state)):
                for ci, col in enumerate(display_cols):
                    cell = env._state.at[ri, col]
                    if pd.isna(cell):
                        target_row, target_col, action_tool = ri, ci, 0
                        action_reason = "Naive baseline: null → IMPUTE_MEDIAN"
                        break
                    if str(cell).startswith("ERR_"):
                        target_row, target_col, action_tool = ri, ci, 0
                        action_reason = "Naive baseline: type error → IMPUTE_MEDIAN"
                        break
                if target_row is not None:
                    break
            action = SurgeonAction(reasoning=action_reason, tool_id=action_tool,
                                   column=target_col if target_col is not None else 0,
                                   row_id=target_row if target_row is not None else 0)

        elif agent_type == "Heuristic Surgeon":
            action = heuristic_surgeon_agent(env._state.copy(), gt)

        else:
            success, message = load_llm()
            if not success:
                yield (_unavailable_html(message), None,
                       _score_card("Before", acc_before, "neutral"), _score_card("After", None),
                       _diff_summary_html(dirty, env._state, gt), session_state)
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
                action = SurgeonAction(reasoning=f"LLM parse failure: {str(exc)[:40]}", tool_id=7, column=0, row_id=0)

        _, total_reward, done, info = env.step(action)
        obs_after = env._make_observation()
        rollouts.append({
            "reasoning":        action.reasoning.replace("EXACT_PARSE: ", ""),
            "tool_name":        SURGEON_TOOLS.get(action.tool_id, {"name": "UNKNOWN"})["name"],
            "reward":           total_reward,
            "selected":         True,
            "is_baseline":      agent_type == "Naive Baseline",
            "row_id":           action.row_id,
            "column_name":      display_cols[action.column] if action.column < len(display_cols) else "?",
            "components":       info.get("reward_components", {}),
            "violation_type":   getattr(obs_after, "violation_type", ""),
            "target_cell_hint": getattr(obs_after, "target_cell_hint", ""),
        })

        yield (*render_ui_state(rollouts, dirty, env._state, gt, acc_before, agent_type,
                                total_steps=max_rollout_steps), session_state)
        if done:
            break


# ─────────────────────────────────────────────────────────────────────────────
#  PREMIUM CSS — OBSIDIAN TERMINAL
# ─────────────────────────────────────────────────────────────────────────────

PREMIUM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  /* Palette */
  --bg:        #030508;
  --s1:        rgba(255,255,255,0.025);
  --s2:        rgba(255,255,255,0.055);
  --s3:        rgba(255,255,255,0.09);
  --border:    rgba(255,255,255,0.07);
  --border-hi: rgba(255,255,255,0.13);

  --t1:  #ffffff;
  --t2:  rgba(255,255,255,0.55);
  --t3:  rgba(255,255,255,0.30);

  --g:        #00e87a;
  --g-dim:    rgba(0,232,122,0.08);
  --g-border: rgba(0,232,122,0.22);

  --b:        #3b9eff;
  --b-dim:    rgba(59,158,255,0.08);
  --b-border: rgba(59,158,255,0.22);

  --r:        #ff4d6a;
  --r-dim:    rgba(255,77,106,0.08);
  --r-border: rgba(255,77,106,0.22);

  --a:        #ffb020;
  --a-dim:    rgba(255,176,32,0.08);
  --a-border: rgba(255,176,32,0.22);

  --mono:   'DM Mono', monospace;
  --sans:   'DM Sans', sans-serif;
  --display:'Syne', sans-serif;

  --r4:  4px;
  --r8:  8px;
  --r12: 12px;
  --r16: 16px;
}

/* ── Reset & Base ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
  background: var(--bg) !important;
  color: var(--t1);
  font-family: var(--sans);
  font-size: 14px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.gradio-container { max-width: 1600px !important; padding: 0 24px 40px !important; }

/* ── Gradio overrides ──────────────────────────────────────── */
.gradio-container .panel, .gradio-container .block,
.gradio-container .wrap, .gradio-container fieldset,
.gradio-container .form { background: transparent !important; border: none !important; box-shadow: none !important; }
.gradio-container .gap { gap: 16px !important; }
.gradio-container label { color: var(--t2) !important; font-family: var(--sans) !important; font-size: 11px !important; letter-spacing: .05em; text-transform: uppercase; }
.gradio-container input, .gradio-container select, .gradio-container textarea { background: var(--s2) !important; border: 1px solid var(--border) !important; color: var(--t1) !important; border-radius: var(--r8) !important; }

/* Buttons */
.gradio-container button {
  font-family: var(--mono) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  border-radius: var(--r8) !important;
  transition: all .18s ease !important;
  letter-spacing: .04em;
}
.gradio-container button.primary, .gradio-container button[variant='primary'] {
  background: linear-gradient(135deg, #00c96a 0%, #0070f3 100%) !important;
  border: 0 !important;
  color: #fff !important;
  box-shadow: 0 0 28px rgba(0,232,122,0.25), 0 4px 16px rgba(0,0,0,0.4) !important;
  padding: 14px 24px !important;
  font-size: 13px !important;
  letter-spacing: .08em !important;
  text-transform: uppercase;
}
.gradio-container button.primary:hover, .gradio-container button[variant='primary']:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 0 40px rgba(0,232,122,0.35), 0 8px 24px rgba(0,0,0,0.5) !important;
}
.gradio-container button.secondary, .gradio-container button[variant='secondary'] {
  background: var(--s2) !important;
  border: 1px solid var(--border-hi) !important;
  color: var(--t1) !important;
  padding: 12px 20px !important;
}
.gradio-container button.secondary:hover { background: var(--s3) !important; border-color: var(--g-border) !important; }

/* Radio */
.gradio-container .radio-group { gap: 8px !important; }
.gradio-container .radio-group label {
  background: var(--s1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r8) !important;
  padding: 10px 16px !important;
  color: var(--t2) !important;
  font-size: 12px !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  cursor: pointer;
  transition: all .15s;
}
.gradio-container .radio-group label:has(input:checked) {
  background: var(--g-dim) !important;
  border-color: var(--g-border) !important;
  color: var(--g) !important;
}

/* Dataframe */
.gradio-container table, .gradio-container .table-wrap {
  font-family: var(--mono) !important;
  font-size: 12px !important;
  background: transparent !important;
}
.gradio-container th {
  background: var(--s2) !important;
  color: var(--t3) !important;
  font-size: 10px !important;
  letter-spacing: .06em;
  text-transform: uppercase;
  padding: 10px 12px !important;
  border-bottom: 1px solid var(--border) !important;
}
.gradio-container td {
  color: var(--t2) !important;
  padding: 9px 12px !important;
  border-bottom: 1px solid rgba(255,255,255,0.04) !important;
}
.gradio-container tr:hover td { background: var(--s1) !important; }

/* LinePlot */
.gradio-container .plot-container { background: var(--s1) !important; border-radius: var(--r12) !important; border: 1px solid var(--border) !important; }

/* ── Section panels ─────────────────────────────────────────── */
.df-panel {
  background: var(--s1);
  border: 1px solid var(--border);
  border-radius: var(--r16);
  padding: 24px;
  position: relative;
  overflow: hidden;
}
.df-panel::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, transparent 60%);
}
.section-label {
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--t3);
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-label::before { content: ''; display: block; width: 16px; height: 1px; background: var(--border-hi); }

/* ── Hero ───────────────────────────────────────────────────── */
.hero {
  position: relative;
  padding: 48px 40px;
  margin: 8px 0 20px;
  border: 1px solid var(--border-hi);
  border-radius: var(--r16);
  overflow: hidden;
  background: linear-gradient(135deg, rgba(0,232,122,0.05) 0%, rgba(0,0,0,0) 40%,
              rgba(59,158,255,0.04) 100%);
}
.hero-bg-grid {
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
  background-size: 48px 48px;
  mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 0%, transparent 100%);
}
.hero-glow {
  position: absolute;
  width: 600px; height: 300px;
  background: radial-gradient(ellipse, rgba(0,232,122,0.12) 0%, transparent 70%);
  top: -80px; left: -100px;
  pointer-events: none;
}
.hero-inner {
  position: relative;
  display: grid;
  grid-template-columns: 1fr 380px;
  gap: 48px;
  align-items: center;
}
.hero-eyebrow {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--g);
  margin-bottom: 16px;
}
.hero-title {
  font-family: var(--display);
  font-size: 64px;
  font-weight: 800;
  line-height: 1.0;
  letter-spacing: -.02em;
  color: var(--t1);
  margin-bottom: 20px;
}
.hero-accent { color: var(--g); }
.hero-body { color: var(--t2); font-size: 16px; line-height: 1.7; max-width: 520px; }
.hero-steps { display: flex; flex-direction: column; gap: 12px; }
.hero-step {
  display: flex; gap: 16px; align-items: flex-start;
  padding: 16px 20px;
  background: var(--s2);
  border: 1px solid var(--border);
  border-radius: var(--r12);
  transition: border-color .2s;
}
.hero-step:hover { border-color: var(--g-border); }
.step-num {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  color: var(--g);
  letter-spacing: .06em;
  padding-top: 2px;
  min-width: 20px;
}
.step-title { font-weight: 600; font-size: 14px; color: var(--t1); margin-bottom: 3px; }
.step-desc { font-size: 12px; color: var(--t3); line-height: 1.5; }

/* Pulse dot */
.pulse-dot {
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--g);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,232,122,0.4); }
  50% { opacity: 0.8; box-shadow: 0 0 0 5px rgba(0,232,122,0); }
}

/* ── Evidence Strip ─────────────────────────────────────────── */
.evidence-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 20px;
}

/* ── KPI Cards ──────────────────────────────────────────────── */
.kpi {
  padding: 18px 20px;
  background: var(--s1);
  border: 1px solid var(--border);
  border-radius: var(--r12);
  display: flex;
  flex-direction: column;
  gap: 6px;
  overflow: hidden;
  transition: border-color .2s, background .2s;
}
.kpi:hover { background: var(--s2); }
.kpi-label {
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--t3);
}
.kpi-value {
  font-family: var(--mono);
  font-size: 28px;
  font-weight: 500;
  line-height: 1.1;
  color: var(--t1);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.kpi-sub {
  font-size: 11px;
  color: var(--t3);
  line-height: 1.4;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ── Mode Banner ────────────────────────────────────────────── */
.mode-banner {
  display: flex; align-items: center; gap: 10px;
  padding: 12px 16px;
  background: var(--s1);
  border: 1px solid var(--border);
  border-radius: var(--r8);
  margin-bottom: 14px;
}
.mode-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.mode-text { font-size: 12px; color: var(--t2); }

/* ── Scenario Cards ─────────────────────────────────────────── */
.scenario-kpis {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}
.scenario-kpis .kpi-value {
  font-size: 20px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.scenario-armed {
  padding: 14px 16px;
  background: var(--a-dim);
  border: 1px solid var(--a-border);
  border-radius: var(--r8);
  margin-bottom: 12px;
}
.scenario-armed-header {
  display: flex; align-items: center; gap: 6px;
  font-family: var(--mono); font-size: 10px;
  letter-spacing: .1em; text-transform: uppercase; color: var(--a);
  margin-bottom: 6px;
}
.scenario-armed-tool {
  font-family: var(--mono); font-size: 16px; font-weight: 500;
  color: var(--t1);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  margin-bottom: 4px;
}
.scenario-armed-stats { font-size: 12px; color: var(--t3); }
.scenario-armed-stats strong { color: var(--t2); }

/* ── Tier Badge ─────────────────────────────────────────────── */
.tier-badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 14px;
  border: 1px solid;
  border-radius: 999px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: .06em;
  margin-top: 10px;
}
.tier-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }

/* ── Rollout / Telemetry ────────────────────────────────────── */
.rollout-root { display: flex; flex-direction: column; gap: 12px; }

.rollout-progress {
  padding: 16px 18px;
  background: var(--s2);
  border: 1px solid var(--border);
  border-radius: var(--r12);
}
.rp-row { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 10px; }
.rp-step { font-family: var(--mono); font-size: 22px; font-weight: 500; color: var(--t1); margin-top: 4px; }
.rp-phase { font-family: var(--mono); font-size: 11px; color: var(--t3); letter-spacing: .06em; }
.rp-track { height: 4px; background: var(--s3); border-radius: 999px; overflow: hidden; }
.rp-fill {
  display: block; height: 100%; border-radius: inherit;
  background: linear-gradient(90deg, var(--g), var(--b));
  transition: width .4s cubic-bezier(.4,0,.2,1);
}

.agent-pill {
  display: flex; align-items: center; gap: 10px;
  padding: 12px 16px;
  background: var(--s1);
  border: 1px solid var(--border);
  border-radius: var(--r8);
  overflow: hidden;
}
.agent-pill-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--g); flex-shrink: 0;
  animation: pulse 2s ease-in-out infinite;
}
.agent-pill-name { font-family: var(--mono); font-size: 13px; font-weight: 500; color: var(--t1); flex-shrink: 0; }
.agent-pill-desc { font-size: 11px; color: var(--t3); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.rollout-kpis {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}
.rollout-kpis .kpi-value { font-size: 22px; }

/* ── Causal Reasoning ───────────────────────────────────────── */
.causal-block {
  padding: 16px 18px;
  background: rgba(59,158,255,0.05);
  border: 1px solid rgba(59,158,255,0.18);
  border-radius: var(--r12);
  animation: fadeUp .25s ease both;
}
.causal-header {
  font-family: var(--mono); font-size: 10px;
  letter-spacing: .1em; text-transform: uppercase;
  color: var(--b); margin-bottom: 8px;
}
.causal-quote {
  font-family: var(--mono); font-size: 13px; font-weight: 500;
  color: var(--t1); margin-bottom: 10px;
  line-height: 1.5;
}
.causal-tags { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }
.ctag {
  display: inline-flex; align-items: center;
  padding: 3px 10px; border: 1px solid var(--border);
  border-radius: 999px; font-family: var(--mono);
  font-size: 11px; font-weight: 500; color: var(--t2);
}
.ctag-dim { color: var(--t3); }
.ctag-green { color: var(--g); border-color: var(--g-border); background: var(--g-dim); }

/* ── Trajectory ─────────────────────────────────────────────── */
.traj-header {
  font-family: var(--mono); font-size: 10px;
  letter-spacing: .1em; text-transform: uppercase;
  color: var(--t3); padding: 0 2px;
}
.traj-list { display: flex; flex-direction: column; gap: 6px; }
.traj-row {
  display: grid;
  grid-template-columns: 52px 1fr auto auto 70px;
  align-items: center;
  gap: 12px;
  padding: 11px 14px;
  border-radius: var(--r8);
  border-left: 3px solid transparent;
  animation: fadeUp .2s ease both;
  overflow: hidden;
}
.traj-win { background: rgba(0,232,122,0.06); border-color: var(--g); }
.traj-loss { background: rgba(255,77,106,0.06); border-color: var(--r); }
.traj-step { font-family: var(--mono); font-size: 11px; font-weight: 500; color: var(--t3); white-space: nowrap; }
.traj-reasoning { font-size: 12px; color: var(--t2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-style: italic; }
.traj-tags { display: flex; gap: 5px; flex-wrap: wrap; justify-content: flex-end; }
.ttag {
  padding: 2px 8px; border-radius: var(--r4);
  font-family: var(--mono); font-size: 10px; font-weight: 500;
  background: var(--s3); color: var(--t2); white-space: nowrap;
}
.ttag-viol { background: var(--r-dim); color: var(--r); }
.traj-loc { font-size: 11px; color: var(--t3); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.traj-reward { font-family: var(--mono); font-size: 13px; font-weight: 500; text-align: right; white-space: nowrap; }
.rew-pos { color: var(--g); }
.rew-neg { color: var(--r); }
.empty-traj { padding: 20px; text-align: center; color: var(--t3); font-size: 12px; }

/* ── Reward DNA ─────────────────────────────────────────────── */
.reward-dna {
  padding: 16px 18px;
  background: var(--s1);
  border: 1px solid var(--border);
  border-radius: var(--r12);
}
.dna-header { font-family: var(--mono); font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: var(--t3); margin-bottom: 12px; }
.dna-row { display: grid; grid-template-columns: 90px 1fr 48px; align-items: center; gap: 10px; margin: 5px 0; }
.dna-label { font-family: var(--mono); font-size: 11px; color: var(--t3); }
.dna-track { height: 5px; background: var(--s3); border-radius: 999px; overflow: hidden; }
.dna-bar { display: block; height: 100%; border-radius: inherit; transition: width .4s ease; }
.dna-pos { background: linear-gradient(90deg, var(--g), var(--b)); }
.dna-neg { background: linear-gradient(90deg, var(--r), var(--a)); }
.dna-val { font-family: var(--mono); font-size: 11px; font-weight: 500; text-align: right; }
.dna-val-pos { color: var(--g); }
.dna-val-neg { color: var(--r); }

/* ── Diff / Change Audit ────────────────────────────────────── */
.diff-root { display: flex; flex-direction: column; gap: 12px; }
.diff-kpis { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.diff-kpis .kpi-value { font-size: 24px; }
.diff-tables { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.diff-panel {
  background: var(--s1);
  border: 1px solid var(--border);
  border-radius: var(--r12);
  overflow: hidden;
}
.diff-panel-title {
  padding: 10px 14px;
  font-family: var(--mono); font-size: 10px;
  letter-spacing: .1em; text-transform: uppercase;
  color: var(--t3);
  background: var(--s2);
  border-bottom: 1px solid var(--border);
}
.table-scroll { overflow-x: auto; }
.dt { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 11px; }
.dt th {
  padding: 8px 12px;
  background: var(--s2);
  color: var(--t3);
  font-size: 9px;
  letter-spacing: .08em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.dt td {
  padding: 7px 12px;
  color: var(--t2);
  border-bottom: 1px solid rgba(255,255,255,0.03);
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.dt tr:last-child td { border-bottom: none; }
.empty-row { text-align: center; color: var(--t3) !important; padding: 16px !important; }
.dbadge {
  display: inline-flex; align-items: center;
  padding: 2px 8px; border-radius: 999px;
  font-size: 9px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
}
.badge-fixed    { background: var(--g-dim); color: var(--g); }
.badge-regressed{ background: var(--r-dim); color: var(--r); }
.badge-shifted  { background: var(--b-dim); color: var(--b); }
.null-tok       { color: var(--a); font-weight: 600; }
.diff-placeholder {
  padding: 24px;
  background: var(--s1);
  border: 1px solid var(--border);
  border-radius: var(--r12);
}
.diff-placeholder-text { font-size: 12px; color: var(--t3); margin-top: 8px; }

/* ── Benchmark Race ─────────────────────────────────────────── */
.benchmark-root {
  padding: 20px 22px;
  background: var(--s1);
  border: 1px solid var(--border);
  border-radius: var(--r12);
}
.bench-header { font-family: var(--mono); font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--t3); margin-bottom: 16px; }
.race-lane { margin-bottom: 16px; }
.race-lane:last-of-type { margin-bottom: 0; }
.race-meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.race-name { font-family: var(--mono); font-size: 13px; font-weight: 500; color: var(--t1); }
.race-val { font-family: var(--mono); font-size: 13px; font-weight: 500; }
.race-good { color: var(--g); }
.race-bad  { color: var(--r); }
.race-neutral { color: var(--t3); }
.race-track { height: 8px; background: var(--s3); border-radius: 999px; overflow: hidden; margin-bottom: 4px; }
.race-bar { display: block; height: 100%; border-radius: inherit; }
.race-bar-good { background: linear-gradient(90deg, var(--g), var(--b)); }
.race-bar-bad  { background: linear-gradient(90deg, var(--r), var(--a)); }
.race-bar-neutral { background: var(--s3); }
.race-detail { font-size: 11px; color: var(--t3); }
.bench-foot { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); font-size: 12px; color: var(--t3); }
.bench-foot strong { color: var(--t2); }

/* ── Architecture Cards ─────────────────────────────────────── */
.arch-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.arch-card {
  padding: 20px;
  background: var(--s1);
  border: 1px solid var(--border);
  border-radius: var(--r12);
  transition: border-color .2s, background .2s;
}
.arch-card:hover { background: var(--s2); border-color: var(--border-hi); }
.arch-icon { font-size: 20px; margin-bottom: 10px; }
.arch-title { font-family: var(--mono); font-size: 13px; font-weight: 500; color: var(--t1); margin-bottom: 6px; }
.arch-desc { font-size: 12px; color: var(--t3); line-height: 1.5; }

/* ── Empty / Unavailable States ─────────────────────────────── */
.empty-state {
  padding: 40px;
  text-align: center;
  display: flex; flex-direction: column; align-items: center; gap: 10px;
}
.empty-icon { font-size: 28px; color: var(--t3); }
.empty-title { font-family: var(--mono); font-size: 16px; font-weight: 500; color: var(--t2); }
.empty-desc { font-size: 12px; color: var(--t3); max-width: 280px; line-height: 1.5; }
.unavail-banner {
  padding: 28px;
  background: var(--r-dim);
  border: 1px solid var(--r-border);
  border-radius: var(--r12);
  text-align: center;
}
.unavail-icon { font-size: 24px; color: var(--r); margin-bottom: 8px; }
.unavail-title { font-family: var(--mono); font-size: 15px; font-weight: 500; color: var(--r); margin-bottom: 6px; }
.unavail-desc { font-size: 12px; color: var(--t3); }

/* ── Mono util ──────────────────────────────────────────────── */
.mono { font-family: var(--mono) !important; }

/* ── Animations ─────────────────────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Responsive ─────────────────────────────────────────────── */
@media (max-width: 1200px) {
  .hero-inner { grid-template-columns: 1fr; gap: 32px; }
  .hero-title { font-size: 48px; }
  .evidence-strip { grid-template-columns: repeat(2, 1fr); }
  .rollout-kpis { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 800px) {
  .hero { padding: 32px 24px; }
  .hero-title { font-size: 36px; }
  .evidence-strip, .scenario-kpis, .diff-kpis { grid-template-columns: 1fr 1fr; }
  .diff-tables, .arch-grid { grid-template-columns: 1fr; }
  .traj-row { grid-template-columns: 40px 1fr 60px; }
  .traj-tags, .traj-loc { display: none; }
}
@media (max-width: 560px) {
  .evidence-strip, .scenario-kpis, .diff-kpis, .rollout-kpis { grid-template-columns: 1fr; }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD DEMO
# ─────────────────────────────────────────────────────────────────────────────

def build_demo():
    choices = available_agent_choices()
    default_choice = "Live GRPO Model" if "Live GRPO Model" in choices else "Heuristic Surgeon"

    with gr.Blocks(title="DataForge Arena", css=PREMIUM_CSS, theme=gr.themes.Base()) as demo:
        session_state = gr.State(_new_session_state())

        gr.HTML(_hero_html())
        evidence_snapshot = gr.HTML(_evidence_snapshot_html())

        with gr.Row(equal_height=False):
            # ── Left: Corrupted Input
            with gr.Column(scale=1):
                gr.HTML("<div class='section-label'>Corrupted Input</div>")
                with gr.Row():
                    btn_easy = gr.Button("⬡  Tier 1 Scenario", variant="secondary")
                    btn_hard = gr.Button("⬡  Tier 3 Adversarial", variant="secondary")
                error_stats = gr.HTML("")
                dirty_view = gr.Dataframe(label="", interactive=False, wrap=False)

            # ── Center: Agent Telemetry
            with gr.Column(scale=2):
                gr.HTML("<div class='section-label'>Agent Telemetry</div>")
                mode_inventory = gr.HTML(_telemetry_intro_html())
                agent_choice = gr.Radio(choices, value=default_choice, label="Execution Path")
                run_btn = gr.Button("▶  Execute Agent", variant="primary", size="lg")
                rollout_html = gr.HTML(_empty_state_html())

            # ── Right: Repaired Output
            with gr.Column(scale=1):
                gr.HTML("<div class='section-label'>Repaired Output</div>")
                repaired_view = gr.Dataframe(label="", interactive=False, wrap=False)
                with gr.Row():
                    score_before = gr.HTML(_score_card("Before", None))
                    score_after  = gr.HTML(_score_card("After",  None))
                diff_html = gr.HTML(_diff_summary_html(None, None, None))

        with gr.Row():
            with gr.Column():
                gr.HTML("<div class='section-label'>Training Evidence</div>")
                refresh_btn = gr.Button("↻  Refresh Evidence", variant="secondary")
                with gr.Row():
                    reward_plot = gr.LinePlot(x="step", y="total_reward", title="Reward Curve",
                                              x_title="Step", y_title="Reward", height=220)
                    difficulty_plot = gr.LinePlot(x="step", y="difficulty", title="Difficulty Escalation",
                                                  x_title="Step", y_title="Tier", height=220)

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<div class='section-label'>Benchmark Snapshot</div>")
                benchmark_html = gr.HTML(_benchmark_race_html())
            with gr.Column(scale=1):
                gr.HTML("<div class='section-label'>How It Works</div>")
                architecture_html = gr.HTML(_architecture_html())

        # ── Event handlers
        def generate_easy(state):
            display, stats_html, next_state = generate_episode(1, state)
            meta = next_state["meta"]
            acc_before = rc._field_accuracy(next_state["dirty"], next_state["gt"])
            full_display = next_state["dirty"][[c for c in next_state["dirty"].columns if c != "_is_deleted"]]
            _, total_errors = summarize_corruption(full_display, HEALTHCARE_SCHEMA)
            return (display, stats_html, next_state,
                    _scenario_ready_html(meta, acc_before, total_errors), None,
                    _score_card("Before", acc_before, "neutral"), _score_card("After", None),
                    _diff_summary_html(next_state["dirty"], next_state["dirty"], next_state["gt"]))

        def generate_hard(state):
            display, stats_html, next_state = generate_episode(3, state)
            meta = next_state["meta"]
            acc_before = rc._field_accuracy(next_state["dirty"], next_state["gt"])
            full_display = next_state["dirty"][[c for c in next_state["dirty"].columns if c != "_is_deleted"]]
            _, total_errors = summarize_corruption(full_display, HEALTHCARE_SCHEMA)
            return (display, stats_html, next_state,
                    _scenario_ready_html(meta, acc_before, total_errors), None,
                    _score_card("Before", acc_before, "neutral"), _score_card("After", None),
                    _diff_summary_html(next_state["dirty"], next_state["dirty"], next_state["gt"]))

        def load_dashboard():
            df = get_training_data()
            return _evidence_snapshot_html(), _benchmark_race_html(), df, df

        scenario_outputs = [
            dirty_view, error_stats, session_state,
            rollout_html, repaired_view,
            score_before, score_after, diff_html,
        ]
        btn_easy.click(fn=generate_easy, inputs=[session_state], outputs=scenario_outputs)
        btn_hard.click(fn=generate_hard, inputs=[session_state], outputs=scenario_outputs)
        run_btn.click(
            fn=simulate_agent, inputs=[agent_choice, session_state],
            outputs=[rollout_html, repaired_view, score_before, score_after, diff_html, session_state],
        )
        refresh_btn.click(fn=load_dashboard, outputs=[evidence_snapshot, benchmark_html, reward_plot, difficulty_plot])
        demo.load(fn=load_dashboard, outputs=[evidence_snapshot, benchmark_html, reward_plot, difficulty_plot])

    return demo


demo = build_demo()

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.queue(default_concurrency_limit=8).launch(
        server_name=server_name,
        server_port=server_port,
    )