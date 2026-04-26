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
from eval.evaluate import _resolve_loadable_model_path, load_eval_pipeline
from training.parser import robust_parse_action
from training.prompt import build_prompt


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "healthcare_clean.csv")
LOG_PATH = os.path.join(ROOT_DIR, "logs", "training_log.csv")
EVAL_PATH = os.path.join(ROOT_DIR, "eval", "results.json")
HEUR_PATH = os.path.join(ROOT_DIR, "eval", "heuristic_results.json")
MODEL_PATH = os.path.join(ROOT_DIR, "outputs", "dataforge-surgeon")

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

MAX_UI_STEPS = 5
TIER_LABELS = {
    "Tier 1": {
        "tier": 1,
        "label": "Tier 1",
        "description": "Nulls, type errors, and range breaks",
    },
    "Tier 2": {
        "tier": 2,
        "label": "Tier 2",
        "description": "Cross-column drift and clustered failures",
    },
    "Tier 3": {
        "tier": 3,
        "label": "Tier 3",
        "description": "Relational integrity and duplicate mutations",
    },
}


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _new_state():
    return {
        "dirty": None,
        "gt": None,
        "meta": None,
        "tier": 1,
        "initial_accuracy": None,
    }


def local_model_available(path=None) -> bool:
    try:
        _resolve_loadable_model_path(path or MODEL_PATH)
        return True
    except FileNotFoundError:
        return False


def available_agent_choices():
    choices = ["Naive Baseline", "Heuristic Surgeon"]
    if local_model_available():
        choices.append("Live GRPO Model")
    return choices


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def get_training_data():
    try:
        df = pd.read_csv(LOG_PATH)
        if len(df):
            return df
    except Exception:
        pass
    return pd.DataFrame({"step": [0], "total_reward": [0.0], "difficulty": [1]})


def _fmt_pp(value):
    return f"{float(value) * 100:+.2f} pp" if value is not None else "Pending"


def _agent_label(label):
    if "Naive" in label:
        return "Naive Baseline"
    if "Heuristic" in label:
        return "Heuristic Surgeon"
    return "Live GRPO Model"


def _align_dup_gt(dirty, gt, meta):
    if meta.get("tool") == "duplicate_row_mutate" and len(dirty) > len(gt):
        src = meta.get("row", 0)
        if src < len(gt):
            return pd.concat([gt, gt.iloc[[src]]], ignore_index=True)
    return gt


def _build_env(dirty, gt, tier):
    corruptor = Corruptor()
    corruptor.force_tier(tier)
    env = DataForgeEnv(corruptor=corruptor, schema=HEALTHCARE_SCHEMA, clean_data=clean_data)
    acc = rc._field_accuracy(dirty, gt)
    env._state = dirty.copy()
    env._ground_truth = gt.copy()
    env._original_dirty = dirty.copy()
    env._prev_accuracy = acc
    env._starting_accuracy = acc
    env._step_count = 0
    env._action_log = []
    env._episode_rewards = []
    env._episode_start = time.time()
    return env, acc


def load_llm():
    global llm_pipeline
    if not local_model_available():
        return False, f"No checkpoint found at {MODEL_PATH}"
    with llm_lock:
        if llm_pipeline is not None:
            return True, "ok"
        try:
            llm_pipeline = load_eval_pipeline(MODEL_PATH)
            return True, "ok"
        except Exception as exc:
            llm_pipeline = None
            return False, str(exc)


def _run_llm(messages):
    with llm_lock:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return llm_pipeline(
                messages,
                max_new_tokens=96,
                temperature=0.1,
                do_sample=False,
                num_return_sequences=1,
            )


def _training_snapshot():
    df = get_training_data()
    latest_reward = None
    best_reward = None
    parse_rate = None
    tiers = "1"
    if "total_reward" in df.columns:
        rewards = pd.to_numeric(df["total_reward"], errors="coerce").dropna()
        if len(rewards):
            latest_reward = float(rewards.iloc[-1])
            best_reward = float(rewards.max())
    if "parse_success_rate" in df.columns:
        parse = pd.to_numeric(df["parse_success_rate"], errors="coerce").dropna()
        if len(parse):
            parse_rate = float(parse.mean() * 100.0)
    if "difficulty" in df.columns:
        levels = sorted(pd.to_numeric(df["difficulty"], errors="coerce").dropna().astype(int).unique())
        if len(levels) == 1:
            tiers = str(levels[0])
        elif len(levels) > 1:
            tiers = f"{levels[0]}-{levels[-1]}"
    return {
        "latest_reward": latest_reward,
        "best_reward": best_reward,
        "parse_rate": parse_rate,
        "tiers": tiers,
    }


# ─── HTML Components ──────────────────────────────────────────────────────────

def _hero_html() -> str:
    grpo = _read_json(EVAL_PATH)
    heur = _read_json(HEUR_PATH)
    training = _training_snapshot()
    grpo_adv = _fmt_pp(grpo.get("surgeon_advantage_accuracy_delta"))
    heur_adv = _fmt_pp(heur.get("surgeon_advantage_accuracy_delta"))
    parse = training["parse_rate"]
    parse_text = f"{parse:.0f}% parse rate" if parse is not None else "Training pending"
    checkpoint_ready = local_model_available()
    checkpoint_text = "Model loaded" if checkpoint_ready else "Baseline mode"
    checkpoint_cls = "pill--live" if checkpoint_ready else "pill--muted"

    return f"""
<div class="df-hero">
  <div class="df-hero__left">
    <div class="df-hero__eyebrow">
      <span class="df-live-dot"></span>
      Meta × PyTorch × HuggingFace × Scaler &nbsp;·&nbsp; OpenEnv Grand Finale 2026
    </div>
    <h1 class="df-hero__title">Data<span class="df-hero__accent">Forge</span></h1>
    <p class="df-hero__sub">Autonomous tabular repair. Ground-truth reward. Causal constraint reasoning.</p>
    <div class="df-hero__tags">
      <span class="df-tag df-tag--green">GRPO {_e(grpo_adv)}</span>
      <span class="df-tag">Heuristic {_e(heur_adv)}</span>
      <span class="df-tag">{_e(parse_text)}</span>
      <span class="df-tag df-tag--{checkpoint_cls}">{_e(checkpoint_text)}</span>
    </div>
  </div>
  <div class="df-hero__right">
    <div class="df-stat-grid">
      <div class="df-stat">
        <div class="df-stat__label">Reward signal</div>
        <div class="df-stat__val">Δ Accuracy</div>
      </div>
      <div class="df-stat">
        <div class="df-stat__label">Action space</div>
        <div class="df-stat__val">8 tools</div>
      </div>
      <div class="df-stat">
        <div class="df-stat__label">Curricula</div>
        <div class="df-stat__val">3 tiers</div>
      </div>
      <div class="df-stat">
        <div class="df-stat__label">Schema</div>
        <div class="df-stat__val">Healthcare</div>
      </div>
    </div>
    <div class="df-hero__desc">
      Multi-signal RL environment for enterprise data cleaning. Each episode: corrupt → observe → repair → reward.
    </div>
  </div>
</div>
"""


def _benchmark_html() -> str:
    grpo = _read_json(EVAL_PATH)
    heur = _read_json(HEUR_PATH)
    training = _training_snapshot()
    heuristic_value = heur.get("surgeon_advantage_accuracy_delta")
    grpo_value = grpo.get("surgeon_advantage_accuracy_delta")
    latest_reward = training.get("latest_reward")
    best_reward = training.get("best_reward")
    parse_rate = training.get("parse_rate")

    def _bar(value):
        if value is None:
            return 0
        pct = max(min((float(value) + 0.5) * 100, 100), 0)
        return pct

    def _val_cls(value):
        if value is None:
            return "df-num--muted"
        return "df-num--green" if float(value) >= 0 else "df-num--red"

    heur_bar = _bar(heuristic_value)
    grpo_bar = _bar(grpo_value)
    latest_text = f"{latest_reward:+.3f}" if latest_reward is not None else "—"
    best_text = f"{best_reward:+.3f}" if best_reward is not None else "—"
    parse_text = f"{parse_rate:.0f}%" if parse_rate is not None else "—"

    return f"""
<div class="df-panel">
  <div class="df-panel__head">
    <span class="df-label">Performance</span>
    <h3 class="df-panel__title">Benchmark evidence</h3>
  </div>
  <div class="df-bench-row">
    <div class="df-bench-row__name">Heuristic Surgeon</div>
    <div class="df-bench-row__note">Rule-based, constraint-aware</div>
    <div class="df-bench-row__bar"><div class="df-bench-row__fill" style="width:{heur_bar:.1f}%"></div></div>
    <div class="df-bench-row__val {_val_cls(heuristic_value)}">{_e(_fmt_pp(heuristic_value))}</div>
  </div>
  <div class="df-bench-row">
    <div class="df-bench-row__name">GRPO Checkpoint</div>
    <div class="df-bench-row__note">Trained RL policy</div>
    <div class="df-bench-row__bar"><div class="df-bench-row__fill df-bench-row__fill--gold" style="width:{grpo_bar:.1f}%"></div></div>
    <div class="df-bench-row__val {_val_cls(grpo_value)}">{_e(_fmt_pp(grpo_value))}</div>
  </div>
  <div class="df-bench-footer">
    <div class="df-bench-kv"><span>Latest reward</span><strong>{_e(latest_text)}</strong></div>
    <div class="df-bench-kv"><span>Best reward</span><strong>{_e(best_text)}</strong></div>
    <div class="df-bench-kv"><span>Parse rate</span><strong>{_e(parse_text)}</strong></div>
  </div>
</div>
"""


def _status_html(tier: int, rows, accuracy, label: str) -> str:
    row_text = str(rows) if rows is not None else "—"
    acc_pct = f"{accuracy:.1%}" if accuracy is not None else "—"
    acc_fill = f"{max(min(accuracy or 0, 1.0), 0.0) * 100:.1f}" if accuracy is not None else "0"
    acc_cls = "status__acc--good" if (accuracy or 0) > 0.75 else "status__acc--warn" if (accuracy or 0) > 0.5 else "status__acc--bad"
    return f"""
<div class="df-statusbar">
  <div class="df-statusbar__item">
    <span class="df-statusbar__k">Tier</span>
    <span class="df-statusbar__v">{tier}</span>
  </div>
  <div class="df-statusbar__sep"></div>
  <div class="df-statusbar__item">
    <span class="df-statusbar__k">Rows</span>
    <span class="df-statusbar__v">{_e(row_text)}</span>
  </div>
  <div class="df-statusbar__sep"></div>
  <div class="df-statusbar__item">
    <span class="df-statusbar__k">Health</span>
    <span class="df-statusbar__v {acc_cls}">{_e(acc_pct)}</span>
  </div>
  <div class="df-statusbar__sep"></div>
  <div class="df-statusbar__item df-statusbar__item--wide">
    <span class="df-statusbar__k">Status</span>
    <span class="df-statusbar__v">{_e(label)}</span>
  </div>
  <div class="df-statusbar__track">
    <div class="df-statusbar__fill {acc_cls}" style="width:{acc_fill}%"></div>
  </div>
</div>
"""


def _accuracy_html(before, after) -> str:
    if before is None or after is None:
        return """
<div class="df-panel df-panel--accent-none">
  <div class="df-label">Accuracy</div>
  <div class="df-empty-state">
    <div class="df-empty-state__icon">◎</div>
    <div class="df-empty-state__text">Run a repair episode to see accuracy metrics</div>
  </div>
</div>
"""
    delta = after - before
    delta_cls = "df-num--green" if delta >= 0 else "df-num--red"
    delta_sign = "+" if delta >= 0 else ""
    fill_pct = f"{max(min(after, 1.0), 0.0) * 100:.1f}"
    delta_bar = f"{max(min(abs(delta), 1.0), 0.0) * 100:.1f}"
    bar_cls = "df-bar__fill--green" if delta >= 0 else "df-bar__fill--red"
    return f"""
<div class="df-panel">
  <div class="df-label">Accuracy</div>
  <div class="df-acc-main">
    <div class="df-acc-col">
      <div class="df-acc-num">{before:.1%}</div>
      <div class="df-acc-lbl">Before</div>
    </div>
    <div class="df-acc-arrow">→</div>
    <div class="df-acc-col">
      <div class="df-acc-num">{after:.1%}</div>
      <div class="df-acc-lbl">After</div>
    </div>
    <div class="df-acc-sep"></div>
    <div class="df-acc-col">
      <div class="df-acc-delta {delta_cls}">{delta_sign}{delta * 100:.2f}<span class="df-acc-unit">pp</span></div>
      <div class="df-acc-lbl">Delta</div>
    </div>
  </div>
  <div class="df-bar df-bar--track">
    <div class="df-bar__fill df-bar__fill--green" style="width:{fill_pct}%"></div>
  </div>
  <div class="df-bar df-bar--delta">
    <div class="df-bar__fill {bar_cls}" style="width:{delta_bar}%"></div>
  </div>
</div>
"""


def _brief_html(meta=None, obs=None, agent_type=None, note=None) -> str:
    corruption = meta.get("tool", "—") if meta else "—"
    target_hint = getattr(obs, "target_cell_hint", "") if obs else ""
    violation = getattr(obs, "violation_type", "") if obs else ""
    error_count = getattr(obs, "total_errors", None) if obs else None
    agent_text = agent_type or "No agent selected"
    note_text = note or "Generate a scenario to begin."
    error_line = str(error_count) if error_count is not None else "—"
    hint_block = target_hint or "Target hint appears after scenario generation."
    violation_block = violation or "Violation type: pending."
    return f"""
<div class="df-panel">
  <div class="df-label">Episode brief</div>
  <div class="df-brief-agent">{_e(agent_text)}</div>
  <div class="df-brief-grid">
    <div class="df-brief-kv">
      <span>Corruption</span>
      <strong>{_e(corruption)}</strong>
    </div>
    <div class="df-brief-kv">
      <span>Errors</span>
      <strong>{_e(error_line)}</strong>
    </div>
  </div>
  <div class="df-brief-hint">{_e(hint_block)}</div>
  <div class="df-brief-note">{_e(violation_block)}</div>
  <div class="df-brief-caption">{_e(note_text)}</div>
</div>
"""


def _empty_timeline_html() -> str:
    return """
<div class="df-panel df-timeline-empty">
  <div class="df-label">Repair trail</div>
  <div class="df-empty-state">
    <div class="df-empty-state__icon">⬡</div>
    <div class="df-empty-state__text">Execute a repair policy to stream the action trail</div>
  </div>
  <div class="df-timeline-hint">
    Each step shows the tool used, the row and column targeted, reward earned, and the agent's reasoning chain.
  </div>
</div>
"""


def _timeline_html(rollouts: list[dict], total_steps: int) -> str:
    if not rollouts:
        return _empty_timeline_html()

    rows_html = []
    cumulative = 0.0
    for idx, item in enumerate(rollouts, start=1):
        reward = float(item.get("reward", 0.0))
        cumulative += reward
        reward_cls = "tl-reward--green" if reward >= 0 else "tl-reward--red"
        cum_cls = "tl-cum--green" if cumulative >= 0 else "tl-cum--red"
        components = item.get("components", {})
        comp_bits = []
        for key in ("constraint_alignment", "schema_alignment", "reasoning_quality", "parse_bonus"):
            v = components.get(key)
            if v is not None:
                comp_bits.append(f"{key.replace('_', ' ')} {float(v):+.2f}")
        comp_text = "  ·  ".join(comp_bits) if comp_bits else ""
        progress = int((idx / total_steps) * 100)
        rows_html.append(f"""
<div class="df-tl-row">
  <div class="df-tl-row__left">
    <div class="df-tl-step">{idx:02d}<span>/{total_steps}</span></div>
    <div class="df-tl-progress-line" style="height:{100 - progress}%"></div>
  </div>
  <div class="df-tl-row__body">
    <div class="df-tl-row__head">
      <span class="df-tl-tool">{_e(item.get("tool_name", "?"))}</span>
      <span class="df-tl-reward {reward_cls}">{reward:+.3f}</span>
      <span class="df-tl-cum {cum_cls}">Σ {cumulative:+.3f}</span>
    </div>
    <div class="df-tl-row__target">row {item.get("row_id", "?")} &nbsp;·&nbsp; column <strong>{_e(item.get("column_name", "?"))}</strong></div>
    <div class="df-tl-row__reason">{_e(item.get("reasoning", ""))}</div>
    {f'<div class="df-tl-row__comp">{_e(comp_text)}</div>' if comp_text else ""}
  </div>
</div>
""")

    return f"""
<div class="df-panel">
  <div class="df-tl-head">
    <div><span class="df-label">Repair trail</span><h3 class="df-tl-title">Episode timeline</h3></div>
    <div class="df-tl-summary">
      <span>{len(rollouts)} actions</span>
      <span>·</span>
      <span class="{'df-num--green' if cumulative >= 0 else 'df-num--red'}">Σ {cumulative:+.3f}</span>
    </div>
  </div>
  <div class="df-tl-body">
    {''.join(rows_html)}
  </div>
</div>
"""


def _diff_html(original, current, gt) -> str:
    if original is None or current is None or gt is None:
        return """
<div class="df-panel">
  <div class="df-label">Repair ledger</div>
  <div class="df-empty-state">
    <div class="df-empty-state__icon">⊞</div>
    <div class="df-empty-state__text">The cell-level diff appears after the first repair action</div>
  </div>
</div>
"""

    cols = [c for c in current.columns if c != "_is_deleted"]
    limit = min(len(current), len(gt))
    fixed = 0
    regressed = 0
    remaining = 0
    changes = []

    for row_idx in range(limit):
        for col in cols:
            before = original.at[row_idx, col]
            after = current.at[row_idx, col]
            target = gt.at[row_idx, col]
            before_ok = rc._values_match(before, target)
            after_ok = rc._values_match(after, target)
            if not after_ok:
                remaining += 1
            if not rc._values_match(before, after):
                if not before_ok and after_ok:
                    fixed += 1
                    status = "Fixed"
                elif before_ok and not after_ok:
                    regressed += 1
                    status = "Regressed"
                else:
                    status = "Shifted"
                if len(changes) < 8:
                    changes.append((status, row_idx, col, before, after, target))

    def status_cls(s):
        if s == "Fixed":
            return "diff-status--green"
        if s == "Regressed":
            return "diff-status--red"
        return "diff-status--muted"

    rows_html = ""
    if changes:
        for status, row_idx, col, before, after, target in changes:
            rows_html += f"""
<tr>
  <td><span class="diff-status {status_cls(status)}">{_e(status)}</span></td>
  <td class="diff-td--mono">{row_idx}</td>
  <td class="diff-td--mono">{_e(col)}</td>
  <td class="diff-td--before">{_e(before)}</td>
  <td class="diff-td--after">{_e(after)}</td>
  <td class="diff-td--target">{_e(target)}</td>
</tr>
"""
    else:
        rows_html = "<tr><td colspan='6' class='diff-td--empty'>No cell changes recorded yet.</td></tr>"

    score_pct = 0 if (fixed + remaining) == 0 else round(fixed / (fixed + remaining) * 100)

    return f"""
<div class="df-panel">
  <div class="df-tl-head">
    <div><span class="df-label">Repair ledger</span><h3 class="df-tl-title">Cell-level diff</h3></div>
    <div class="df-diff-score">
      <span class="df-num--green">{fixed} fixed</span>
      <span class="df-num--red">{regressed} regressed</span>
      <span class="df-muted">{remaining} remaining</span>
    </div>
  </div>
  <div class="df-bar df-bar--track" style="margin-bottom:20px">
    <div class="df-bar__fill df-bar__fill--green" style="width:{score_pct}%"></div>
  </div>
  <div class="df-diff-wrap">
    <table class="df-diff-table">
      <thead>
        <tr>
          <th>Status</th><th>Row</th><th>Column</th>
          <th>Before</th><th>After</th><th>Target</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>
"""


# ─── Business logic (unchanged) ───────────────────────────────────────────────

def heuristic_surgeon(state, gt):
    cols = [c for c in state.columns if c != "_is_deleted"]
    for row_idx in range(min(len(state), len(gt))):
        for col_idx, col in enumerate(cols):
            cell = state.at[row_idx, col]
            gt_cell = gt.at[row_idx, col]
            if pd.isna(cell) and pd.notna(gt_cell):
                col_type = HEALTHCARE_SCHEMA.get(col, {}).get("type", "str")
                tool_id = 0 if col_type in ("int", "float") else 1
                tool_name = "IMPUTE_MEDIAN" if tool_id == 0 else "IMPUTE_MODE"
                return SurgeonAction(
                    reasoning=f"Null in {col} use {tool_name}",
                    tool_id=tool_id,
                    column=col_idx,
                    row_id=row_idx,
                )
            if pd.notna(cell) and pd.notna(gt_cell) and str(cell) != str(gt_cell):
                if str(cell).startswith("ERR_"):
                    col_type = HEALTHCARE_SCHEMA.get(col, {}).get("type", "str")
                    tool_id = 0 if col_type in ("int", "float") else 1
                    return SurgeonAction(
                        reasoning=f"Type error in {col}",
                        tool_id=tool_id,
                        column=col_idx,
                        row_id=row_idx,
                    )
                return SurgeonAction(
                    reasoning=f"Format error in {col}",
                    tool_id=3,
                    column=col_idx,
                    row_id=row_idx,
                )
    if len(state) > len(gt):
        return SurgeonAction(reasoning="Duplicate row delete", tool_id=4, column=0, row_id=len(state) - 1)
    return SurgeonAction(reasoning="No errors detected", tool_id=7, column=0, row_id=0)


def _naive_action(state):
    cols = [c for c in state.columns if c != "_is_deleted"]
    for row_idx in range(len(state)):
        for col_idx, col in enumerate(cols):
            cell = state.at[row_idx, col]
            if pd.isna(cell):
                return SurgeonAction(
                    reasoning=f"Null in {col}",
                    tool_id=0,
                    column=col_idx,
                    row_id=row_idx,
                )
            if str(cell).startswith("ERR_"):
                return SurgeonAction(
                    reasoning=f"Type error in {col}",
                    tool_id=0,
                    column=col_idx,
                    row_id=row_idx,
                )
    return SurgeonAction(reasoning="No errors detected", tool_id=7, column=0, row_id=0)


def _scenario_snapshot(dirty, gt, meta, tier):
    env, accuracy = _build_env(dirty, gt, tier)
    obs = env._make_observation()
    cols = [c for c in dirty.columns if c != "_is_deleted"]
    total_errors = obs.total_errors
    dirty_display = dirty[cols].head(10).copy()
    return {
        "obs": obs,
        "accuracy": accuracy,
        "total_errors": total_errors,
        "dirty_display": dirty_display,
        "repaired_display": dirty_display.copy(),
        "status_html": _status_html(tier, len(dirty), accuracy, "Scenario ready"),
        "accuracy_html": _accuracy_html(accuracy, accuracy),
        "brief_html": _brief_html(
            meta=meta,
            obs=obs,
            agent_type="Scenario seeded",
            note=TIER_LABELS.get(f"Tier {tier}", {}).get("description", ""),
        ),
    }


def _seed_session(tier_label, session_state):
    session_state = dict(session_state or _new_state())
    tier_info = TIER_LABELS.get(tier_label, TIER_LABELS["Tier 1"])
    tier = int(tier_info["tier"])
    corruptor = Corruptor()
    corruptor.force_tier(tier)
    sample = clean_data.sample(n=min(50, len(clean_data))).reset_index(drop=True)
    dirty, gt, meta = corruptor.generate_episode(sample)
    gt = _align_dup_gt(dirty, gt, meta)
    session_state.update(
        {
            "dirty": dirty.copy(),
            "gt": gt.copy(),
            "meta": meta,
            "tier": tier,
        }
    )
    return session_state, dirty, gt, meta, tier


def generate_episode(tier_label, session_state):
    session_state, dirty, gt, meta, tier = _seed_session(tier_label, session_state)
    snap = _scenario_snapshot(dirty, gt, meta, tier)
    return (
        snap["status_html"],
        snap["dirty_display"],
        snap["repaired_display"],
        snap["accuracy_html"],
        snap["brief_html"],
        _empty_timeline_html(),
        _diff_html(dirty, dirty, gt),
        session_state,
    )


def _live_grpo_action(env):
    ok, message = load_llm()
    if not ok:
        raise RuntimeError(message)
    obs = env._make_observation()
    messages = [
        {"role": "system", "content": build_prompt(obs)},
        {"role": "user", "content": f"Observation: {obs.model_dump_json()}\nOutput valid JSON only."},
    ]
    output = _run_llm(messages)
    raw = output[0]["generated_text"][-1]["content"]
    return robust_parse_action(raw, require_fields=True)


def simulate_with_repaired(agent_value, tier_label, session_state):
    agent_type = _agent_label(agent_value)
    session_state = dict(session_state or _new_state())
    dirty = session_state.get("dirty")
    gt = session_state.get("gt")
    requested_tier = int(TIER_LABELS.get(tier_label, TIER_LABELS["Tier 1"])["tier"])
    tier = int(session_state.get("tier", requested_tier))
    meta = session_state.get("meta") or {}

    if dirty is None or gt is None or tier != requested_tier:
        session_state, dirty, gt, meta, tier = _seed_session(tier_label, session_state)

    env, acc_before = _build_env(dirty.copy(), gt.copy(), tier)
    cols = [c for c in env._state.columns if c != "_is_deleted"]
    rollouts = []

    for _ in range(MAX_UI_STEPS):
        try:
            if agent_type == "Naive Baseline":
                action = _naive_action(env._state.copy())
            elif agent_type == "Heuristic Surgeon":
                action = heuristic_surgeon(env._state.copy(), gt)
            else:
                action = _live_grpo_action(env)
        except Exception as exc:
            yield (
                _status_html(tier, len(env._state), acc_before, "Execution blocked"),
                dirty[cols].head(10).copy(),
                env._state[cols].head(10).copy(),
                _accuracy_html(acc_before, None),
                _brief_html(
                    meta=meta,
                    obs=env._make_observation(),
                    agent_type=agent_type,
                    note=f"Live model unavailable: {str(exc)[:120]}",
                ),
                _timeline_html(rollouts, MAX_UI_STEPS),
                _diff_html(dirty, env._state, gt),
                session_state,
            )
            return

        _, total_reward, done, info = env.step(action)
        obs = env._make_observation()
        components = info.get("reward_components", {})
        rollouts.append(
            {
                "reasoning": action.reasoning.replace("EXACT_PARSE:", "").strip(),
                "tool_name": SURGEON_TOOLS.get(action.tool_id, {"name": "?"})["name"],
                "reward": total_reward,
                "row_id": action.row_id,
                "column_name": cols[action.column] if action.column < len(cols) else "?",
                "components": components,
                "violation_type": getattr(obs, "violation_type", ""),
                "target_cell_hint": getattr(obs, "target_cell_hint", ""),
            }
        )

        current_acc = rc._field_accuracy(env._state, gt)
        note = "Last action executed cleanly."
        if info.get("invalid_action"):
            note = "Agent produced an invalid action."
        yield (
            _status_html(tier, len(env._state), current_acc, f"{agent_type} · step {len(rollouts)}/{MAX_UI_STEPS}"),
            dirty[cols].head(10).copy(),
            env._state[cols].head(10).copy(),
            _accuracy_html(acc_before, current_acc),
            _brief_html(meta=meta, obs=obs, agent_type=agent_type, note=note),
            _timeline_html(rollouts, MAX_UI_STEPS),
            _diff_html(dirty, env._state, gt),
            session_state,
        )
        if done:
            break


def _load_initial_view():
    session_state, dirty, gt, meta, tier = _seed_session("Tier 1", _new_state())
    snap = _scenario_snapshot(dirty, gt, meta, tier)
    return (
        _hero_html(),
        _benchmark_html(),
        get_training_data(),
        snap["status_html"],
        snap["dirty_display"],
        snap["repaired_display"],
        snap["accuracy_html"],
        snap["brief_html"],
        _empty_timeline_html(),
        _diff_html(dirty, dirty, gt),
        session_state,
    )


# ─── Design System ────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --void:    #020304;
  --base:    #06080D;
  --sur:     #0C0F17;
  --raised:  #11151F;
  --border:  rgba(255,255,255,0.07);
  --border2: rgba(255,255,255,0.12);
  --text:    #E8EDF5;
  --text2:   #6B7A92;
  --text3:   #3A4456;
  --green:   #00E87A;
  --green2:  rgba(0,232,122,0.12);
  --green3:  rgba(0,232,122,0.06);
  --red:     #FF3D5A;
  --red2:    rgba(255,61,90,0.1);
  --gold:    #F5B731;
  --gold2:   rgba(245,183,49,0.12);
  --shadow:  0 24px 64px rgba(0,0,0,0.5);
  --radius:  16px;
  --radius2: 10px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--base) !important;
  color: var(--text) !important;
  font-family: 'Syne', sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

/* ── Grid noise overlay ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
  background-size: 180px;
  pointer-events: none;
  z-index: 0;
  opacity: 0.6;
}

.gradio-container {
  max-width: 1600px !important;
  margin: 0 auto !important;
  padding: 24px 28px 64px !important;
  background: transparent !important;
  position: relative;
  z-index: 1;
}

.gradio-container * {
  font-family: 'Syne', sans-serif;
}

.gradio-container code,
.gradio-container pre,
.df-tl-row__comp,
.df-diff-table,
.diff-td--mono {
  font-family: 'JetBrains Mono', monospace !important;
}

/* ── Gradio overrides ── */
.gradio-html,
.gradio-dataframe,
.gradio-plot { background: transparent !important; border: none !important; box-shadow: none !important; }

.gradio-dataframe {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--sur) !important;
  overflow: hidden !important;
}

.gradio-dataframe table { font-size: 12px !important; }
.gradio-dataframe th { background: var(--raised) !important; color: var(--text2) !important; font-size: 11px !important; letter-spacing: 0.08em !important; }
.gradio-dataframe td { color: var(--text) !important; border-color: var(--border) !important; }

.gradio-radio label {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius2) !important;
  background: var(--sur) !important;
  color: var(--text2) !important;
  padding: 12px 14px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  transition: all 0.18s ease !important;
  cursor: pointer !important;
}

.gradio-radio label:hover {
  border-color: var(--border2) !important;
  color: var(--text) !important;
  background: var(--raised) !important;
}

.gradio-radio label.selected {
  background: var(--green2) !important;
  border-color: var(--green) !important;
  color: var(--text) !important;
}

.gradio-radio .wrap { gap: 8px !important; }

/* Tab styling */
.tab-nav { border-bottom: 1px solid var(--border) !important; background: transparent !important; }
.tab-nav button {
  font-family: 'Syne', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--text2) !important;
  padding: 10px 18px !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
  transition: all 0.18s !important;
}
.tab-nav button.selected {
  color: var(--text) !important;
  border-bottom-color: var(--green) !important;
}
.tab-nav button:hover { color: var(--text) !important; }

/* ── Hero ── */
.df-hero {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 24px;
  padding: 40px 44px;
  border: 1px solid var(--border);
  border-radius: 24px;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%),
    var(--sur);
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}

.df-hero::before {
  content: 'DataForge';
  position: absolute;
  right: -20px;
  bottom: -40px;
  font-size: 160px;
  font-weight: 800;
  color: rgba(255,255,255,0.018);
  pointer-events: none;
  letter-spacing: -8px;
  user-select: none;
}

.df-hero__eyebrow {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text2);
  margin-bottom: 16px;
}

.df-live-dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 0 rgba(0,232,122,0.5);
  animation: pulse-dot 2.4s ease infinite;
}

@keyframes pulse-dot {
  0%   { box-shadow: 0 0 0 0 rgba(0,232,122,0.5); }
  60%  { box-shadow: 0 0 0 8px rgba(0,232,122,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,232,122,0); }
}

.df-hero__title {
  font-size: 72px;
  font-weight: 800;
  line-height: 0.92;
  letter-spacing: -3px;
  margin-bottom: 18px;
}

.df-hero__accent { color: var(--green); }

.df-hero__sub {
  font-size: 16px;
  line-height: 1.65;
  color: var(--text2);
  max-width: 520px;
  margin-bottom: 24px;
}

.df-hero__tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.df-tag {
  padding: 7px 13px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--raised);
  color: var(--text2);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.03em;
}

.df-tag--green {
  border-color: rgba(0,232,122,0.3);
  background: var(--green2);
  color: var(--green);
}

.df-tag--pill--live {
  border-color: rgba(0,232,122,0.3);
  background: var(--green2);
  color: var(--green);
}

.df-tag--pill--muted { color: var(--text3); }

.df-hero__right {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.df-stat-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.df-stat {
  padding: 16px 18px;
  border: 1px solid var(--border);
  border-radius: var(--radius2);
  background: var(--raised);
}

.df-stat__label {
  font-size: 11px;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 6px;
}

.df-stat__val {
  font-size: 18px;
  font-weight: 700;
  color: var(--text);
}

.df-hero__desc {
  font-size: 13px;
  line-height: 1.7;
  color: var(--text2);
  padding: 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius2);
  background: var(--raised);
}

/* ── Layout panels ── */
.df-sidebar {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.df-sidebar-brand {
  padding: 20px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--sur);
}

.df-sidebar-brand__title {
  font-size: 13px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 4px;
}

.df-sidebar-brand__sub {
  font-size: 12px;
  color: var(--text2);
  line-height: 1.5;
}

.df-sidebar-section {
  padding: 18px 20px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--sur);
}

.df-sidebar-section label.block {
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--text2) !important;
  margin-bottom: 12px !important;
}

/* ── Buttons ── */
button.df-btn-primary {
  width: 100% !important;
  min-height: 52px !important;
  border-radius: var(--radius2) !important;
  border: none !important;
  background: var(--text) !important;
  color: var(--void) !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 14px !important;
  font-weight: 800 !important;
  letter-spacing: 0.02em !important;
  cursor: pointer !important;
  transition: all 0.18s ease !important;
  box-shadow: 0 4px 20px rgba(232,237,245,0.1) !important;
}

button.df-btn-primary:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(232,237,245,0.18) !important;
}

button.df-btn-secondary {
  width: 100% !important;
  min-height: 48px !important;
  border-radius: var(--radius2) !important;
  border: 1px solid var(--border2) !important;
  background: transparent !important;
  color: var(--text) !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 13px !important;
  font-weight: 700 !important;
  cursor: pointer !important;
  transition: all 0.18s ease !important;
}

button.df-btn-secondary:hover {
  background: var(--raised) !important;
  border-color: var(--border2) !important;
}

/* ── Generic panel ── */
.df-panel {
  padding: 22px 24px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--sur);
}

.df-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--text2);
  margin-bottom: 4px;
  display: block;
}

.df-panel__head { margin-bottom: 18px; }
.df-panel__title {
  font-size: 20px;
  font-weight: 700;
  margin-top: 4px;
}

/* ── Status bar ── */
.df-statusbar {
  display: flex;
  align-items: center;
  gap: 0;
  padding: 0 20px;
  border: 1px solid var(--border);
  border-radius: var(--radius2);
  background: var(--raised);
  min-height: 48px;
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
}

.df-statusbar__item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 16px;
  height: 48px;
}

.df-statusbar__item--wide { flex: 1; }

.df-statusbar__sep {
  width: 1px;
  height: 20px;
  background: var(--border);
}

.df-statusbar__k {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text3);
}

.df-statusbar__v {
  font-size: 13px;
  font-weight: 700;
  color: var(--text);
}

.status__acc--good { color: var(--green) !important; }
.status__acc--warn { color: var(--gold) !important; }
.status__acc--bad  { color: var(--red) !important; }

.df-statusbar__track {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--border);
}

.df-statusbar__fill {
  height: 100%;
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}

.df-statusbar__fill.status__acc--good { background: var(--green); }
.df-statusbar__fill.status__acc--warn { background: var(--gold); }
.df-statusbar__fill.status__acc--bad  { background: var(--red); }

/* ── Accuracy ── */
.df-acc-main {
  display: flex;
  align-items: center;
  gap: 20px;
  margin: 18px 0 20px;
  flex-wrap: wrap;
}

.df-acc-col { text-align: center; }

.df-acc-num {
  font-size: 36px;
  font-weight: 800;
  letter-spacing: -1.5px;
  line-height: 1;
}

.df-acc-delta {
  font-size: 42px;
  font-weight: 800;
  letter-spacing: -2px;
  line-height: 1;
}

.df-acc-unit {
  font-size: 18px;
  font-weight: 600;
  margin-left: 2px;
}

.df-acc-lbl {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text2);
  margin-top: 8px;
}

.df-acc-arrow {
  font-size: 24px;
  color: var(--text3);
}

.df-acc-sep {
  width: 1px;
  height: 50px;
  background: var(--border);
  margin: 0 4px;
}

/* ── Progress bars ── */
.df-bar {
  border-radius: 999px;
  overflow: hidden;
  background: var(--border);
}

.df-bar--track { height: 4px; }
.df-bar--delta { height: 2px; margin-top: 6px; }

.df-bar__fill {
  height: 100%;
  border-radius: 999px;
  transition: width 0.7s cubic-bezier(0.4,0,0.2,1);
}

.df-bar__fill--green { background: var(--green); }
.df-bar__fill--red   { background: var(--red); }
.df-bar__fill--gold  { background: var(--gold); }

/* ── Number colors ── */
.df-num--green { color: var(--green); }
.df-num--red   { color: var(--red); }
.df-num--muted { color: var(--text2); }
.df-muted      { color: var(--text2); }

/* ── Brief ── */
.df-brief-agent {
  font-size: 20px;
  font-weight: 700;
  margin: 6px 0 16px;
}

.df-brief-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 16px;
}

.df-brief-kv {
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius2);
  background: var(--raised);
}

.df-brief-kv span {
  display: block;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text2);
  margin-bottom: 4px;
}

.df-brief-kv strong {
  font-size: 14px;
  font-weight: 700;
}

.df-brief-hint {
  font-size: 13px;
  line-height: 1.6;
  color: var(--text);
  margin-bottom: 10px;
}

.df-brief-note {
  font-size: 12px;
  color: var(--text2);
  margin-bottom: 8px;
}

.df-brief-caption {
  font-size: 11px;
  color: var(--text3);
  font-family: 'JetBrains Mono', monospace;
}

/* ── Benchmark ── */
.df-bench-row {
  display: grid;
  grid-template-columns: 160px 1fr 140px 100px;
  align-items: center;
  gap: 14px;
  padding: 14px 0;
  border-top: 1px solid var(--border);
}

.df-bench-row:first-of-type { border-top: none; }

.df-bench-row__name {
  font-size: 14px;
  font-weight: 700;
}

.df-bench-row__note {
  font-size: 12px;
  color: var(--text2);
}

.df-bench-row__bar {
  height: 4px;
  border-radius: 999px;
  background: var(--border);
  overflow: hidden;
}

.df-bench-row__fill {
  height: 100%;
  background: var(--green);
  border-radius: 999px;
  transition: width 0.8s ease;
}

.df-bench-row__fill--gold { background: var(--gold); }

.df-bench-row__val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  font-weight: 600;
  text-align: right;
}

.df-bench-footer {
  display: flex;
  gap: 24px;
  margin-top: 18px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
}

.df-bench-kv { display: flex; flex-direction: column; gap: 3px; }
.df-bench-kv span { font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.1em; }
.df-bench-kv strong { font-size: 15px; font-weight: 700; }

/* ── Timeline ── */
.df-tl-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 20px;
}

.df-tl-title {
  font-size: 20px;
  font-weight: 700;
  margin-top: 4px;
}

.df-tl-summary {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--text2);
}

.df-tl-body { display: flex; flex-direction: column; gap: 0; }

.df-tl-row {
  display: grid;
  grid-template-columns: 60px 1fr;
  gap: 16px;
  padding: 18px 0;
  border-top: 1px solid var(--border);
  animation: tl-in 0.3s ease both;
}

.df-tl-row:first-child { border-top: none; padding-top: 0; }

@keyframes tl-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

.df-tl-row__left { text-align: center; position: relative; }

.df-tl-step {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 600;
  color: var(--gold);
  white-space: nowrap;
}

.df-tl-step span { color: var(--text3); }

.df-tl-progress-line {
  width: 1px;
  background: linear-gradient(to bottom, var(--border2), transparent);
  margin: 6px auto 0;
}

.df-tl-row__head {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 8px;
}

.df-tl-tool {
  font-size: 15px;
  font-weight: 700;
}

.df-tl-reward {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  font-weight: 600;
}

.df-tl-reward.tl-reward--green { color: var(--green); }
.df-tl-reward.tl-reward--red   { color: var(--red); }

.df-tl-cum {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text3);
  margin-left: auto;
}

.df-tl-cum.df-num--green { color: rgba(0,232,122,0.6); }
.df-tl-cum.df-num--red   { color: rgba(255,61,90,0.6); }

.df-tl-row__target {
  font-size: 12px;
  color: var(--text2);
  margin-bottom: 10px;
}

.df-tl-row__reason {
  font-size: 14px;
  line-height: 1.65;
  color: var(--text);
}

.df-tl-row__comp {
  margin-top: 8px;
  font-size: 11px;
  color: var(--text3);
  letter-spacing: 0.03em;
}

.df-timeline-hint {
  margin-top: 16px;
  font-size: 12px;
  color: var(--text3);
  line-height: 1.6;
  border-top: 1px solid var(--border);
  padding-top: 14px;
}

/* ── Empty states ── */
.df-empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 40px 20px;
  color: var(--text3);
}

.df-empty-state__icon {
  font-size: 32px;
  opacity: 0.4;
}

.df-empty-state__text {
  font-size: 14px;
  text-align: center;
  max-width: 280px;
  line-height: 1.6;
}

/* ── Diff ── */
.df-diff-score {
  display: flex;
  align-items: center;
  gap: 14px;
  font-size: 13px;
  font-weight: 700;
}

.df-diff-wrap { overflow-x: auto; margin-top: 4px; }

.df-diff-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.df-diff-table th {
  text-align: left;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text2);
  white-space: nowrap;
}

.df-diff-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}

.df-diff-table tr:last-child td { border-bottom: none; }
.df-diff-table tr:hover td { background: var(--raised); }

.diff-status {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.diff-status--green { background: var(--green2); color: var(--green); }
.diff-status--red   { background: var(--red2);   color: var(--red);   }
.diff-status--muted { background: var(--raised);  color: var(--text2); }

.diff-td--mono { font-family: 'JetBrains Mono', monospace; }
.diff-td--before { color: var(--text2); text-decoration: line-through; font-family: 'JetBrains Mono', monospace; }
.diff-td--after  { color: var(--text);  font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.diff-td--target { color: var(--green); font-family: 'JetBrains Mono', monospace; }
.diff-td--empty  { color: var(--text3); text-align: center; padding: 32px; }

/* ── Footer watermark ── */
.df-footer {
  margin-top: 48px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.df-footer__brand {
  font-size: 14px;
  font-weight: 700;
}

.df-footer__brand span { color: var(--green); }

.df-footer__meta {
  font-size: 12px;
  color: var(--text3);
}

/* ── Responsive ── */
@media (max-width: 1100px) {
  .df-hero { grid-template-columns: 1fr; }
  .df-hero__title { font-size: 52px; }
  .df-hero::before { display: none; }
  .df-bench-row { grid-template-columns: 1fr 1fr; }
  .df-bench-row__bar { display: none; }
}
"""


def build_demo():
    available = available_agent_choices()
    default_agent = "Live GRPO Model" if "Live GRPO Model" in available else "Heuristic Surgeon"

    with gr.Blocks(title="DataForge Arena", css=CSS) as demo:
        state = gr.State(_new_state())

        # ── Full-width hero
        hero = gr.HTML(_hero_html())

        with gr.Row(equal_height=False, elem_classes=["df-main-row"]):

            # ── Left sidebar – controls
            with gr.Column(scale=3, min_width=260, elem_classes=["df-sidebar"]):

                gr.HTML("""
<div class="df-sidebar-brand">
  <div class="df-sidebar-brand__title">Control Room</div>
  <div class="df-sidebar-brand__sub">Choose a repair policy, set complexity, then seed a scenario and execute.</div>
</div>
""")

                with gr.Group(elem_classes=["df-sidebar-section"]):
                    agent_pick = gr.Radio(
                        choices=available,
                        value=default_agent,
                        label="Repair policy",
                    )

                with gr.Group(elem_classes=["df-sidebar-section"]):
                    tier_pick = gr.Radio(
                        choices=list(TIER_LABELS.keys()),
                        value="Tier 1",
                        label="Complexity tier",
                    )

                new_btn = gr.Button(
                    "⬡ Seed New Scenario",
                    variant="secondary",
                    elem_classes=["df-btn-secondary"],
                )
                run_btn = gr.Button(
                    "Execute Repair Policy",
                    variant="primary",
                    elem_classes=["df-btn-primary"],
                )

                benchmark = gr.HTML(_benchmark_html())

            # ── Right main – tabbed content
            with gr.Column(scale=9):

                status_html = gr.HTML(_status_html(1, None, None, "Seed a scenario to begin"))

                with gr.Tabs():

                    with gr.Tab("⬡  Environment"):
                        with gr.Row():
                            dirty_view = gr.Dataframe(
                                label="Corrupted input",
                                interactive=False,
                                wrap=False,
                            )
                            repaired_view = gr.Dataframe(
                                label="Repaired state",
                                interactive=False,
                                wrap=False,
                            )
                        with gr.Row():
                            accuracy_html = gr.HTML(_accuracy_html(None, None))
                            brief_html = gr.HTML(_brief_html())

                    with gr.Tab("◈  Repair Trail"):
                        timeline_html = gr.HTML(_empty_timeline_html())

                    with gr.Tab("⊞  Diff Ledger"):
                        diff_html = gr.HTML(_diff_html(None, None, None))

                    with gr.Tab("◎  Intelligence"):
                        reward_plot = gr.LinePlot(
                            value=get_training_data(),
                            x="step",
                            y="total_reward",
                            title="Training reward trajectory",
                            height=260,
                            x_title="Step",
                            y_title="Reward",
                            tooltip=["step", "total_reward"],
                        )

        gr.HTML("""
<div class="df-footer">
  <div class="df-footer__brand">Data<span>Forge</span> Arena</div>
  <div class="df-footer__meta">Meta × PyTorch × HuggingFace × Scaler · OpenEnv Grand Finale 2026</div>
</div>
""")

        # ── Events

        def on_generate(tier_value, session_value):
            return generate_episode(tier_value, session_value)

        def on_execute(agent_value, tier_value, session_value):
            yield from simulate_with_repaired(agent_value, tier_value, session_value)

        new_btn.click(
            fn=on_generate,
            inputs=[tier_pick, state],
            outputs=[
                status_html, dirty_view, repaired_view,
                accuracy_html, brief_html,
                timeline_html, diff_html, state,
            ],
        )

        run_btn.click(
            fn=on_execute,
            inputs=[agent_pick, tier_pick, state],
            outputs=[
                status_html, dirty_view, repaired_view,
                accuracy_html, brief_html,
                timeline_html, diff_html, state,
            ],
        )

        demo.load(
            fn=_load_initial_view,
            outputs=[
                hero, benchmark, reward_plot,
                status_html, dirty_view, repaired_view,
                accuracy_html, brief_html,
                timeline_html, diff_html, state,
            ],
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.queue(default_concurrency_limit=8).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        theme=gr.themes.Base(),
        css=CSS,
        footer_links=["gradio", "settings"],
    )