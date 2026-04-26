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


def _hero_html() -> str:
    grpo = _read_json(EVAL_PATH)
    heur = _read_json(HEUR_PATH)
    training = _training_snapshot()
    grpo_adv = _fmt_pp(grpo.get("surgeon_advantage_accuracy_delta"))
    heur_adv = _fmt_pp(heur.get("surgeon_advantage_accuracy_delta"))
    parse = training["parse_rate"]
    parse_text = f"{parse:.0f}% structured output" if parse is not None else "Structured output pending"
    checkpoint_text = "Checkpoint ready" if local_model_available() else "Checkpoint gated"
    return f"""
<section class="hero-shell">
  <div class="hero-copy">
    <div class="hero-kicker">Meta x PyTorch x OpenEnv Hackathon 2026</div>
    <h1 class="hero-title">DataForge Arena</h1>
    <p class="hero-subtitle">
      World-model-driven data repair. One live episode, one repair policy, one clean decision trail.
    </p>
    <div class="hero-strip">
      <span class="hero-pill">GRPO {grpo_adv}</span>
      <span class="hero-pill">Heuristic {heur_adv}</span>
      <span class="hero-pill">{_e(parse_text)}</span>
      <span class="hero-pill">{_e(checkpoint_text)}</span>
    </div>
  </div>
  <div class="hero-panel">
    <div class="hero-panel-label">Environment</div>
    <div class="hero-panel-title">Autonomous tabular repair with causal constraints</div>
    <div class="hero-panel-grid">
      <div><span>Reward</span><strong>Ground truth delta</strong></div>
      <div><span>Signals</span><strong>Constraint, schema, parse</strong></div>
      <div><span>Tiers</span><strong>Three adaptive curricula</strong></div>
      <div><span>Mode</span><strong>Judge-ready HF Space</strong></div>
    </div>
  </div>
</section>
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

    def row(label, value, note):
        tone = "good" if value is not None and value >= 0 else "muted"
        return f"""
<div class="benchmark-row">
  <div>
    <div class="benchmark-name">{_e(label)}</div>
    <div class="benchmark-note">{_e(note)}</div>
  </div>
  <div class="benchmark-value benchmark-{tone}">{_e(_fmt_pp(value))}</div>
</div>
"""

    latest_text = f"{latest_reward:+.2f}" if latest_reward is not None else "Pending"
    best_text = f"{best_reward:+.2f}" if best_reward is not None else "Pending"
    parse_text = f"{parse_rate:.0f}%" if parse_rate is not None else "Pending"
    return f"""
<section class="benchmark-shell">
  <div class="section-tag">Performance</div>
  <h2 class="section-title">Benchmark evidence</h2>
  {row("Heuristic Surgeon", heuristic_value, "Rule-based constraint-aware repairs")}
  {row("GRPO Checkpoint", grpo_value, "Trained policy checkpoint")}
  <div class="benchmark-footer">
    <span>Latest reward <strong>{_e(latest_text)}</strong></span>
    <span>Best reward <strong>{_e(best_text)}</strong></span>
    <span>Parse <strong>{_e(parse_text)}</strong></span>
  </div>
</section>
"""


def _status_html(tier: int, rows: int | None, accuracy: float | None, label: str) -> str:
    row_text = str(rows) if rows is not None else "--"
    acc_text = f"{accuracy:.1%}" if accuracy is not None else "--"
    return f"""
<section class="status-shell">
  <div class="status-item"><span>Tier</span><strong>{tier}</strong></div>
  <div class="status-item"><span>Rows</span><strong>{_e(row_text)}</strong></div>
  <div class="status-item"><span>Health</span><strong>{_e(acc_text)}</strong></div>
  <div class="status-item"><span>Status</span><strong>{_e(label)}</strong></div>
</section>
"""


def _accuracy_html(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return """
<section class="metric-shell">
  <div class="section-tag">Accuracy</div>
  <h3 class="metric-title">No run yet</h3>
  <p class="metric-note">Generate a scenario to compare corrupted and repaired state.</p>
</section>
"""
    delta = after - before
    tone = "good" if delta >= 0 else "bad"
    return f"""
<section class="metric-shell">
  <div class="section-tag">Accuracy</div>
  <div class="metric-grid">
    <div><span>Before</span><strong>{before:.1%}</strong></div>
    <div><span>After</span><strong>{after:.1%}</strong></div>
    <div><span>Delta</span><strong class="{tone}">{delta * 100:+.2f} pp</strong></div>
  </div>
  <div class="metric-track"><span class="metric-fill" style="width:{max(min(after, 1.0), 0.0) * 100:.1f}%"></span></div>
</section>
"""


def _brief_html(meta=None, obs=None, agent_type: str | None = None, note: str | None = None) -> str:
    corruption = meta.get("tool", "Pending") if meta else "Pending"
    target_hint = getattr(obs, "target_cell_hint", "") if obs else ""
    violation = getattr(obs, "violation_type", "") if obs else ""
    error_count = getattr(obs, "total_errors", None) if obs else None
    agent_text = agent_type or "No agent selected"
    note_text = note or "Generate a scenario to inspect the most suspicious row."
    error_line = str(error_count) if error_count is not None else "--"
    hint_block = target_hint or "Target hint will appear after scenario generation."
    violation_block = violation or "Violation type pending."
    return f"""
<section class="brief-shell">
  <div class="section-tag">Run brief</div>
  <h3 class="brief-title">{_e(agent_text)}</h3>
  <div class="brief-grid">
    <div><span>Corruption</span><strong>{_e(corruption)}</strong></div>
    <div><span>Errors</span><strong>{_e(error_line)}</strong></div>
  </div>
  <p class="brief-hint">{_e(hint_block)}</p>
  <p class="brief-note">{_e(violation_block)}</p>
  <p class="brief-caption">{_e(note_text)}</p>
</section>
"""


def _empty_timeline_html() -> str:
    return """
<section class="timeline-shell">
  <div class="section-tag">Repair trace</div>
  <h3 class="timeline-title">Awaiting scenario</h3>
  <p class="timeline-note">Create a scenario, then execute a repair policy to stream the action trail.</p>
</section>
"""


def _timeline_html(rollouts: list[dict], total_steps: int) -> str:
    if not rollouts:
        return _empty_timeline_html()

    rows = []
    for idx, item in enumerate(rollouts, start=1):
        reward = float(item.get("reward", 0.0))
        reward_cls = "good" if reward >= 0 else "bad"
        components = item.get("components", {})
        component_bits = []
        for key in ("constraint_alignment", "schema_alignment", "reasoning_quality", "parse_bonus"):
            value = components.get(key)
            if value is None:
                continue
            component_bits.append(f"{key.replace('_', ' ')} {float(value):+.2f}")
        component_text = " | ".join(component_bits) if component_bits else "No component trace"
        rows.append(
            f"""
<div class="timeline-row">
  <div class="timeline-step">Step {idx:02d} / {total_steps}</div>
  <div class="timeline-body">
    <div class="timeline-head">
      <strong>{_e(item.get("tool_name", "?"))}</strong>
      <span class="timeline-reward {reward_cls}">{reward:+.2f}</span>
    </div>
    <div class="timeline-meta">row {item.get("row_id", "?")} | column {_e(item.get("column_name", "?"))}</div>
    <div class="timeline-reason">{_e(item.get("reasoning", ""))}</div>
    <div class="timeline-components">{_e(component_text)}</div>
  </div>
</div>
"""
        )
    return f"""
<section class="timeline-shell">
  <div class="section-tag">Repair trace</div>
  <h3 class="timeline-title">Episode timeline</h3>
  {''.join(rows)}
</section>
"""


def _diff_html(original, current, gt) -> str:
    if original is None or current is None or gt is None:
        return """
<section class="diff-shell">
  <div class="section-tag">Repair delta</div>
  <h3 class="diff-title">No diff yet</h3>
  <p class="diff-note">The repair ledger will appear after the first action executes.</p>
</section>
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
                    changes.append(
                        (status, row_idx, col, before, after, target)
                    )

    rows = ""
    if changes:
        for status, row_idx, col, before, after, target in changes:
            rows += f"""
<tr>
  <td>{_e(status)}</td>
  <td>{row_idx}</td>
  <td>{_e(col)}</td>
  <td>{_e(before)}</td>
  <td>{_e(after)}</td>
  <td>{_e(target)}</td>
</tr>
"""
    else:
        rows = "<tr><td colspan='6'>No cell changes recorded yet.</td></tr>"

    return f"""
<section class="diff-shell">
  <div class="section-tag">Repair delta</div>
  <div class="diff-stats">
    <div><span>Fixed</span><strong>{fixed}</strong></div>
    <div><span>Regressed</span><strong>{regressed}</strong></div>
    <div><span>Remaining</span><strong>{remaining}</strong></div>
  </div>
  <div class="diff-table-wrap">
    <table class="diff-table">
      <thead>
        <tr>
          <th>Status</th>
          <th>Row</th>
          <th>Column</th>
          <th>Before</th>
          <th>After</th>
          <th>Target</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>
</section>
"""


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
            _status_html(tier, len(env._state), current_acc, f"{agent_type} step {len(rollouts)}/{MAX_UI_STEPS}"),
            dirty[cols].head(10).copy(),
            env._state[cols].head(10).copy(),
            _accuracy_html(acc_before, current_acc),
            _brief_html(
                meta=meta,
                obs=obs,
                agent_type=agent_type,
                note=note,
            ),
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


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #06070a;
  --bg-2: #0d1015;
  --panel: rgba(15, 18, 24, 0.88);
  --panel-2: rgba(21, 25, 32, 0.92);
  --line: rgba(255, 255, 255, 0.08);
  --text: #f5f7fb;
  --muted: #9aa3b2;
  --soft: #6b7280;
  --emerald: #3ddc97;
  --gold: #f2c66d;
  --red: #ff6b6b;
  --shadow: 0 30px 80px rgba(0, 0, 0, 0.35);
}

body {
  background:
    linear-gradient(180deg, #050608 0%, #07090d 40%, #06070a 100%) !important;
  color: var(--text) !important;
  font-family: "Manrope", sans-serif !important;
}

.gradio-container {
  max-width: 1520px !important;
  background: transparent !important;
  padding: 28px 24px 48px !important;
}

.gradio-container * {
  font-family: "Manrope", sans-serif;
}

.gradio-container .monospace,
.gradio-container code,
.gradio-container pre,
.diff-table,
.timeline-components {
  font-family: "IBM Plex Mono", monospace !important;
}

.gradio-html,
.gradio-dataframe,
.gradio-plot {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.hero-shell {
  display: grid;
  grid-template-columns: 1.6fr 1fr;
  gap: 24px;
  padding: 30px 34px;
  border: 1px solid var(--line);
  border-radius: 26px;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015)),
    linear-gradient(180deg, rgba(15,18,24,0.96), rgba(11,13,18,0.92));
  box-shadow: var(--shadow);
}

.hero-kicker,
.section-tag {
  color: var(--gold);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.hero-title {
  margin: 12px 0 8px;
  font-size: 54px;
  line-height: 1;
  letter-spacing: 0;
}

.hero-subtitle {
  margin: 0;
  max-width: 760px;
  color: var(--muted);
  font-size: 18px;
  line-height: 1.65;
}

.hero-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 24px;
}

.hero-pill {
  padding: 10px 14px;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 999px;
  background: rgba(255,255,255,0.04);
  color: var(--text);
  font-size: 13px;
  font-weight: 600;
}

.hero-panel {
  padding: 22px;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
}

.hero-panel-label {
  color: var(--soft);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.16em;
}

.hero-panel-title {
  margin-top: 12px;
  font-size: 22px;
  font-weight: 700;
  line-height: 1.3;
}

.hero-panel-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-top: 22px;
}

.hero-panel-grid span,
.benchmark-note,
.brief-caption,
.metric-note,
.timeline-note,
.diff-note {
  color: var(--muted);
  font-size: 13px;
}

.hero-panel-grid strong,
.metric-grid strong,
.brief-grid strong,
.status-item strong,
.diff-stats strong {
  display: block;
  margin-top: 4px;
  font-size: 16px;
}

.control-shell,
.surface-shell {
  padding: 24px;
  border: 1px solid var(--line);
  border-radius: 22px;
  background: var(--panel);
  box-shadow: var(--shadow);
}

.surface-shell {
  background: linear-gradient(180deg, rgba(16,18,24,0.96), rgba(12,14,19,0.94));
}

.pane-heading {
  margin: 0 0 8px;
  font-size: 28px;
  line-height: 1.1;
}

.pane-copy {
  margin: 0 0 20px;
  color: var(--muted);
  line-height: 1.7;
}

.guide-grid {
  display: grid;
  gap: 10px;
  margin: 14px 0 24px;
}

.guide-card {
  padding: 14px 16px;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  background: rgba(255,255,255,0.03);
}

.guide-card strong {
  display: block;
  margin-bottom: 4px;
}

.guide-card span {
  color: var(--muted);
  font-size: 13px;
}

.gradio-radio {
  gap: 10px !important;
}

.gradio-radio label {
  border: 1px solid rgba(255,255,255,0.08) !important;
  background: rgba(255,255,255,0.035) !important;
  border-radius: 18px !important;
  padding: 14px 16px !important;
}

.gradio-radio label.selected {
  background: rgba(61, 220, 151, 0.12) !important;
  border-color: rgba(61, 220, 151, 0.35) !important;
}

button.primary {
  min-height: 56px !important;
  border-radius: 16px !important;
  border: none !important;
  background: linear-gradient(135deg, #f5f7fb, #dfe5f0) !important;
  color: #0b0d12 !important;
  font-size: 15px !important;
  font-weight: 800 !important;
  box-shadow: 0 18px 35px rgba(245, 247, 251, 0.12) !important;
}

button.primary:hover {
  filter: brightness(1.03);
  transform: translateY(-1px);
}

.status-shell,
.metric-shell,
.brief-shell,
.timeline-shell,
.diff-shell,
.benchmark-shell {
  padding: 22px;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px;
  background: var(--panel-2);
}

.status-shell,
.diff-stats,
.metric-grid,
.brief-grid,
.benchmark-footer {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.status-item span,
.metric-grid span,
.brief-grid span,
.diff-stats span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

.metric-title,
.brief-title,
.timeline-title,
.diff-title,
.section-title {
  margin: 8px 0 14px;
  font-size: 24px;
  line-height: 1.2;
}

.metric-track {
  height: 8px;
  margin-top: 18px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  overflow: hidden;
}

.metric-fill {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, var(--emerald), #8af0c0);
}

.good {
  color: var(--emerald);
}

.bad {
  color: var(--red);
}

.brief-hint,
.brief-note {
  margin: 10px 0 0;
  color: var(--text);
  line-height: 1.6;
}

.timeline-row {
  display: grid;
  grid-template-columns: 120px 1fr;
  gap: 16px;
  padding: 16px 0;
  border-top: 1px solid rgba(255,255,255,0.06);
}

.timeline-row:first-of-type {
  border-top: none;
}

.timeline-step {
  color: var(--gold);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.timeline-head {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: baseline;
}

.timeline-reward {
  font-weight: 800;
}

.timeline-meta {
  margin-top: 6px;
  color: var(--muted);
  font-size: 13px;
}

.timeline-reason {
  margin-top: 10px;
  font-size: 15px;
  line-height: 1.7;
}

.timeline-components {
  margin-top: 10px;
  color: var(--muted);
  font-size: 12px;
}

.diff-table-wrap {
  overflow-x: auto;
}

.diff-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 18px;
  font-size: 12px;
}

.diff-table th,
.diff-table td {
  text-align: left;
  padding: 12px 10px;
  border-top: 1px solid rgba(255,255,255,0.06);
}

.diff-table th {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 11px;
}

.benchmark-row {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: baseline;
  padding: 14px 0;
  border-top: 1px solid rgba(255,255,255,0.06);
}

.benchmark-row:first-of-type {
  border-top: none;
}

.benchmark-name {
  font-size: 16px;
  font-weight: 700;
}

.benchmark-value {
  font-size: 18px;
  font-weight: 800;
}

.benchmark-muted {
  color: var(--muted);
}

.benchmark-good {
  color: var(--emerald);
}

.benchmark-footer {
  margin-top: 18px;
}

.benchmark-footer span {
  color: var(--muted);
  font-size: 13px;
}

.benchmark-footer strong {
  color: var(--text);
}

.table-wrap,
.gradio-dataframe,
.gradio-plot {
  border-radius: 18px !important;
}

.gradio-dataframe {
  border: 1px solid rgba(255,255,255,0.08) !important;
  background: rgba(255,255,255,0.02) !important;
}

@media (max-width: 1100px) {
  .hero-shell {
    grid-template-columns: 1fr;
  }

  .status-shell,
  .diff-stats,
  .metric-grid,
  .brief-grid,
  .benchmark-footer {
    grid-template-columns: 1fr 1fr;
  }

  .timeline-row {
    grid-template-columns: 1fr;
  }
}
"""


def build_demo():
    available = available_agent_choices()
    default_agent = "Heuristic Surgeon"
    if "Live GRPO Model" in available:
        default_agent = "Live GRPO Model"

    with gr.Blocks(title="DataForge Arena") as demo:
        state = gr.State(_new_state())

        hero = gr.HTML(_hero_html())

        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                gr.HTML(
                    """
<section class="control-shell">
  <div class="section-tag">Control room</div>
  <h2 class="pane-heading">Run a live repair episode</h2>
  <p class="pane-copy">
    Choose a repair policy, select corruption depth, and watch every reward-bearing action land on the table.
  </p>
  <div class="guide-grid">
    <div class="guide-card"><strong>Naive Baseline</strong><span>Greedy cell cleanup with no schema reasoning.</span></div>
    <div class="guide-card"><strong>Heuristic Surgeon</strong><span>Rule-based path that tracks constraint classes.</span></div>
    <div class="guide-card"><strong>Live GRPO Model</strong><span>Checkpoint-backed inference when weights are present.</span></div>
  </div>
</section>
"""
                )
                agent_pick = gr.Radio(
                    choices=available,
                    value=default_agent,
                    label="Repair policy",
                )
                tier_pick = gr.Radio(
                    choices=list(TIER_LABELS.keys()),
                    value="Tier 1",
                    label="Complexity",
                )
                new_btn = gr.Button("Seed New Scenario", variant="primary", elem_classes=["primary"])
                run_btn = gr.Button("Execute Repair Policy", variant="primary", elem_classes=["primary"])
                benchmark = gr.HTML(_benchmark_html())
                reward_plot = gr.LinePlot(
                    value=get_training_data(),
                    x="step",
                    y="total_reward",
                    title="Training reward trajectory",
                    height=220,
                    x_title="Step",
                    y_title="Reward",
                    tooltip=["step", "total_reward"],
                )

            with gr.Column(scale=8):
                status_html = gr.HTML(_status_html(1, None, None, "Generate a scenario"))
                with gr.Row():
                    accuracy_html = gr.HTML(_accuracy_html(None, None))
                    brief_html = gr.HTML(_brief_html())
                with gr.Row():
                    dirty_view = gr.Dataframe(
                        label="Corrupted dataset slice",
                        interactive=False,
                        wrap=False,
                    )
                    repaired_view = gr.Dataframe(
                        label="Repaired state",
                        interactive=False,
                        wrap=False,
                    )
                timeline_html = gr.HTML(_empty_timeline_html())
                diff_html = gr.HTML(_diff_html(None, None, None))

        def on_generate(tier_value, session_value):
            return generate_episode(tier_value, session_value)

        def on_execute(agent_value, tier_value, session_value):
            yield from simulate_with_repaired(agent_value, tier_value, session_value)

        new_btn.click(
            fn=on_generate,
            inputs=[tier_pick, state],
            outputs=[
                status_html,
                dirty_view,
                repaired_view,
                accuracy_html,
                brief_html,
                timeline_html,
                diff_html,
                state,
            ],
        )

        run_btn.click(
            fn=on_execute,
            inputs=[agent_pick, tier_pick, state],
            outputs=[
                status_html,
                dirty_view,
                repaired_view,
                accuracy_html,
                brief_html,
                timeline_html,
                diff_html,
                state,
            ],
        )

        demo.load(
            fn=_load_initial_view,
            outputs=[
                hero,
                benchmark,
                reward_plot,
                status_html,
                dirty_view,
                repaired_view,
                accuracy_html,
                brief_html,
                timeline_html,
                diff_html,
                state,
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
