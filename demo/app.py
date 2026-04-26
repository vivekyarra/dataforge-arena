from __future__ import annotations

import html
import json
import logging
import os
import sys
import threading
import time
import warnings
import math
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

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT_DIR, "data", "healthcare_clean.csv")
LOG_PATH   = os.path.join(ROOT_DIR, "logs", "training_log.csv")
EVAL_PATH  = os.path.join(ROOT_DIR, "eval", "results.json")
HEUR_PATH  = os.path.join(ROOT_DIR, "eval", "heuristic_results.json")
MODEL_PATH = os.path.join(ROOT_DIR, "outputs", "dataforge-surgeon")

clean_data   = pd.read_csv(DATA_PATH)
rc           = RewardComputer()
llm_pipeline = None
llm_lock     = threading.Lock()

logger = logging.getLogger("dataforge.demo")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(h)
logger.setLevel(os.getenv("DATAFORGE_LOG_LEVEL", "INFO"))
logger.propagate = False


# ══════════════════════════════════════════════════════════════════════════════
#  INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def _e(v) -> str:
    return html.escape(str(v), quote=True)

def _new_state():
    return {"dirty": None, "gt": None, "meta": None, "tier": 1}

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

def _align_dup_gt(dirty, gt, meta):
    if meta.get("tool") == "duplicate_row_mutate" and len(dirty) > len(gt):
        src = meta.get("row", 0)
        if src < len(gt):
            return pd.concat([gt, gt.iloc[[src]]], ignore_index=True)
    return gt

def _build_env(dirty, gt, tier):
    c = Corruptor(); c.force_tier(tier)
    env = DataForgeEnv(corruptor=c, schema=HEALTHCARE_SCHEMA, clean_data=clean_data)
    acc = rc._field_accuracy(dirty, gt)
    env._state          = dirty.copy()
    env._ground_truth   = gt.copy()
    env._original_dirty = dirty.copy()
    env._prev_accuracy  = acc
    env._starting_accuracy = acc
    env._step_count     = 0
    env._action_log     = []
    env._episode_rewards = []
    env._episode_start  = time.time()
    return env, acc

def _read_json(path):
    try:
        with open(path, "r") as f: return json.load(f)
    except: return {}

def get_training_data():
    try:
        df = pd.read_csv(LOG_PATH)
        if len(df): return df
    except: pass
    return pd.DataFrame({"step":[0],"total_reward":[0],"difficulty":[1]})

def _fmt_pp(v):
    return f"{float(v)*100:+.2f} pp" if v is not None else "—"

def _agent_label(t):
    return {"Naive Baseline":"Naive Baseline","Heuristic Surgeon":"Heuristic Surgeon"}.get(t,"Live GRPO")

def load_llm():
    global llm_pipeline
    if not local_model_available():
        return False, f"No checkpoint found at {MODEL_PATH}"
    with llm_lock:
        if llm_pipeline is not None: return True, "ok"
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
            return llm_pipeline(messages, max_new_tokens=96, temperature=0.1,
                                do_sample=False, num_return_sequences=1)


# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def _ring_svg(value: float | None, label: str, tone: str = "neutral",
              size: int = 108, sw: int = 9) -> str:
    """Animated SVG accuracy ring."""
    r = (size / 2) - sw
    circ = 2 * math.pi * r
    colors = {
        "good":    "#00ff88",
        "bad":     "#ff3d57",
        "warn":    "#ffb020",
        "neutral": "rgba(255,255,255,0.22)",
    }
    col = colors.get(tone, colors["neutral"])
    if value is not None:
        filled = circ * min(max(value, 0), 1)
        gap    = circ - filled
        da     = f"{filled:.2f} {gap:.2f}"
        txt    = f"{value:.1%}"
    else:
        da  = f"0 {circ:.2f}"
        txt = "—"
    cx = cy = size / 2
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
    stroke="rgba(255,255,255,0.06)" stroke-width="{sw}"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
    stroke="{col}" stroke-width="{sw}"
    stroke-dasharray="{da}" stroke-linecap="round"
    transform="rotate(-90 {cx} {cy})"
    style="transition: stroke-dasharray .8s cubic-bezier(.4,0,.2,1);"/>
  <text x="{cx}" y="{cy - 6}" text-anchor="middle" fill="white"
    font-size="16" font-weight="600"
    font-family="'DM Mono',monospace">{_e(txt)}</text>
  <text x="{cx}" y="{cy + 10}" text-anchor="middle"
    fill="rgba(255,255,255,0.35)" font-size="8"
    font-family="'DM Mono',monospace" letter-spacing="1.5">{_e(label.upper())}</text>
</svg>"""


def _spark_bars(values: list[float], color: str = "#00ff88", height: int = 28) -> str:
    """Tiny inline sparkbar from a list of floats."""
    if not values: return ""
    mn, mx = min(values), max(values)
    span = mx - mn or 1
    w = 3
    gap = 2
    total_w = len(values) * (w + gap) - gap
    bars = []
    for i, v in enumerate(values):
        h = max(2, int((v - mn) / span * height))
        x = i * (w + gap)
        y = height - h
        bars.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="1" fill="{color}" opacity="0.85"/>')
    return (f'<svg width="{total_w}" height="{height}" viewBox="0 0 {total_w} {height}" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(bars)}</svg>')


def _stat_inline(label: str, value: str, tone: str = "neutral") -> str:
    col = {"good": "#00ff88", "bad": "#ff3d57", "warn": "#ffb020"}.get(tone, "rgba(255,255,255,0.7)")
    return (f"<div class='si'>"
            f"<div class='si-label'>{_e(label)}</div>"
            f"<div class='si-value' style='color:{col}'>{_e(value)}</div>"
            f"</div>")


def _cell(v) -> str:
    if pd.isna(v): return "<span class='null-v'>NULL</span>"
    return _e(v)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _topbar_html() -> str:
    ev   = _read_json(EVAL_PATH)
    base = _read_json(HEUR_PATH)
    ckpt = local_model_available()
    ha   = base.get("surgeon_advantage_accuracy_delta")
    ga   = ev.get("surgeon_advantage_accuracy_delta")
    return f"""
<div class="topbar">
  <div class="tb-brand">
    <div class="tb-dot"></div>
    <span class="tb-name">DataForge Arena</span>
    <span class="tb-tag">RL Data Repair</span>
  </div>
  <div class="tb-stats">
    <div class="tb-stat">
      <span class="tb-stat-label">Heuristic</span>
      <span class="tb-stat-val {'tb-good' if ha and ha>=0 else ''}">{_fmt_pp(ha)}</span>
    </div>
    <div class="tb-divider"></div>
    <div class="tb-stat">
      <span class="tb-stat-label">GRPO</span>
      <span class="tb-stat-val {'tb-good' if ga and ga>=0 else ''}">{_fmt_pp(ga)}</span>
    </div>
    <div class="tb-divider"></div>
    <div class="tb-stat">
      <span class="tb-stat-label">Model</span>
      <span class="tb-stat-val {'tb-good' if ckpt else 'tb-dim'}">{'Ready' if ckpt else 'Gated'}</span>
    </div>
  </div>
  <div class="tb-live">
    <span class="tb-live-dot"></span>
    <span>Live</span>
  </div>
</div>"""


def _empty_rollout() -> str:
    return """
<div class="empty-center">
  <svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
    <circle cx="20" cy="20" r="18" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="2"/>
    <path d="M14 20 L18 24 L26 16" stroke="rgba(255,255,255,0.15)" stroke-width="2"
      fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <div class="empty-label">Generate a scenario to begin</div>
</div>"""


def _scenario_stats_html(meta, acc_before, total_errors, tier) -> str:
    tier_colors = {1: "#00ff88", 2: "#ffb020", 3: "#ff3d57"}
    col = tier_colors.get(tier, "#fff")
    tier_labels = {1: "T1 Simple", 2: "T2 Cluster", 3: "T3 Relational"}
    tool = meta.get("tool", "?")
    return f"""
<div class="scenario-bar">
  <div class="sb-pill" style="color:{col}; border-color:{col}44;">
    <span class="sb-dot" style="background:{col};"></span>
    {tier_labels.get(tier, f'T{tier}')}
  </div>
  <div class="sb-facts">
    {_stat_inline("Health", f"{acc_before:.1%}", "warn" if acc_before < 0.95 else "good")}
    {_stat_inline("Issues", str(total_errors), "bad" if total_errors else "good")}
    {_stat_inline("Type", tool[:22], "warn")}
  </div>
</div>"""


def _accuracy_display(acc_before, acc_after) -> str:
    """Side-by-side rings + delta bar."""
    delta = (acc_after - acc_before) if acc_after is not None else None
    before_ring = _ring_svg(acc_before, "Before", "neutral")
    after_tone  = ("good" if (delta or 0) >= 0 else "bad") if acc_after is not None else "neutral"
    after_ring  = _ring_svg(acc_after,  "After",  after_tone)
    if delta is not None:
        d_col = "#00ff88" if delta >= 0 else "#ff3d57"
        d_sym = "↑" if delta >= 0 else "↓"
        delta_html = f"""
        <div class="acc-delta" style="color:{d_col};">
          <div class="acc-delta-sym">{d_sym}</div>
          <div class="acc-delta-val">{delta*100:+.2f} pp</div>
          <div class="acc-delta-label">delta</div>
        </div>"""
    else:
        delta_html = "<div class='acc-delta-empty'>—</div>"
    return f"""
<div class="acc-rings">
  <div class="ring-wrap">{before_ring}</div>
  {delta_html}
  <div class="ring-wrap">{after_ring}</div>
</div>"""


def _diff_html(original, current, gt) -> str:
    if original is None or current is None or gt is None:
        return "<div class='diff-empty'>No diff available yet.</div>"

    cols  = [c for c in current.columns if c != "_is_deleted"]
    rlim  = min(len(current), len(gt))
    fixed = reg = rem = 0
    changed_rows  = []
    broken_rows   = []

    for ri in range(rlim):
        for col in cols:
            bef    = original.at[ri, col]
            aft    = current.at[ri, col]
            tgt    = gt.at[ri, col]
            bef_ok = rc._values_match(bef, tgt)
            aft_ok = rc._values_match(aft, tgt)
            if not aft_ok:
                rem += 1
                if len(broken_rows) < 6:
                    broken_rows.append((ri, col, _cell(aft), _cell(tgt)))
            if not rc._values_match(bef, aft):
                if not bef_ok and aft_ok:   fixed += 1; badge, bc = "Fixed",     "bf"
                elif bef_ok and not aft_ok: reg   += 1; badge, bc = "Regressed", "br"
                else:                                    badge, bc = "Shifted",   "bs"
                if len(changed_rows) < 6:
                    changed_rows.append((badge, bc, ri, col, _cell(bef), _cell(aft), _cell(tgt)))

    def rows_changed(rows):
        if not rows: return "<tr><td colspan='6' class='dt-empty'>No edits yet</td></tr>"
        return "".join(
            f"<tr><td><span class='dbadge {bc}'>{badge}</span></td>"
            f"<td>{ri}</td><td>{_e(col)}</td><td>{bef}</td><td>{aft}</td><td>{tgt}</td></tr>"
            for badge,bc,ri,col,bef,aft,tgt in rows)

    def rows_broken(rows):
        if not rows: return "<tr><td colspan='4' class='dt-empty'>All aligned ✓</td></tr>"
        return "".join(
            f"<tr><td>{ri}</td><td>{_e(col)}</td><td>{aft}</td><td>{tgt}</td></tr>"
            for ri,col,aft,tgt in rows)

    fc = "#00ff88" if fixed else "rgba(255,255,255,0.3)"
    rc2 = "#ff3d57" if reg   else "#00ff88"
    rm  = "#ffb020" if rem   else "#00ff88"

    return f"""
<div class="diff-root">
  <div class="diff-stats-row">
    <div class="dsr-item"><span class="dsr-val" style="color:{fc}">{fixed}</span><span class="dsr-label">Fixed</span></div>
    <div class="dsr-sep"></div>
    <div class="dsr-item"><span class="dsr-val" style="color:{rc2}">{reg}</span><span class="dsr-label">Regressed</span></div>
    <div class="dsr-sep"></div>
    <div class="dsr-item"><span class="dsr-val" style="color:{rm}">{rem}</span><span class="dsr-label">Remaining</span></div>
  </div>
  <div class="diff-grid">
    <div class="diff-panel">
      <div class="dp-head">Changed Cells</div>
      <div class="dp-scroll">
        <table class="dt"><thead><tr><th>Status</th><th>Row</th><th>Col</th><th>Before</th><th>After</th><th>Target</th></tr></thead>
        <tbody>{rows_changed(changed_rows)}</tbody></table>
      </div>
    </div>
    <div class="diff-panel">
      <div class="dp-head">Still Broken</div>
      <div class="dp-scroll">
        <table class="dt"><thead><tr><th>Row</th><th>Col</th><th>Current</th><th>Target</th></tr></thead>
        <tbody>{rows_broken(broken_rows)}</tbody></table>
      </div>
    </div>
  </div>
</div>"""


def _rollout_html(rollouts, dirty, current, gt, acc_before, agent_type,
                  total_steps=5, pending_step=None) -> tuple:
    acc_after    = rc._field_accuracy(current, gt)
    delta        = acc_after - acc_before
    total_reward = sum(r.get("reward",0) for r in rollouts)
    done_steps   = len(rollouts)
    pct          = min(100, (done_steps + (0.4 if pending_step else 0)) / total_steps * 100)

    last = rollouts[-1] if rollouts else {}
    lr   = last.get("reasoning",""); lr = lr[:64]+"…" if len(lr)>64 else lr
    lviol= last.get("violation_type","")
    ltool= last.get("tool_name","")

    # Trajectory rows
    traj = ""
    for i, r in enumerate(rollouts):
        rew   = r.get("reward", 0)
        win   = rew >= 0
        rsn   = r.get("reasoning",""); rsn = rsn[:52]+"…" if len(rsn)>52 else rsn
        tnm   = r.get("tool_name","?")
        rowid = r.get("row_id","?")
        coln  = r.get("column_name","?")
        vt    = r.get("violation_type","")
        traj += f"""
<div class="tr-row {'tr-win' if win else 'tr-loss'}">
  <span class="tr-n">{'✓' if win else '✗'}{i+1:02d}</span>
  <span class="tr-rsn">{_e(rsn)}</span>
  <span class="tr-tool">{_e(tnm)}</span>
  <span class="tr-loc">r{rowid}/{_e(coln)}</span>
  <span class="tr-rew {'tr-rew-pos' if win else 'tr-rew-neg'}">{rew:+.2f}</span>
</div>"""

    # Reward DNA
    comps = last.get("components",{}) if rollouts else {}
    dna_labels = {"accuracy_delta":"Accuracy","constraint_alignment":"Constraint",
                  "schema_alignment":"Schema","outlier_targeting":"Outlier",
                  "reasoning_quality":"Reasoning","parse_bonus":"Parse","anti_hack":"Anti-Hack"}
    scale = max([abs(float(v)) for v in comps.values() if isinstance(v,(int,float))]+[0.01])
    dna_rows = ""
    for k, lab in dna_labels.items():
        v   = float(comps.get(k,0))
        pct2= min(100, abs(v)/scale*100)
        pos = v>=0
        dna_rows += f"""
<div class="dna-r">
  <span class="dna-lbl">{_e(lab)}</span>
  <div class="dna-track"><span class="dna-fill {'df-p' if pos else 'df-n'}" style="width:{pct2:.0f}%"></span></div>
  <span class="dna-v {'dv-p' if pos else 'dv-n'}">{v:+.2f}</span>
</div>"""

    d_col = "#00ff88" if delta >= 0 else "#ff3d57"

    html_out = f"""
<div class="ro-root">
  <div class="ro-progress">
    <div class="ro-prog-meta">
      <span class="ro-prog-label">Step {done_steps}/{total_steps}</span>
      <span class="ro-prog-agent">{_e(_agent_label(agent_type))}</span>
      <span class="ro-prog-state">{'Running…' if pending_step else 'Complete'}</span>
    </div>
    <div class="ro-track"><span class="ro-fill" style="width:{pct:.1f}%"></span></div>
  </div>

  <div class="ro-metrics">
    <div class="rom-item">
      <div class="rom-val" style="color:{d_col}">{delta*100:+.2f}</div>
      <div class="rom-label">pp delta</div>
    </div>
    <div class="rom-sep"></div>
    <div class="rom-item">
      <div class="rom-val {'rom-pos' if total_reward>=0 else 'rom-neg'}">{total_reward:+.2f}</div>
      <div class="rom-label">reward</div>
    </div>
    <div class="rom-sep"></div>
    <div class="rom-item">
      <div class="rom-val">{done_steps}</div>
      <div class="rom-label">calls</div>
    </div>
  </div>

  {f'''<div class="ro-causal">
    <div class="rc-label">Last reasoning</div>
    <div class="rc-text">"{_e(lr)}"</div>
    <div class="rc-tags">
      {f'<span class="rc-tag rc-viol">{_e(lviol)}</span>' if lviol else ''}
      {f'<span class="rc-tag rc-tool">{_e(ltool)}</span>' if ltool else ''}
    </div>
  </div>''' if lr else ''}

  <div class="ro-traj">
    <div class="rot-head">Trajectory</div>
    <div class="rot-list">
      {traj if traj else '<div class="rot-empty">No steps yet</div>'}
    </div>
  </div>

  {f'<div class="ro-dna"><div class="dna-head">Reward DNA</div>{dna_rows}</div>' if dna_rows else ''}
</div>"""

    diff_out  = _diff_html(dirty, current, gt)
    acc_disp  = _accuracy_display(acc_before, acc_after)
    return html_out, acc_disp, diff_out


def _benchmark_html() -> str:
    ev   = _read_json(EVAL_PATH)
    base = _read_json(HEUR_PATH)
    training = _get_training_summary()
    ha = base.get("surgeon_advantage_accuracy_delta")
    ga = ev.get("surgeon_advantage_accuracy_delta")
    scale = max(abs(ha or 0), abs(ga or 0), 0.01)

    def lane(label, val, detail):
        if val is None:
            pct, col, vtxt = 15, "rgba(255,255,255,0.2)", "Pending"
        else:
            pct  = 15 + abs(val)/scale * 85
            col  = "#00ff88" if val >= 0 else "#ff3d57"
            vtxt = _fmt_pp(val)
        return f"""
<div class="bm-lane">
  <div class="bm-meta">
    <span class="bm-name">{_e(label)}</span>
    <span class="bm-val" style="color:{col}">{_e(vtxt)}</span>
  </div>
  <div class="bm-track"><span class="bm-bar" style="width:{pct:.1f}%; background:{col}44; border-right: 2px solid {col};"></span></div>
  <div class="bm-detail">{_e(detail)}</div>
</div>"""

    lr  = training.get("latest_reward")
    ia  = training.get("invalid_action_rate")
    lr_txt = f"{lr:+.2f}" if lr is not None else "—"
    ia_txt = f"{ia:.1f}%" if ia is not None else "—"

    return f"""
<div class="bm-root">
  <div class="bm-header">Benchmark Race</div>
  {lane("Heuristic Surgeon", ha, "Rule-based baseline")}
  {lane("GRPO Checkpoint",   ga, "Trained checkpoint")}
  <div class="bm-foot">
    <span>Latest reward <b>{lr_txt}</b></span>
    <span>Invalid actions <b>{ia_txt}</b></span>
  </div>
</div>"""


def _arch_html() -> str:
    items = [
        ("↗", "Observe",  "Schema, suspect rows, and recent actions → structured prompt."),
        ("⚡", "Act",      "One constrained JSON repair action per step."),
        ("◈",  "Score",   "Accuracy delta · tool logic · efficiency · anti-shortcut."),
        ("↑",  "Escalate","Simple nulls → cluster → relational failures."),
    ]
    cards = "".join(f"""
<div class="arch-c">
  <div class="arch-icon">{icon}</div>
  <div class="arch-title">{_e(t)}</div>
  <div class="arch-desc">{_e(d)}</div>
</div>""" for icon,t,d in items)
    return f"<div class='arch-grid'>{cards}</div>"


def _mode_banner() -> str:
    ckpt = local_model_available()
    col  = "#00ff88" if ckpt else "rgba(255,255,255,0.3)"
    msg  = "GRPO checkpoint detected — live inference enabled." if ckpt else "No local checkpoint. Baseline + heuristic paths active."
    return f"""
<div class="mode-b" style="border-color:{'rgba(0,255,136,0.2)' if ckpt else 'rgba(255,255,255,0.07)'}">
  <span class="mode-dot" style="background:{col};{'animation:pulse 2s infinite' if ckpt else ''}"></span>
  <span class="mode-txt">{_e(msg)}</span>
</div>"""


def _get_training_summary():
    df = get_training_data()
    s  = {"parse_success":None,"parse_first":None,"parse_last":None,
          "invalid_action_rate":None,"tiers":"—","latest_reward":None,"best_reward":None}
    if df.empty: return s
    if "parse_success_rate" in df:
        v = pd.to_numeric(df["parse_success_rate"], errors="coerce").dropna()
        if len(v):
            s["parse_success"] = float(v.mean()*100)
            s["parse_first"]   = float(v.iloc[0]*100)
            s["parse_last"]    = float(v.iloc[-1]*100)
    if "invalid_action_rate" in df:
        v = pd.to_numeric(df["invalid_action_rate"], errors="coerce").dropna()
        if len(v): s["invalid_action_rate"] = float(v.iloc[-1]*100)
    if "difficulty" in df:
        t = sorted(pd.to_numeric(df["difficulty"], errors="coerce").dropna().astype(int).unique())
        if t: s["tiers"] = f"{t[0]}–{t[-1]}" if len(t)>1 else str(t[0])
    if "total_reward" in df:
        v = pd.to_numeric(df["total_reward"], errors="coerce").dropna()
        if len(v):
            s["latest_reward"] = float(v.iloc[-1])
            s["best_reward"]   = float(v.max())
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def heuristic_surgeon(state, gt):
    cols = [c for c in state.columns if c != "_is_deleted"]
    for ri in range(min(len(state), len(gt))):
        for ci, col in enumerate(cols):
            cell  = state.at[ri, col]
            gtcel = gt.at[ri, col]
            if pd.isna(cell) and pd.notna(gtcel):
                t = HEALTHCARE_SCHEMA.get(col,{}).get("type","str")
                tid = 0 if t in ("int","float") else 1
                return SurgeonAction(reasoning=f"Null in '{col}' → {'IMPUTE_MEDIAN' if tid==0 else 'IMPUTE_MODE'}",
                                     tool_id=tid, column=ci, row_id=ri)
            if pd.notna(cell) and pd.notna(gtcel) and str(cell) != str(gtcel):
                if str(cell).startswith("ERR_"):
                    t = HEALTHCARE_SCHEMA.get(col,{}).get("type","str")
                    tid = 0 if t in ("int","float") else 1
                    return SurgeonAction(reasoning=f"Type error in '{col}'", tool_id=tid, column=ci, row_id=ri)
                return SurgeonAction(reasoning=f"Format error in '{col}'", tool_id=3, column=ci, row_id=ri)
    if len(state) > len(gt):
        return SurgeonAction(reasoning="Duplicate row → DELETE_ROW", tool_id=4, column=0, row_id=len(state)-1)
    return SurgeonAction(reasoning="No errors detected", tool_id=7, column=0, row_id=0)


def generate_episode(tier, session_state):
    session_state = dict(session_state or _new_state())
    tier = int(tier)
    c = Corruptor(); c.force_tier(tier)
    sample = clean_data.sample(n=min(50, len(clean_data))).reset_index(drop=True)
    dirty, gt, meta = c.generate_episode(sample)
    gt = _align_dup_gt(dirty, gt, meta)
    session_state.update({"dirty":dirty.copy(),"gt":gt.copy(),"meta":meta,"tier":tier})

    cols   = [c for c in dirty.columns if c != "_is_deleted"]
    disp   = dirty[cols].head(8).copy()
    _, tot = summarize_corruption(dirty[cols], HEALTHCARE_SCHEMA)
    acc    = rc._field_accuracy(dirty, gt)
    stats  = _scenario_stats_html(meta, acc, tot, tier)
    return disp, stats, session_state


def simulate_agent(agent_type, session_state):
    session_state = dict(session_state or _new_state())
    if session_state.get("dirty") is None:
        yield (_empty_rollout(),
               _accuracy_display(None, None),
               _diff_html(None, None, None),
               session_state)
        return

    dirty = session_state["dirty"].copy()
    gt    = session_state["gt"].copy()
    tier  = int(session_state.get("tier", 1))
    env, acc_before = _build_env(dirty, gt, tier)
    cols  = [c for c in env._state.columns if c != "_is_deleted"]
    rollouts = []
    MAX  = 5

    for step_idx in range(MAX):
        ro, acc_d, diff = _rollout_html(rollouts, dirty, env._state, gt, acc_before,
                                        agent_type, MAX, step_idx+1)
        yield ro, acc_d, diff, session_state

        # ── Pick action
        if agent_type == "Naive Baseline":
            tr = tc = None; tid = 7; rsn = "No errors."
            for ri in range(len(env._state)):
                for ci, col in enumerate(cols):
                    cell = env._state.at[ri, col]
                    if pd.isna(cell):
                        tr,tc,tid,rsn = ri,ci,0,"Null → IMPUTE_MEDIAN"; break
                    if str(cell).startswith("ERR_"):
                        tr,tc,tid,rsn = ri,ci,0,"Type error → IMPUTE_MEDIAN"; break
                if tr is not None: break
            action = SurgeonAction(reasoning=rsn, tool_id=tid,
                                   column=tc if tc is not None else 0,
                                   row_id=tr if tr is not None else 0)

        elif agent_type == "Heuristic Surgeon":
            action = heuristic_surgeon(env._state.copy(), gt)

        else:
            ok, msg = load_llm()
            if not ok:
                yield (f"<div class='err-banner'>{_e(msg)}</div>",
                       _accuracy_display(acc_before, None),
                       _diff_html(dirty, env._state, gt),
                       session_state)
                return
            obs = env._make_observation()
            messages = [
                {"role":"system","content": build_prompt(obs)},
                {"role":"user","content":f"Observation: {obs.model_dump_json()}\nOutput valid JSON only."},
            ]
            try:
                out    = _run_llm(messages)
                raw    = out[0]["generated_text"][-1]["content"]
                action = robust_parse_action(raw, require_fields=True)
            except Exception as exc:
                action = SurgeonAction(reasoning=f"LLM error: {str(exc)[:40]}", tool_id=7, column=0, row_id=0)

        _, total_reward, done, info = env.step(action)
        obs2 = env._make_observation()
        rollouts.append({
            "reasoning":        action.reasoning.replace("EXACT_PARSE: ",""),
            "tool_name":        SURGEON_TOOLS.get(action.tool_id,{"name":"?"})["name"],
            "reward":           total_reward,
            "row_id":           action.row_id,
            "column_name":      cols[action.column] if action.column < len(cols) else "?",
            "components":       info.get("reward_components",{}),
            "violation_type":   getattr(obs2,"violation_type",""),
            "target_cell_hint": getattr(obs2,"target_cell_hint",""),
        })

        ro, acc_d, diff = _rollout_html(rollouts, dirty, env._state, gt, acc_before, agent_type, MAX)
        yield ro, acc_d, diff, session_state
        if done: break


# ══════════════════════════════════════════════════════════════════════════════
#  CSS — MISSION CONTROL
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&family=Bebas+Neue&display=swap');

:root {
  --bg:    #08090e;
  --p1:    rgba(255,255,255,0.028);
  --p2:    rgba(255,255,255,0.055);
  --p3:    rgba(255,255,255,0.09);
  --b0:    rgba(255,255,255,0.055);
  --b1:    rgba(255,255,255,0.10);
  --t1:    #ffffff;
  --t2:    rgba(255,255,255,0.55);
  --t3:    rgba(255,255,255,0.28);
  --g:     #00ff88;
  --r:     #ff3d57;
  --a:     #ffb020;
  --bl:    #3b9eff;
  --mono:  'DM Mono', monospace;
  --sans:  'DM Sans', sans-serif;
  --r6:  6px;
  --r10: 10px;
  --r14: 14px;
}

*, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }

body, .gradio-container {
  background: var(--bg) !important;
  color: var(--t1);
  font-family: var(--sans);
  font-size: 13px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.gradio-container { max-width: 1560px !important; padding: 0 20px 32px !important; }

/* ── Gradio resets ── */
.gradio-container .block, .gradio-container .panel,
.gradio-container .wrap, .gradio-container fieldset {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}
.gradio-container .gap { gap: 14px !important; }
.gradio-container label {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  color: var(--t3) !important;
  text-transform: uppercase;
  letter-spacing: .1em;
}

/* ── Buttons ── */
.gradio-container button {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  border-radius: var(--r10) !important;
  letter-spacing: .06em;
  transition: all .15s !important;
}
.gradio-container button.primary, .gradio-container button[variant='primary'] {
  background: linear-gradient(135deg, #00cc6a 0%, #0066ff 100%) !important;
  border: 0 !important;
  color: #fff !important;
  font-size: 12px !important;
  letter-spacing: .1em !important;
  text-transform: uppercase !important;
  padding: 14px 28px !important;
  box-shadow: 0 0 32px rgba(0,255,136,.22), 0 4px 20px rgba(0,0,0,.4) !important;
}
.gradio-container button.primary:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 0 48px rgba(0,255,136,.32), 0 8px 28px rgba(0,0,0,.5) !important;
}
.gradio-container button.secondary, .gradio-container button[variant='secondary'] {
  background: var(--p2) !important;
  border: 1px solid var(--b0) !important;
  color: var(--t2) !important;
  padding: 11px 18px !important;
}
.gradio-container button.secondary:hover {
  background: var(--p3) !important;
  border-color: rgba(0,255,136,.22) !important;
  color: var(--t1) !important;
}

/* ── Radio ── */
.gradio-container .radio-group { gap: 6px !important; flex-wrap: wrap !important; }
.gradio-container .radio-group label {
  background: var(--p1) !important;
  border: 1px solid var(--b0) !important;
  border-radius: var(--r6) !important;
  padding: 8px 14px !important;
  color: var(--t2) !important;
  font-size: 11px !important;
  text-transform: none !important;
  cursor: pointer;
  transition: all .15s;
}
.gradio-container .radio-group label:has(input:checked) {
  background: rgba(0,255,136,.08) !important;
  border-color: rgba(0,255,136,.3) !important;
  color: var(--g) !important;
}

/* ── Dataframe ── */
.gradio-container .table-wrap, .gradio-container table {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  background: transparent !important;
}
.gradio-container th {
  background: var(--p2) !important;
  color: var(--t3) !important;
  font-size: 9px !important;
  letter-spacing: .08em;
  text-transform: uppercase;
  padding: 8px 10px !important;
  border-bottom: 1px solid var(--b0) !important;
  white-space: nowrap;
}
.gradio-container td {
  color: var(--t2) !important;
  padding: 7px 10px !important;
  border-bottom: 1px solid rgba(255,255,255,.03) !important;
  font-size: 11px !important;
}
.gradio-container tr:hover td { background: rgba(255,255,255,.02) !important; }

/* ── LinePlot ── */
.gradio-container .plot-container {
  background: var(--p1) !important;
  border: 1px solid var(--b0) !important;
  border-radius: var(--r14) !important;
  padding: 12px !important;
}

/* ══════════════════════════════════════════
   CUSTOM COMPONENTS
═══════════════════════════════════════════ */

/* Top bar */
.topbar {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 12px 20px;
  margin: 8px 0 14px;
  background: var(--p1);
  border: 1px solid var(--b0);
  border-radius: var(--r14);
}
.tb-brand { display:flex; align-items:center; gap:10px; }
.tb-dot {
  width:8px; height:8px; border-radius:50%;
  background: var(--g);
  box-shadow: 0 0 8px var(--g);
  animation: pulse 2s infinite;
}
.tb-name {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 22px;
  letter-spacing: .06em;
  color: var(--t1);
  line-height: 1;
}
.tb-tag {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--t3);
  letter-spacing: .1em;
  text-transform: uppercase;
  border: 1px solid var(--b0);
  padding: 2px 8px;
  border-radius: 999px;
}
.tb-stats { display:flex; align-items:center; gap:4px; margin-left:auto; }
.tb-stat { display:flex; flex-direction:column; align-items:center; gap:2px; padding: 0 12px; }
.tb-stat-label { font-family:var(--mono); font-size:9px; color:var(--t3); text-transform:uppercase; letter-spacing:.08em; }
.tb-stat-val   { font-family:var(--mono); font-size:14px; font-weight:500; color:var(--t2); }
.tb-good { color: var(--g) !important; }
.tb-dim  { color: var(--t3) !important; }
.tb-divider { width:1px; height:28px; background:var(--b0); margin:0 4px; }
.tb-live {
  display:flex; align-items:center; gap:6px;
  font-family:var(--mono); font-size:10px; color:var(--g);
  border:1px solid rgba(0,255,136,.2); padding:4px 12px; border-radius:999px;
  margin-left:16px;
}
.tb-live-dot {
  width:6px; height:6px; border-radius:50%;
  background:var(--g); animation:pulse 1.5s infinite;
}

/* Section panels */
.pane {
  background: var(--p1);
  border: 1px solid var(--b0);
  border-radius: var(--r14);
  padding: 18px;
  overflow: hidden;
}
.pane-label {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--t3);
  letter-spacing: .12em;
  text-transform: uppercase;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.pane-label::before {
  content:'';
  display:block; width:14px; height:1px;
  background:var(--b1);
}

/* Scenario bar */
.scenario-bar {
  display:flex; align-items:center; gap:12px;
  flex-wrap:wrap;
  padding:10px 14px;
  background:var(--p2); border:1px solid var(--b0);
  border-radius:var(--r10); margin-bottom:10px;
}
.sb-pill {
  display:inline-flex; align-items:center; gap:6px;
  font-family:var(--mono); font-size:10px; font-weight:500;
  border:1px solid; border-radius:999px; padding:3px 10px;
  white-space:nowrap;
}
.sb-dot { width:5px; height:5px; border-radius:50%; flex-shrink:0; }
.sb-facts { display:flex; align-items:center; gap:16px; flex-wrap:wrap; }

/* Inline stat */
.si { display:flex; flex-direction:column; gap:1px; }
.si-label { font-family:var(--mono); font-size:8px; color:var(--t3); text-transform:uppercase; letter-spacing:.08em; }
.si-value { font-family:var(--mono); font-size:13px; font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:140px; }

/* Mode banner */
.mode-b {
  display:flex; align-items:center; gap:8px;
  padding:9px 14px; background:var(--p1);
  border:1px solid; border-radius:var(--r10); margin-bottom:10px;
}
.mode-dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
.mode-txt { font-size:12px; color:var(--t2); }

/* Accuracy rings */
.acc-rings {
  display:flex; align-items:center; justify-content:center;
  gap:16px; padding:16px 0;
}
.ring-wrap { display:flex; flex-direction:column; align-items:center; }
.acc-delta {
  display:flex; flex-direction:column; align-items:center; gap:2px;
  padding:0 8px;
}
.acc-delta-sym { font-size:20px; font-weight:700; line-height:1; }
.acc-delta-val { font-family:var(--mono); font-size:14px; font-weight:500; }
.acc-delta-label { font-family:var(--mono); font-size:9px; color:var(--t3); text-transform:uppercase; letter-spacing:.08em; }
.acc-delta-empty { font-family:var(--mono); font-size:24px; color:var(--t3); padding:0 16px; }

/* Rollout */
.ro-root { display:flex; flex-direction:column; gap:12px; }

.ro-progress {
  padding:14px 16px;
  background:var(--p2); border:1px solid var(--b0);
  border-radius:var(--r10);
}
.ro-prog-meta {
  display:flex; align-items:center; gap:10px; margin-bottom:8px;
  flex-wrap:wrap;
}
.ro-prog-label { font-family:var(--mono); font-size:16px; font-weight:500; color:var(--t1); }
.ro-prog-agent { font-family:var(--mono); font-size:11px; color:var(--g); margin-left:auto; }
.ro-prog-state { font-family:var(--mono); font-size:10px; color:var(--t3); }
.ro-track { height:3px; background:var(--p3); border-radius:999px; overflow:hidden; }
.ro-fill  {
  display:block; height:100%; border-radius:inherit;
  background:linear-gradient(90deg, var(--g), var(--bl));
  transition: width .5s cubic-bezier(.4,0,.2,1);
}

.ro-metrics {
  display:flex; align-items:center; gap:0;
  background:var(--p1); border:1px solid var(--b0);
  border-radius:var(--r10); overflow:hidden;
}
.rom-item { flex:1; display:flex; flex-direction:column; align-items:center; gap:2px; padding:12px; }
.rom-val  { font-family:var(--mono); font-size:20px; font-weight:500; color:var(--t1); }
.rom-pos  { color:var(--g); }
.rom-neg  { color:var(--r); }
.rom-label{ font-family:var(--mono); font-size:9px; color:var(--t3); text-transform:uppercase; letter-spacing:.08em; }
.rom-sep  { width:1px; height:40px; background:var(--b0); flex-shrink:0; }

.ro-causal {
  padding:12px 14px;
  background:rgba(59,158,255,.05);
  border:1px solid rgba(59,158,255,.14);
  border-radius:var(--r10);
}
.rc-label { font-family:var(--mono); font-size:9px; color:rgba(59,158,255,.7); letter-spacing:.1em; text-transform:uppercase; margin-bottom:5px; }
.rc-text  { font-family:var(--mono); font-size:12px; color:var(--t1); line-height:1.5; margin-bottom:7px; font-style:italic; }
.rc-tags  { display:flex; gap:6px; flex-wrap:wrap; }
.rc-tag   { font-family:var(--mono); font-size:10px; padding:2px 9px; border-radius:999px; border:1px solid; }
.rc-viol  { color:var(--r); border-color:rgba(255,61,87,.25); background:rgba(255,61,87,.08); }
.rc-tool  { color:var(--g); border-color:rgba(0,255,136,.25); background:rgba(0,255,136,.08); }

.ro-traj { display:flex; flex-direction:column; gap:6px; }
.rot-head { font-family:var(--mono); font-size:9px; color:var(--t3); letter-spacing:.1em; text-transform:uppercase; margin-bottom:2px; }
.rot-list { display:flex; flex-direction:column; gap:4px; }
.tr-row {
  display:grid;
  grid-template-columns: 38px 1fr auto auto 56px;
  align-items:center; gap:10px;
  padding:9px 12px;
  border-radius:var(--r6);
  border-left:2px solid transparent;
  animation:fadeUp .2s ease both;
  overflow:hidden;
}
.tr-win  { background:rgba(0,255,136,.05); border-color:var(--g); }
.tr-loss { background:rgba(255,61,87,.05);  border-color:var(--r); }
.tr-n    { font-family:var(--mono); font-size:10px; color:var(--t3); white-space:nowrap; }
.tr-rsn  { font-family:var(--mono); font-size:11px; color:var(--t2); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-style:italic; }
.tr-tool { font-family:var(--mono); font-size:10px; background:var(--p3); color:var(--t2); padding:2px 8px; border-radius:4px; white-space:nowrap; }
.tr-loc  { font-family:var(--mono); font-size:10px; color:var(--t3); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.tr-rew  { font-family:var(--mono); font-size:12px; font-weight:500; text-align:right; white-space:nowrap; }
.tr-rew-pos { color:var(--g); }
.tr-rew-neg { color:var(--r); }
.rot-empty  { padding:14px; text-align:center; color:var(--t3); font-size:11px; }

.ro-dna { display:flex; flex-direction:column; gap:5px; padding:12px 14px; background:var(--p1); border:1px solid var(--b0); border-radius:var(--r10); }
.dna-head { font-family:var(--mono); font-size:9px; color:var(--t3); letter-spacing:.1em; text-transform:uppercase; margin-bottom:6px; }
.dna-r    { display:grid; grid-template-columns:80px 1fr 40px; align-items:center; gap:8px; }
.dna-lbl  { font-family:var(--mono); font-size:10px; color:var(--t3); }
.dna-track{ height:4px; background:var(--p3); border-radius:999px; overflow:hidden; }
.dna-fill { display:block; height:100%; border-radius:inherit; }
.df-p  { background:linear-gradient(90deg,var(--g),var(--bl)); }
.df-n  { background:linear-gradient(90deg,var(--r),var(--a)); }
.dna-v { font-family:var(--mono); font-size:10px; font-weight:500; text-align:right; }
.dv-p  { color:var(--g); }
.dv-n  { color:var(--r); }

/* Diff */
.diff-root { display:flex; flex-direction:column; gap:10px; }
.diff-stats-row {
  display:flex; align-items:center; gap:0;
  background:var(--p1); border:1px solid var(--b0);
  border-radius:var(--r10); overflow:hidden;
}
.dsr-item { flex:1; display:flex; flex-direction:column; align-items:center; gap:2px; padding:11px 12px; }
.dsr-val  { font-family:var(--mono); font-size:22px; font-weight:500; line-height:1; }
.dsr-label{ font-family:var(--mono); font-size:9px; color:var(--t3); text-transform:uppercase; letter-spacing:.08em; }
.dsr-sep  { width:1px; height:36px; background:var(--b0); }

.diff-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
.diff-panel { background:var(--p1); border:1px solid var(--b0); border-radius:var(--r10); overflow:hidden; }
.dp-head    { padding:8px 12px; font-family:var(--mono); font-size:9px; color:var(--t3); text-transform:uppercase; letter-spacing:.1em; background:var(--p2); border-bottom:1px solid var(--b0); }
.dp-scroll  { overflow-x:auto; }
.dt { width:100%; border-collapse:collapse; font-family:var(--mono); font-size:10px; }
.dt th { padding:6px 10px; background:var(--p2); color:var(--t3); font-size:9px; text-transform:uppercase; letter-spacing:.06em; border-bottom:1px solid var(--b0); white-space:nowrap; }
.dt td { padding:6px 10px; color:var(--t2); border-bottom:1px solid rgba(255,255,255,.025); max-width:140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.dt tr:last-child td { border-bottom:none; }
.dt-empty { text-align:center; color:var(--t3) !important; padding:14px !important; }
.dbadge { display:inline-flex; align-items:center; padding:1px 7px; border-radius:999px; font-size:8px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; }
.bf { background:rgba(0,255,136,.1); color:var(--g); }
.br { background:rgba(255,61,87,.1);  color:var(--r); }
.bs { background:rgba(59,158,255,.1); color:var(--bl); }
.null-v { color:var(--a); font-weight:600; }
.diff-empty { padding:20px; color:var(--t3); font-size:11px; text-align:center; }

/* Benchmark */
.bm-root { padding:16px 18px; background:var(--p1); border:1px solid var(--b0); border-radius:var(--r14); }
.bm-header { font-family:var(--mono); font-size:9px; letter-spacing:.12em; text-transform:uppercase; color:var(--t3); margin-bottom:14px; }
.bm-lane { margin-bottom:14px; }
.bm-lane:last-of-type { margin-bottom:0; }
.bm-meta { display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; }
.bm-name { font-family:var(--mono); font-size:12px; font-weight:500; color:var(--t1); }
.bm-val  { font-family:var(--mono); font-size:13px; font-weight:500; }
.bm-track{ height:6px; background:var(--p3); border-radius:999px; overflow:hidden; margin-bottom:3px; }
.bm-bar  { display:block; height:100%; border-radius:inherit; }
.bm-detail { font-size:10px; color:var(--t3); }
.bm-foot { display:flex; gap:14px; flex-wrap:wrap; margin-top:12px; padding-top:12px; border-top:1px solid var(--b0); font-size:11px; color:var(--t3); }
.bm-foot b { color:var(--t2); }

/* Arch */
.arch-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
.arch-c { padding:16px; background:var(--p1); border:1px solid var(--b0); border-radius:var(--r10); }
.arch-icon  { font-size:18px; margin-bottom:8px; }
.arch-title { font-family:var(--mono); font-size:12px; font-weight:500; color:var(--t1); margin-bottom:5px; }
.arch-desc  { font-size:11px; color:var(--t3); line-height:1.5; }

/* Empty / Error */
.empty-center {
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:10px; padding:40px 20px; text-align:center;
}
.empty-label { font-family:var(--mono); font-size:11px; color:var(--t3); }
.err-banner {
  padding:20px; background:rgba(255,61,87,.08);
  border:1px solid rgba(255,61,87,.2); border-radius:var(--r10);
  font-family:var(--mono); font-size:11px; color:var(--r);
}

/* Animations */
@keyframes pulse {
  0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(0,255,136,.4); }
  50%      { opacity:.8; box-shadow:0 0 0 5px rgba(0,255,136,0); }
}
@keyframes fadeUp {
  from { opacity:0; transform:translateY(6px); }
  to   { opacity:1; transform:translateY(0); }
}

/* Responsive */
@media (max-width:1100px) {
  .arch-grid { grid-template-columns:repeat(2,1fr); }
  .tr-row { grid-template-columns:32px 1fr 48px; }
  .tr-tool,.tr-loc { display:none; }
}
@media (max-width:720px) {
  .diff-grid, .arch-grid { grid-template-columns:1fr; }
  .topbar { flex-wrap:wrap; }
  .tb-stats { margin-left:0; }
}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD UI
# ══════════════════════════════════════════════════════════════════════════════

def build_demo():
    choices = available_agent_choices()
    default = "Live GRPO Model" if "Live GRPO Model" in choices else "Heuristic Surgeon"

    with gr.Blocks(title="DataForge Arena", css=CSS, theme=gr.themes.Base()) as demo:
        state = gr.State(_new_state())

        # ── Top bar (auto-refreshed on load)
        topbar = gr.HTML(_topbar_html())

        with gr.Row(equal_height=False):
            # ── LEFT: Input
            with gr.Column(scale=1, min_width=260):
                gr.HTML("<div class='pane-label'>01 · Input</div>")
                with gr.Row():
                    btn1 = gr.Button("⬡ Tier 1", variant="secondary")
                    btn3 = gr.Button("⬡ Tier 3", variant="secondary")
                scenario_stats = gr.HTML("")
                dirty_view = gr.Dataframe(label="", interactive=False, wrap=False, max_rows=8)

            # ── CENTER: Agent
            with gr.Column(scale=2, min_width=420):
                gr.HTML("<div class='pane-label'>02 · Agent</div>")
                mode_inv   = gr.HTML(_mode_banner())
                agent_pick = gr.Radio(choices, value=default, label="Execution Path")
                exec_btn   = gr.Button("▶  Execute Agent", variant="primary", size="lg")
                rollout_out = gr.HTML(_empty_rollout())

            # ── RIGHT: Output
            with gr.Column(scale=1, min_width=280):
                gr.HTML("<div class='pane-label'>03 · Output</div>")
                acc_display  = gr.HTML(_accuracy_display(None, None))
                repaired_view = gr.Dataframe(label="", interactive=False, wrap=False, max_rows=8)
                diff_out     = gr.HTML(_diff_html(None, None, None))

        # ── Bottom: Training + Benchmark
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                gr.HTML("<div class='pane-label'>Training Evidence</div>")
                refresh_btn = gr.Button("↻ Refresh", variant="secondary")
                with gr.Row():
                    reward_plot = gr.LinePlot(x="step", y="total_reward",
                                             title="Reward Curve", x_title="Step", y_title="Reward", height=200)
                    diff_plot   = gr.LinePlot(x="step", y="difficulty",
                                             title="Tier Escalation", x_title="Step", y_title="Tier", height=200)
            with gr.Column(scale=2):
                gr.HTML("<div class='pane-label'>Benchmark</div>")
                bench_html = gr.HTML(_benchmark_html())
                gr.HTML("<div class='pane-label' style='margin-top:14px'>How It Works</div>")
                arch_html  = gr.HTML(_arch_html())

        # ── Handlers
        def on_gen(tier_int, s):
            disp, stats, ns = generate_episode(tier_int, s)
            acc   = rc._field_accuracy(ns["dirty"], ns["gt"])
            cols  = [c for c in ns["dirty"].columns if c != "_is_deleted"]
            _,tot = summarize_corruption(ns["dirty"][cols], HEALTHCARE_SCHEMA)
            return (disp, stats, ns,
                    _empty_rollout(),
                    _accuracy_display(acc, None),
                    None,
                    _diff_html(ns["dirty"], ns["dirty"], ns["gt"]))

        def load_dash():
            df = get_training_data()
            return _topbar_html(), _benchmark_html(), df, df

        gen_outs = [dirty_view, scenario_stats, state,
                    rollout_out, acc_display, repaired_view, diff_out]

        btn1.click(fn=lambda s: on_gen(1, s), inputs=[state], outputs=gen_outs)
        btn3.click(fn=lambda s: on_gen(3, s), inputs=[state], outputs=gen_outs)

        exec_btn.click(
            fn=simulate_agent,
            inputs=[agent_pick, state],
            outputs=[rollout_out, acc_display, diff_out, state],
        )

        # After execution update repaired view — we need to also return repaired_view
        # Wrap simulate_agent to yield repaired data too
        def simulate_with_repaired(agent_type, session_state):
            session_state = dict(session_state or _new_state())
            dirty  = session_state.get("dirty")
            gt     = session_state.get("gt")
            tier   = int(session_state.get("tier", 1))

            if dirty is None:
                yield (_empty_rollout(), _accuracy_display(None,None),
                       _diff_html(None,None,None), None, session_state)
                return

            env, acc_before = _build_env(dirty.copy(), gt.copy(), tier)
            cols     = [c for c in env._state.columns if c != "_is_deleted"]
            rollouts = []
            MAX      = 5

            for step_idx in range(MAX):
                ro, acc_d, diff = _rollout_html(rollouts, dirty, env._state, gt, acc_before,
                                                agent_type, MAX, step_idx+1)
                repaired = env._state[cols].head(8).copy()
                yield ro, acc_d, diff, repaired, session_state

                if agent_type == "Naive Baseline":
                    tr=tc=None; tid=7; rsn="No errors."
                    for ri in range(len(env._state)):
                        for ci, col in enumerate(cols):
                            cell = env._state.at[ri, col]
                            if pd.isna(cell):     tr,tc,tid,rsn=ri,ci,0,"Null→IMPUTE_MEDIAN"; break
                            if str(cell).startswith("ERR_"): tr,tc,tid,rsn=ri,ci,0,"ERR→IMPUTE_MEDIAN"; break
                        if tr is not None: break
                    action = SurgeonAction(reasoning=rsn,tool_id=tid,
                                           column=tc if tc is not None else 0,
                                           row_id=tr if tr is not None else 0)
                elif agent_type == "Heuristic Surgeon":
                    action = heuristic_surgeon(env._state.copy(), gt)
                else:
                    ok,msg = load_llm()
                    if not ok:
                        yield (f"<div class='err-banner'>{_e(msg)}</div>",
                               _accuracy_display(acc_before,None),
                               _diff_html(dirty,env._state,gt), None, session_state)
                        return
                    obs = env._make_observation()
                    msgs = [{"role":"system","content":build_prompt(obs)},
                            {"role":"user","content":f"Observation: {obs.model_dump_json()}\nOutput valid JSON only."}]
                    try:
                        out  = _run_llm(msgs)
                        raw  = out[0]["generated_text"][-1]["content"]
                        action = robust_parse_action(raw, require_fields=True)
                    except Exception as exc:
                        action = SurgeonAction(reasoning=f"LLM error: {str(exc)[:40]}",tool_id=7,column=0,row_id=0)

                _, total_reward, done, info = env.step(action)
                obs2 = env._make_observation()
                rollouts.append({
                    "reasoning":        action.reasoning.replace("EXACT_PARSE:","").strip(),
                    "tool_name":        SURGEON_TOOLS.get(action.tool_id,{"name":"?"})["name"],
                    "reward":           total_reward,
                    "row_id":           action.row_id,
                    "column_name":      cols[action.column] if action.column < len(cols) else "?",
                    "components":       info.get("reward_components",{}),
                    "violation_type":   getattr(obs2,"violation_type",""),
                    "target_cell_hint": getattr(obs2,"target_cell_hint",""),
                })

                ro, acc_d, diff = _rollout_html(rollouts, dirty, env._state, gt, acc_before, agent_type, MAX)
                repaired = env._state[cols].head(8).copy()
                yield ro, acc_d, diff, repaired, session_state
                if done: break

        exec_btn.click(
            fn=simulate_with_repaired,
            inputs=[agent_pick, state],
            outputs=[rollout_out, acc_display, diff_out, repaired_view, state],
        )

        refresh_btn.click(fn=load_dash, outputs=[topbar, bench_html, reward_plot, diff_plot])
        demo.load(fn=load_dash, outputs=[topbar, bench_html, reward_plot, diff_plot])

    return demo


demo = build_demo()

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.queue(default_concurrency_limit=8).launch(server_name=server_name, server_port=server_port)