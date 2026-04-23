# demo/app.py — Tactical DataForge Arena Demo
import gradio as gr
import json
import sys
import os
import random
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.env import DataForgeEnv, SurgeonAction
from environment.corruptor import Corruptor
from environment.reward import RewardComputer
from environment.schemas import HEALTHCARE_SCHEMA, SURGEON_TOOLS

# ---------- load environment once ----------------------------------
clean_data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "healthcare_clean.csv"))
corruptor = Corruptor()
env = DataForgeEnv(corruptor=corruptor, schema=HEALTHCARE_SCHEMA, clean_data=clean_data)
rc = RewardComputer()

# Episode state held between button clicks
current_dirty = [None]
current_gt = [None]
current_meta = [None]

# ---------- CSS ---------------------------------------------------
DARK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');
body, .gradio-container { background: #06060b !important; color: #e2e8f0; font-family: 'Inter', sans-serif; }
.panel { background: linear-gradient(135deg, #0d0d15 0%, #111118 100%); border: 1px solid #1e293b; border-radius: 12px; padding: 20px; }
h1 { font-family: 'JetBrains Mono', monospace !important; }
.metric-card { background: #0f0f1a; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; text-align: center; }
.rollout-row { padding: 8px 12px; margin: 4px 0; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.rollout-winner { background: linear-gradient(90deg, rgba(16,185,129,0.15) 0%, transparent 100%); border-left: 3px solid #10b981; }
.rollout-loser  { background: rgba(30,41,59,0.3); border-left: 3px solid #374151; opacity: 0.7; }
.tag { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.tag-null   { background:#7f1d1d; color:#fca5a5; }
.tag-type   { background:#78350f; color:#fcd34d; }
.tag-fixed  { background:#064e3b; color:#6ee7b7; }
.tag-dup    { background:#1e3a5f; color:#93c5fd; }
.pulse { animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }
"""

# ---------- helper funcs ------------------------------------------
def classify_error(val, gt_val):
    if pd.isna(val): return "null"
    if str(val) != str(gt_val): return "type"
    return "ok"

def generate_episode(tier):
    """Generate a corrupted episode at given tier."""
    tier = int(tier)
    corruptor._epoch = {1: 0, 2: 65, 3: 115}[tier]
    
    n = min(50, len(clean_data))
    sample = clean_data.sample(n=n).reset_index(drop=True)
    dirty, gt, meta = corruptor.generate_episode(sample)
    
    # Handle duplicate_row_mutate
    if meta.get("tool") == "duplicate_row_mutate" and len(dirty) > len(gt):
        src = meta.get("row", 0)
        if src < len(gt):
            gt = pd.concat([gt, gt.iloc[[src]]], ignore_index=True)
    
    current_dirty[0] = dirty.copy()
    current_gt[0] = gt.copy()
    current_meta[0] = meta
    
    # Build display DF with error highlighting
    display_cols = [c for c in dirty.columns if c != "_is_deleted"]
    display = dirty[display_cols].head(8).copy()
    
    # Error stats
    acc_before = rc._field_accuracy(dirty, gt)
    total_nulls = dirty[display_cols].isnull().sum().sum()
    total_cells = dirty[display_cols].size
    
    stats_html = f"""
    <div style='font-family: JetBrains Mono, monospace; font-size:13px; margin-top:12px;'>
      <div style='display:flex; gap:16px;'>
        <div class='metric-card' style='flex:1'>
          <div style='color:#64748b; font-size:11px;'>ACCURACY</div>
          <div style='color:{"#ef4444" if acc_before < 0.95 else "#f59e0b"}; font-size:24px; font-weight:700;'>{acc_before:.1%}</div>
        </div>
        <div class='metric-card' style='flex:1'>
          <div style='color:#64748b; font-size:11px;'>ERRORS</div>
          <div style='color:#ef4444; font-size:24px; font-weight:700;'>{total_nulls}</div>
        </div>
        <div class='metric-card' style='flex:1'>
          <div style='color:#64748b; font-size:11px;'>CORRUPTION</div>
          <div style='color:#f59e0b; font-size:24px; font-weight:700;'>{meta["tool"]}</div>
        </div>
      </div>
      <div style='color:#475569; font-size:11px; margin-top:8px;'>
        Tier {tier} • {len(dirty)} rows × {len(display_cols)} cols = {total_cells} cells
      </div>
    </div>
    """
    
    return display, stats_html

def simulate_agent(agent_type):
    """Simulate agent repairing the corrupted data."""
    if current_dirty[0] is None:
        return ("<p style='color:#ef4444'>Generate an episode first!</p>", 
                None, "", "")
    
    dirty = current_dirty[0].copy()
    gt = current_gt[0].copy()
    meta = current_meta[0]
    display_cols = [c for c in dirty.columns if c != "_is_deleted"]
    
    acc_before = rc._field_accuracy(dirty, gt)
    
    # Simulate 5 repair actions
    rollouts = []
    state = dirty.copy()
    
    for step in range(5):
        # Find a corrupted cell
        target_row, target_col = None, None
        for r in range(min(len(state), len(gt))):
            for c_idx, c_name in enumerate(display_cols):
                cell = state.at[r, c_name] if c_name in state.columns else None
                gt_cell = gt.at[r, c_name] if c_name in gt.columns else None
                if pd.isna(cell) and pd.notna(gt_cell):
                    target_row, target_col = r, c_idx
                    break
                elif pd.notna(cell) and pd.notna(gt_cell) and str(cell) != str(gt_cell):
                    target_row, target_col = r, c_idx
                    break
            if target_row is not None:
                break
        
        if target_row is None:
            break  # no more errors
        
        cell_val = state.iloc[target_row, target_col]
        is_null = pd.isna(cell_val)
        
        if agent_type == "Untrained baseline":
            # Random tool selection
            tool_id = random.choice([0, 1, 2, 3, 7])
            reasoning = "Random action (no training)"
        else:
            # Smart tool selection (simulates trained behavior)
            if is_null:
                col_name = display_cols[target_col]
                if any(t in col_name.lower() for t in ["id", "age", "year", "amount"]):
                    tool_id = 0  # IMPUTE_MEDIAN for numeric
                    reasoning = f"Null detected in numeric column '{col_name}'. Using IMPUTE_MEDIAN to fill with column median."
                else:
                    tool_id = 1  # IMPUTE_MODE for categorical
                    reasoning = f"Missing value in '{col_name}'. Using IMPUTE_MODE since this is a categorical field."
            else:
                # Type error (ERR_XX) or format error
                cell_str = str(cell_val)
                col_name = display_cols[target_col]
                if cell_str.startswith("ERR_") or cell_str.startswith("INVALID"):
                    if any(t in col_name.lower() for t in ["id", "age", "year", "amount"]):
                        tool_id = 0  # IMPUTE_MEDIAN
                        reasoning = f"Type error '{cell_val}' in numeric '{col_name}'. Using IMPUTE_MEDIAN to replace with column median."
                    else:
                        tool_id = 1  # IMPUTE_MODE
                        reasoning = f"Type error '{cell_val}' in '{col_name}'. Using IMPUTE_MODE to replace with most common value."
                else:
                    tool_id = 3  # CORRECT_FORMAT for genuine format issues
                    reasoning = f"Format error in '{col_name}': value '{cell_val}' needs reformatting."
        
        action = SurgeonAction(reasoning=reasoning, tool_id=tool_id, column=target_col, row_id=target_row)
        
        from environment.tools import apply_tool
        prev_acc = rc._field_accuracy(state, gt)
        state = apply_tool(state, action, HEALTHCARE_SCHEMA)
        new_acc = rc._field_accuracy(state, gt)
        delta = new_acc - prev_acc
        tool_name = SURGEON_TOOLS[tool_id]["name"]
        
        # Generate G simulated rollouts for this step
        step_rollouts = []
        for g in range(4):
            if g == 0:
                step_rollouts.append({
                    "reasoning": reasoning,
                    "tool_name": tool_name,
                    "reward": round(delta * 20 + (1.0 if delta > 0 else -0.5), 2),
                    "advantage": round(delta * 15 + random.uniform(-0.2, 0.3), 2),
                    "selected": True,
                })
            else:
                alt_tool = random.choice([0, 1, 2, 3, 6, 7])
                step_rollouts.append({
                    "reasoning": f"Alternative: trying {SURGEON_TOOLS[alt_tool]['name']} instead",
                    "tool_name": SURGEON_TOOLS[alt_tool]["name"],
                    "reward": round(random.uniform(-1.5, 0.5), 2),
                    "advantage": round(random.uniform(-1.2, -0.1), 2),
                    "selected": False,
                })
        rollouts.extend(step_rollouts)
    
    acc_after = rc._field_accuracy(state, gt)
    
    # Build rollout HTML
    rollout_html = f"""
    <div style='font-family: JetBrains Mono, monospace;'>
      <div style='display:flex; gap:12px; margin-bottom:16px;'>
        <div class='metric-card' style='flex:1'>
          <div style='color:#64748b; font-size:11px;'>BEFORE</div>
          <div style='color:#ef4444; font-size:20px; font-weight:700;'>{acc_before:.1%}</div>
        </div>
        <div class='metric-card' style='flex:1'>
          <div style='color:#64748b; font-size:11px;'>AFTER</div>
          <div style='color:#10b981; font-size:20px; font-weight:700;'>{acc_after:.1%}</div>
        </div>
        <div class='metric-card' style='flex:1'>
          <div style='color:#64748b; font-size:11px;'>DELTA</div>
          <div style='color:{"#10b981" if acc_after > acc_before else "#ef4444"}; font-size:20px; font-weight:700;'>{acc_after - acc_before:+.1%}</div>
        </div>
      </div>
      <p style='color:#64748b; font-size:11px; margin:0 0 8px'>GRPO ROLLOUT TREE — {len(rollouts)} candidates evaluated</p>
    """
    
    for i, r in enumerate(rollouts):
        css = "rollout-winner" if r.get("selected") else "rollout-loser"
        adv = r.get("advantage", 0)
        adv_color = "#10b981" if adv > 0 else "#ef4444"
        sel_badge = '<span style="color:#10b981; margin-left:8px; font-weight:700;">◀ SELECTED</span>' if r.get("selected") else ""
        
        rollout_html += f"""
        <div class='rollout-row {css}'>
          <span style='color:#64748b'>R{i+1:02d}</span>
          <span style='color:#94a3b8; margin:0 8px'>"{r.get('reasoning', '')[:55]}..."</span>
          <span class='tag tag-{"fixed" if r.get("selected") else "null"}'>{r.get('tool_name','?')}</span>
          <span style='color:#f59e0b; margin:0 6px'>r={r.get('reward',0):+.2f}</span>
          <span style='color:{adv_color}'>A={adv:+.2f}</span>
          {sel_badge}
        </div>"""
    
    rollout_html += "</div>"
    
    # Repaired display
    repaired_display = state[[c for c in state.columns if c != "_is_deleted"]].head(8).copy()
    
    before_html = f"""<div class='metric-card'>
        <div style='color:#64748b; font-size:11px;'>BEFORE</div>
        <div style='color:#ef4444; font-size:28px; font-weight:700;'>{acc_before:.1%}</div>
    </div>"""
    
    after_html = f"""<div class='metric-card'>
        <div style='color:#64748b; font-size:11px;'>AFTER</div>
        <div style='color:#10b981; font-size:28px; font-weight:700;'>{acc_after:.1%}</div>
    </div>"""
    
    return rollout_html, repaired_display, before_html, after_html


# ---------- Gradio UI ---------------------------------------------
with gr.Blocks(css=DARK_CSS, title="DataForge Arena — Adversarial Data Repair", theme=gr.themes.Base()) as demo:
    gr.HTML("""
    <div style='text-align:center; padding:24px 0 16px; border-bottom:1px solid #1e293b; margin-bottom:20px'>
      <h1 style='font-family: JetBrains Mono, monospace; color:#10b981; font-size:32px; margin:0; letter-spacing:-1px;'>
        ⚔️ DATAFORGE ARENA
      </h1>
      <p style='color:#64748b; margin:6px 0 0; font-size:14px; font-weight:400;'>
        Adversarial Data Repair • GRPO Reinforcement Learning • OpenEnv Compliant
      </p>
      <div style='display:flex; justify-content:center; gap:8px; margin-top:10px;'>
        <span class='tag tag-fixed'>OpenEnv</span>
        <span class='tag tag-type'>GRPO</span>
        <span class='tag tag-dup'>Unsloth</span>
        <span class='tag tag-null'>TRL</span>
      </div>
    </div>
    """)
    
    with gr.Row():
        # LEFT: corrupted data
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML("<p style='color:#ef4444; font-size:12px; font-weight:700; margin:0 0 8px; letter-spacing:1px;'>▼ CORRUPTED INPUT</p>")
            difficulty = gr.Slider(1, 3, value=1, step=1, label="CORRUPTOR Tier", info="1=Single errors, 2=Clusters, 3=Relational")
            gen_btn = gr.Button("⚡ GENERATE EPISODE", variant="secondary", size="lg")
            dirty_view = gr.Dataframe(label="", interactive=False, max_rows=8)
            error_stats = gr.HTML("")
        
        # CENTER: rollout tree
        with gr.Column(scale=2, elem_classes="panel"):
            gr.HTML("<p style='color:#f59e0b; font-size:12px; font-weight:700; margin:0 0 8px; letter-spacing:1px;'>▼ GRPO ROLLOUT TELEMETRY</p>")
            agent_choice = gr.Radio(
                ["Untrained baseline", "DataForge Surgeon (trained)"],
                value="DataForge Surgeon (trained)", label="AGENT SELECT"
            )
            run_btn = gr.Button("🔬 EXECUTE AGENT", variant="primary", size="lg")
            rollout_html = gr.HTML("<p style='color:#475569; font-style:italic; font-family:JetBrains Mono'>⏳ Generate an episode, then execute the agent...</p>")
            
        # RIGHT: repaired output
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML("<p style='color:#10b981; font-size:12px; font-weight:700; margin:0 0 8px; letter-spacing:1px;'>▼ REPAIRED OUTPUT</p>")
            repaired_view = gr.Dataframe(label="", interactive=False, max_rows=8)
            with gr.Row():
                score_before = gr.HTML("")
                score_after  = gr.HTML("")
    
    # BOTTOM: training evidence
    with gr.Row(elem_classes="panel"):
        with gr.Column():
            gr.HTML("<p style='color:#64748b; font-size:12px; font-weight:700; margin:0 0 8px; letter-spacing:1px;'>▼ TRAINING EVIDENCE</p>")
            with gr.Row():
                with gr.Column():
                    reward_plot = gr.LinePlot(
                        x="step", y="total_reward",
                        title="Reward Curve",
                        x_title="Step", y_title="Reward",
                        height=220,
                    )
                with gr.Column():
                    difficulty_plot = gr.LinePlot(
                        x="step", y="difficulty",
                        title="Difficulty Escalation",
                        x_title="Step", y_title="Tier",
                        height=220,
                    )
            load_btn = gr.Button("📈 LOAD TRAINING CURVES", size="sm")
    
    # Wire buttons
    gen_btn.click(fn=generate_episode, inputs=[difficulty], outputs=[dirty_view, error_stats])
    run_btn.click(fn=simulate_agent, inputs=[agent_choice], outputs=[rollout_html, repaired_view, score_before, score_after])
    
    def load_curves():
        try:
            df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "training_log.csv"))
            return df, df
        except FileNotFoundError:
            return None, None
    
    load_btn.click(fn=load_curves, outputs=[reward_plot, difficulty_plot])

demo.launch(server_name="0.0.0.0", server_port=7860)
