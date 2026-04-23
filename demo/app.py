# demo/app.py
import gradio as gr
import json
import pandas as pd

DARK_CSS = """
body, .gradio-container { background: #0a0a0f !important; color: #e2e8f0; }
.panel { background: #111118; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; }
.metric { font-family: 'Courier New', monospace; font-size: 24px; font-weight: bold; }
.metric.good { color: #10b981; }
.metric.bad  { color: #ef4444; }
.metric.mid  { color: #f59e0b; }
.rollout-winner { border-left: 3px solid #10b981; padding-left: 8px; margin: 4px 0; }
.rollout-loser  { border-left: 3px solid #374151; padding-left: 8px; margin: 4px 0; opacity: 0.6; }
.tag-null   { background:#7f1d1d; color:#fca5a5; padding:2px 6px; border-radius:4px; font-size:11px; }
.tag-type   { background:#78350f; color:#fcd34d; padding:2px 6px; border-radius:4px; font-size:11px; }
.tag-format { background:#1e3a5f; color:#93c5fd; padding:2px 6px; border-radius:4px; font-size:11px; }
.tag-ok     { background:#064e3b; color:#6ee7b7; padding:2px 6px; border-radius:4px; font-size:11px; }
"""

def color_cell(val, gt_val):
    if pd.isna(val):
        return "NULL"
    if val != gt_val:
        return f"ERR:{val}"
    return str(val)

def format_rollout_display(rollouts: list) -> str:
    """Format GRPO rollouts as tactical telemetry."""
    html = "<div style='font-family:monospace; font-size:13px;'>"
    html += "<p style='color:#64748b; margin:0 0 8px'>GRPO ROLLOUT TREE -- 8 candidates evaluated</p>"
    
    for i, r in enumerate(rollouts):
        selected = r.get("selected", False)
        css = "rollout-winner" if selected else "rollout-loser"
        adv = r.get("advantage", 0)
        adv_color = "#10b981" if adv > 0 else "#ef4444"
        
        html += f"""
        <div class='{css}' style='margin-bottom:6px;'>
          <span style='color:#94a3b8'>R{i+1}</span>
          <span style='color:#e2e8f0; margin:0 8px'>"{r.get('reasoning', '')[:60]}..."</span>
          <span style='background:#1e293b; padding:2px 6px; border-radius:4px; font-size:11px'>
            tool={r.get('tool_name','?')}
          </span>
          <span style='color:#f59e0b; margin:0 6px'>reward={r.get('reward',0):+.2f}</span>
          <span style='color:{adv_color}'>A={r.get('advantage',0):+.2f}</span>
          {'<span style="color:#10b981; margin-left:8px"><- SELECTED</span>' if selected else ''}
        </div>"""
    
    html += "</div>"
    return html

with gr.Blocks(css=DARK_CSS, title="DataForge Arena") as demo:
    gr.HTML("""
    <div style='text-align:center; padding:20px 0 10px; border-bottom:1px solid #1e293b; margin-bottom:16px'>
      <h1 style='font-family:monospace; color:#10b981; font-size:28px; margin:0'>
        DATAFORGE ARENA
      </h1>
      <p style='color:#64748b; margin:4px 0 0; font-size:13px'>
        Adversarial Data Repair - GRPO Reinforcement Learning - Self-Improving RL Environment
      </p>
    </div>
    """)
    
    with gr.Row():
        # LEFT: corrupted data
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML("<p style='color:#ef4444; font-size:12px; margin:0 0 8px'>* CORRUPTED INPUT</p>")
            difficulty = gr.Slider(1, 3, value=1, step=1, label="CORRUPTOR tier")
            gen_btn = gr.Button("GENERATE EPISODE", variant="secondary")
            dirty_view = gr.Dataframe(label="", interactive=False, max_rows=10)
            error_stats = gr.HTML("")
        
        # CENTER: rollout tree
        with gr.Column(scale=2, elem_classes="panel"):
            gr.HTML("<p style='color:#f59e0b; font-size:12px; margin:0 0 8px'>* GRPO ROLLOUT TELEMETRY</p>")
            agent_choice = gr.Radio(
                ["Untrained baseline", "DataForge Surgeon (trained)"],
                value="Untrained baseline", label="AGENT"
            )
            run_btn = gr.Button("EXECUTE AGENT", variant="primary")
            rollout_html = gr.HTML("<p style='color:#475569; font-style:italic'>Awaiting execution...</p>")
            
        # RIGHT: repaired output
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML("<p style='color:#10b981; font-size:12px; margin:0 0 8px'>v REPAIRED OUTPUT</p>")
            repaired_view = gr.Dataframe(label="", interactive=False, max_rows=10)
            with gr.Row():
                score_before = gr.HTML("")
                score_after  = gr.HTML("")
    
    # BOTTOM: training evidence
    with gr.Row(elem_classes="panel"):
        gr.HTML("<p style='color:#64748b; font-size:12px; margin:0 0 8px'>* TRAINING EVIDENCE</p>")
        with gr.Column():
            reward_plot = gr.LinePlot(
                x="step", y="total_reward",
                title="Total reward over training steps",
                x_title="Training step",
                y_title="Reward",
                height=200,
            )
        with gr.Column():
            difficulty_plot = gr.LinePlot(
                x="step", y="difficulty",
                title="CORRUPTOR difficulty escalation",
                height=200,
            )
    
    load_btn = gr.Button("LOAD TRAINING CURVES", size="sm")
    
    def load_curves():
        try:
            df = pd.read_csv("logs/training_log.csv")
            return df, df
        except FileNotFoundError:
            return None, None
    
    load_btn.click(fn=load_curves, outputs=[reward_plot, difficulty_plot])

demo.launch(server_name="0.0.0.0", server_port=7860)
