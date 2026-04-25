# demo/app.py -- Tactical DataForge Arena Demo
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
from training.parser import robust_parse_action
from training.prompt import build_prompt

# ---------- resolve paths -----------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "healthcare_clean.csv")
LOG_PATH = os.path.join(ROOT_DIR, "logs", "training_log.csv")

# ---------- load environment once ---------------------------------
clean_data = pd.read_csv(DATA_PATH)
corruptor = Corruptor()
env = DataForgeEnv(corruptor=corruptor, schema=HEALTHCARE_SCHEMA, clean_data=clean_data)
rc = RewardComputer()

# Episode state held between button clicks
current_dirty = [None]
current_gt = [None]
current_meta = [None]

llm_pipeline = None

def load_llm():
    """Lazy load the transformers pipeline on first inference request."""
    global llm_pipeline
    if llm_pipeline is not None:
        return True
    try:
        from transformers import pipeline
        import torch
        print("Loading Live LLM Inference Pipeline...")
        device = 0 if torch.cuda.is_available() else -1
        model_path = "outputs/dataforge-surgeon"
        if not os.path.exists(model_path):
            print("WARNING: Trained LoRA model not found. Falling back to base model.")
            model_path = "Qwen/Qwen2.5-1.5B-Instruct"
            
        llm_pipeline = pipeline("text-generation", model=model_path, 
                                device=device, torch_dtype=torch.bfloat16 if device==0 else torch.float32)
        print("Live LLM Loaded Successfully.")
        return True
    except Exception as e:
        print(f"Error loading LLM: {e}")
        return False


# ---------- CSS ---------------------------------------------------
DARK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
body, .gradio-container { background: #06060b !important; color: #f1f5f9; font-family: 'Inter', sans-serif; }
.panel { background: linear-gradient(135deg, rgba(13,13,21,0.9) 0%, rgba(17,17,24,0.95) 100%); backdrop-filter: blur(12px); border: 1px solid #1e293b; border-top: 1px solid #334155; border-radius: 12px; padding: 20px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.8), 0 8px 10px -6px rgba(0, 0, 0, 0.5); }
h1 { font-family: 'JetBrains Mono', monospace !important; text-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
.metric-card { background: linear-gradient(180deg, #151522 0%, #0b0b12 100%); border: 1px solid #1e293b; border-top: 1px solid #334155; border-radius: 8px; padding: 16px; text-align: center; box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 4px 6px rgba(0,0,0,0.4); word-break: break-word; transition: transform 0.2s ease; }
.metric-card:hover { transform: translateY(-2px); }
.rollout-row { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; padding: 10px 14px; margin: 6px 0; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 12px; transition: all 0.2s ease; }
.rollout-winner { background: linear-gradient(90deg, rgba(16,185,129,0.2) 0%, rgba(16,185,129,0.05) 60%, transparent 100%); border-left: 3px solid #10b981; box-shadow: inset 1px 0 0 rgba(16,185,129,0.5); }
.rollout-loser  { background: rgba(239,68,68,0.15); border-left: 3px solid #ef4444; opacity: 0.9; }
.rollout-row:hover { background: rgba(51,65,85,0.5); opacity: 1; }
.tag { padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace; white-space: nowrap; box-shadow: 0 2px 4px rgba(0,0,0,0.3); text-shadow: 0 1px 1px rgba(0,0,0,0.5); }
.tag-null   { background:linear-gradient(135deg, #7f1d1d, #991b1b); color:#fca5a5; border: 1px solid #b91c1c; }
.tag-type   { background:linear-gradient(135deg, #78350f, #92400e); color:#fde68a; border: 1px solid #b45309; }
.tag-fixed  { background:linear-gradient(135deg, #064e3b, #065f46); color:#a7f3d0; border: 1px solid #059669; }
.tag-dup    { background:linear-gradient(135deg, #1e3a8a, #1d4ed8); color:#bfdbfe; border: 1px solid #2563eb; }
.pulse { animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); } 50% { opacity:0.8; box-shadow: 0 0 10px 4px rgba(16,185,129,0); } }
"""

# ---------- training data loader ----------
def get_training_data():
    try:
        df = pd.read_csv(LOG_PATH)
        if len(df) > 0:
            return df
    except:
        pass
    return pd.DataFrame({"step": [0], "total_reward": [0], "difficulty": [1]})


# ---------- helper funcs ------------------------------------------
def generate_episode(tier):
    tier = int(tier)
    corruptor._epoch = {1: 0, 2: 65, 3: 115}[tier]

    n = min(50, len(clean_data))
    sample = clean_data.sample(n=n).reset_index(drop=True)
    dirty, gt, meta = corruptor.generate_episode(sample)

    if meta.get("tool") == "duplicate_row_mutate" and len(dirty) > len(gt):
        src = meta.get("row", 0)
        if src < len(gt):
            gt = pd.concat([gt, gt.iloc[[src]]], ignore_index=True)

    current_dirty[0] = dirty.copy()
    current_gt[0] = gt.copy()
    current_meta[0] = meta

    display_cols = [c for c in dirty.columns if c != "_is_deleted"]
    display = dirty[display_cols].head(8).copy()

    acc_before = rc._field_accuracy(dirty, gt)
    total_nulls = dirty[display_cols].isnull().sum().sum()
    total_cells = dirty[display_cols].size

    stats_html = f"""
    <div style='font-family: JetBrains Mono, monospace; font-size:13px; margin-top:12px; padding-bottom:8px;'>
      <div style='display:flex; gap:16px;'>
        <div class='metric-card' style='flex:1'>
          <div style='color:#94a3b8; font-size:11px; letter-spacing:1px; margin-bottom:4px; font-weight:600;'>DATASET HEALTH</div>
          <div style='color:{"#ef4444" if acc_before < 0.95 else "#fcd34d"}; font-size:24px; font-weight:700; text-shadow: 0 2px 4px rgba(0,0,0,0.5);'>{acc_before:.1%}</div>
        </div>
        <div class='metric-card' style='flex:1'>
          <div style='color:#94a3b8; font-size:11px; letter-spacing:1px; margin-bottom:4px; font-weight:600;'>ERRORS</div>
          <div style='color:#ef4444; font-size:24px; font-weight:700; text-shadow: 0 2px 4px rgba(0,0,0,0.5);'>{total_nulls}</div>
        </div>
        <div class='metric-card' style='flex:1'>
          <div style='color:#94a3b8; font-size:11px; letter-spacing:1px; margin-bottom:4px; font-weight:600;'>CORRUPTION</div>
          <div style='color:#fcd34d; font-size:{"16px" if len(meta["tool"]) > 16 else "20px"}; font-weight:700; line-height:1.2; text-shadow: 0 2px 4px rgba(0,0,0,0.5);'>{meta["tool"]}</div>
        </div>
      </div>
    </div>
    """
    return display, stats_html

def render_ui_state(rollouts, current_state, gt, acc_before, agent_type):
    acc_after = rc._field_accuracy(current_state, gt)
    success_rate_improvement = ((acc_after - acc_before) / (1.0 - acc_before)) * 100 if acc_before < 1.0 else 0

    rollout_html = f"""
    <div style='font-family: JetBrains Mono, monospace;'>
      <div style='display:flex; gap:12px; margin-bottom:16px;'>
        <div class='metric-card' style='flex:1'>
          <div style='color:#94a3b8; font-size:11px; letter-spacing:1px; margin-bottom:4px; font-weight:600;'>DATASET HEALTH (BEFORE)</div>
          <div style='color:#ef4444; font-size:24px; font-weight:700; text-shadow: 0 2px 4px rgba(0,0,0,0.5);'>{acc_before:.1%}</div>
        </div>
        <div class='metric-card' style='flex:1'>
          <div style='color:#94a3b8; font-size:11px; letter-spacing:1px; margin-bottom:4px; font-weight:600;'>DATASET HEALTH (AFTER)</div>
          <div style='color:{"#ef4444" if acc_after < acc_before else "#10b981"}; font-size:24px; font-weight:700; text-shadow: 0 2px 4px rgba(0,0,0,0.5);'>{acc_after:.1%}</div>
        </div>
        <div class='metric-card' style='flex:1'>
          <div style='color:#94a3b8; font-size:11px; letter-spacing:1px; margin-bottom:4px; font-weight:600;'>CORRECTION SUCCESS RATE</div>
          <div style='color:{"#10b981" if success_rate_improvement > 0 else "#ef4444"}; font-size:24px; font-weight:700; text-shadow: 0 2px 4px rgba(0,0,0,0.5);'>{success_rate_improvement:+.1f}%</div>
        </div>
      </div>
      <p style='color:#64748b; font-size:11px; margin:0 0 8px'>{"BASELINE EXECUTION LOG" if agent_type == "Naive Rule-Based Baseline" else "LIVE LLM INFERENCE TRAJECTORY"}</p>
    """

    for i, r in enumerate(rollouts):
        if r.get("is_baseline"):
            css = "rollout-loser" if r.get("reward", 0) < 0 else "rollout-winner"
        else:
            css = "rollout-winner"
            
        reasoning_text = r.get('reasoning', '').replace('"', '&quot;')
        if len(reasoning_text) > 55: reasoning_text = reasoning_text[:55] + "..."
        
        rollout_html += f"""
        <div class='rollout-row {css}'>
          <div style='color:#94a3b8; font-weight:700;'>STEP {i+1:02d}</div>
          <div style='color:#cbd5e1; flex:1; min-width:180px; font-style:italic;'>"{reasoning_text}"</div>
          <div class='tag tag-{"null" if r.get("is_baseline") else "fixed"}'>{r.get('tool_name','?')}</div>
          <div style='color:#fcd34d; font-weight:600; white-space:nowrap;'>Reward={r.get('reward',0):+.2f}</div>
        </div>"""

    rollout_html += "</div>"
    repaired_display = current_state[[c for c in current_state.columns if c != "_is_deleted"]].head(8).copy()

    before_html = f"""<div class='metric-card'>
        <div style='color:#94a3b8; font-size:11px; letter-spacing:1px; margin-bottom:4px; font-weight:600;'>BEFORE</div>
        <div style='color:#ef4444; font-size:28px; font-weight:700; text-shadow: 0 2px 4px rgba(0,0,0,0.5); white-space:nowrap;'>{acc_before:.1%}</div>
    </div>"""

    after_html = f"""<div class='metric-card'>
        <div style='color:#94a3b8; font-size:11px; letter-spacing:1px; margin-bottom:4px; font-weight:600;'>AFTER</div>
        <div style='color:{"#ef4444" if acc_after < acc_before else "#10b981"}; font-size:28px; font-weight:700; text-shadow: 0 2px 4px rgba(0,0,0,0.5); white-space:nowrap;'>{acc_after:.1%}</div>
    </div>"""

    return rollout_html, repaired_display, before_html, after_html


def simulate_agent(agent_type):
    if current_dirty[0] is None:
        yield ("<p style='color:#ef4444'>Generate an episode first!</p>", None, "", "")
        return

    dirty = current_dirty[0].copy()
    gt = current_gt[0].copy()
    meta = current_meta[0]
    display_cols = [c for c in dirty.columns if c != "_is_deleted"]
    acc_before = rc._field_accuracy(dirty, gt)
    
    # Initialize env explicitly for this rollout
    env._state = dirty.copy()
    env._ground_truth = gt.copy()
    env._original_dirty = dirty.copy()
    env._prev_accuracy = acc_before
    env._starting_accuracy = acc_before
    env._step_count = 0
    env._action_log = []
    import time
    env._episode_start = time.time()

    rollouts = []
    
    # 5 Steps max
    for step in range(5):
        if agent_type == "Naive Rule-Based Baseline":
            target_row, target_col = None, None
            action_tool = 7
            action_reason = "No errors found."
            
            for r in range(len(env._state)):
                for c_idx, c_name in enumerate(display_cols):
                    cell = env._state.at[r, c_name] if c_name in env._state.columns else None
                    if pd.isna(cell):
                        target_row, target_col = r, c_idx
                        action_tool = 0 # IMPUTE_MEDIAN
                        action_reason = "Naive baseline: Null found. Imputing median."
                        break
                    elif str(cell).startswith("ERR_"):
                        target_row, target_col = r, c_idx
                        action_tool = 0 # IMPUTE_MEDIAN
                        action_reason = "Naive baseline: Type error found. Imputing median."
                        break
                if target_row is not None: break
            
            action = SurgeonAction(reasoning=action_reason, tool_id=action_tool, 
                                   column=target_col if target_col else 0, 
                                   row_id=target_row if target_row else 0)
            
            # Apply Action
            obs, total_reward, done, info = env.step(action)
            rollouts.append({
                "reasoning": action.reasoning,
                "tool_name": SURGEON_TOOLS[action.tool_id]["name"],
                "reward": total_reward,
                "advantage": total_reward - 0.5,
                "selected": True,
                "is_baseline": True
            })
            
            yield render_ui_state(rollouts, env._state, gt, acc_before, agent_type)
            if done: break

        else:
            # LIVE LLM INFERENCE
            if not load_llm():
                yield ("<p style='color:#ef4444'>Failed to load LLM locally (Check CUDA/RAM).</p>", None, "", "")
                return
            
            obs = env._make_observation()
            prompt = build_prompt(obs)
            
            # Formulate chat
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Observation: {obs.model_dump_json()}\\nOutput valid JSON only."}
            ]
            
            try:
                outputs = llm_pipeline(messages, max_new_tokens=256, temperature=0.1, do_sample=False, num_return_sequences=1)
                generated_text = outputs[0]["generated_text"][-1]["content"]
                action = robust_parse_action(generated_text)
            except Exception as e:
                print(f"LLM Inference failed: {e}")
                # Fallback to a safe action if the model outputs absolute garbage
                action = SurgeonAction(reasoning=f"LLM parse failure: {str(e)[:40]}", tool_id=7, column=0, row_id=0)

            obs, total_reward, done, info = env.step(action)
            
            rollouts.append({
                "reasoning": action.reasoning,
                "tool_name": SURGEON_TOOLS[action.tool_id]["name"],
                "reward": total_reward,
                "advantage": total_reward - 0.1,
                "selected": True,
                "is_baseline": False
            })
            
            yield render_ui_state(rollouts, env._state, gt, acc_before, agent_type)
            if done: break


# ---------- Gradio UI ---------------------------------------------
with gr.Blocks(title="DataForge Arena") as demo:
    gr.HTML("""
    <div style='text-align:center; padding:24px 0 16px; border-bottom:1px solid #1e293b; margin-bottom:20px'>
      <h1 style='font-family: JetBrains Mono, monospace; color:#10b981; font-size:32px; margin:0; letter-spacing:-1px;'>
        DATAFORGE ARENA
      </h1>
      <p style='color:#64748b; margin:6px 0 0; font-size:14px; font-weight:400;'>
        Self-improving data repair agents trained in adversarial environments
      </p>
      <div style='display:flex; justify-content:center; gap:8px; margin-top:10px;'>
        <span class='tag tag-fixed'>PyTorch</span>
        <span class='tag tag-type'>TRL GRPO</span>
        <span class='tag tag-dup'>OpenEnv</span>
        <span class='tag tag-null'>Live Inference</span>
      </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML("<p style='color:#ef4444; font-size:12px; font-weight:700; margin:0 0 8px; letter-spacing:1px;'>CORRUPTED INPUT</p>")
            with gr.Row():
                btn_easy = gr.Button("🔥 Easy Scenario (Tier 1)", variant="secondary")
                btn_hard = gr.Button("☠️ Adversarial Scenario (Tier 3)", variant="secondary")
            
            dirty_view = gr.Dataframe(label="", interactive=False)
            error_stats = gr.HTML("")

        with gr.Column(scale=2, elem_classes="panel"):
            gr.HTML("<p style='color:#f59e0b; font-size:12px; font-weight:700; margin:0 0 8px; letter-spacing:1px;'>AGENT TELEMETRY</p>")
            agent_choice = gr.Radio(
                ["Naive Rule-Based Baseline", "DataForge Surgeon (Live Inference)"],
                value="DataForge Surgeon (Live Inference)", label="AGENT SELECT"
            )
            run_btn = gr.Button("EXECUTE AGENT", variant="primary", size="lg")
            rollout_html = gr.HTML("<p style='color:#475569; font-style:italic; font-family:JetBrains Mono'>Select a scenario, then execute the agent to see live inference.</p>")

        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML("<p style='color:#10b981; font-size:12px; font-weight:700; margin:0 0 8px; letter-spacing:1px;'>REPAIRED OUTPUT</p>")
            repaired_view = gr.Dataframe(label="", interactive=False)
            with gr.Row():
                score_before = gr.HTML("")
                score_after  = gr.HTML("")

    # BOTTOM: training evidence
    with gr.Row(elem_classes="panel"):
        with gr.Column():
            gr.HTML("<p style='color:#64748b; font-size:12px; font-weight:700; margin:0 0 8px; letter-spacing:1px;'>TRAINING EVIDENCE</p>")
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

    # Wire buttons
    btn_easy.click(fn=lambda: generate_episode(1), outputs=[dirty_view, error_stats])
    btn_hard.click(fn=lambda: generate_episode(3), outputs=[dirty_view, error_stats])
    run_btn.click(fn=simulate_agent, inputs=[agent_choice],
                  outputs=[rollout_html, repaired_view, score_before, score_after])

    def load_curves():
        df = get_training_data()
        return df, df

    demo.load(fn=load_curves, outputs=[reward_plot, difficulty_plot])

demo.launch(server_name="0.0.0.0", server_port=7860, css=DARK_CSS, theme=gr.themes.Base())
