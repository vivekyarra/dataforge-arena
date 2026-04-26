import re
with open('demo/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Extract CSS and everything before it
css_start = content.find('CSS = """')
if css_start == -1:
    css_start = content.find("CSS = '''")
    if css_start == -1:
        css_start = content.find('CSS = ')

head = content[:css_start]

# 2. Extract everything after demo = build_demo()
build_demo_end = content.find('if __name__ == "__main__":')

tail = content[build_demo_end:]

css = '''CSS = """
:root {
  --bg: #050505;
  --panel: #0f0f0f;
  --border: rgba(255,255,255,0.08);
  --t1: #ffffff;
  --t2: #a0a0a0;
  --t3: #555555;
  --g: #10b981;
  --r: #ef4444;
  --a: #f59e0b;
  --bl: #3b82f6;
  --body: "Inter", -apple-system, sans-serif;
  --mono: "DM Mono", monospace;
  --radius: 12px;
}
body {
  background-color: var(--bg) !important;
  font-family: var(--body) !important;
  color: var(--t1) !important;
  margin: 0;
  padding: 0;
}
* { box-sizing: border-box; }

/* Base Gradio Overrides */
.gradio-container {
  background-color: var(--bg) !important;
  max-width: 1400px !important;
  border: none !important;
}

/* Sidebar and Main */
.sidebar {
  background-color: var(--bg);
  padding: 16px 24px;
}
.main-content {
  background-color: var(--bg);
  padding: 16px 24px;
}

/* Card/Panel Styling */
.gradio-html, .gradio-dataframe, .gradio-accordion, .gradio-plot {
  background-color: var(--panel) !important;
  box-shadow: 0 0 0 1px var(--border), 0 4px 20px rgba(0,0,0,0.4) !important;
  border: none !important;
  border-radius: var(--radius) !important;
  margin-bottom: 16px !important;
  overflow: hidden !important;
}

/* Typography Overrides */
.gradio-html * {
  font-family: var(--body);
}
.pane-label {
  font-family: var(--body);
  font-size: 11px;
  font-weight: 600;
  color: var(--t3);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 12px;
}

/* Custom Radio Cards */
.custom-radio {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}
.custom-radio .wrap {
  display: flex !important;
  flex-direction: column !important;
  gap: 12px !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.custom-radio label {
  background: var(--panel) !important;
  box-shadow: 0 0 0 1px var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 16px 20px !important;
  display: flex !important;
  align-items: center !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  margin: 0 !important;
}
.custom-radio label:hover {
  background: #151515 !important;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.15) !important;
}
.custom-radio label.selected {
  box-shadow: 0 0 0 1px var(--border), inset 4px 0 0 var(--g) !important;
  background: rgba(16,185,129,0.03) !important;
}
.custom-radio input[type="radio"] { display: none !important; }
.custom-radio .ml-2 {
  font-family: var(--body) !important;
  font-weight: 500 !important;
  color: var(--t1) !important;
  font-size: 13px !important;
  margin: 0 !important;
  width: 100%;
}

/* Primary Button (Claude style) */
button.primary {
  background: #ffffff !important;
  color: #000000 !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 14px 24px !important;
  font-weight: 600 !important;
  font-family: var(--body) !important;
  font-size: 14px !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 15px rgba(255,255,255,0.1) !important;
  width: 100% !important;
}
button.primary:hover {
  background: #e0e0e0 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(255,255,255,0.15) !important;
}

/* Accordion */
.gradio-accordion > label {
  background-color: var(--panel) !important;
  color: var(--t1) !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  padding: 16px 24px !important;
  border-bottom: 1px solid var(--border) !important;
}

/* Dataframe */
.table-wrap { background: transparent !important; border: none !important; }
.dt { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 11px; }
.dt th { padding: 10px 16px; background: rgba(255,255,255,0.02); color: var(--t3); font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border); text-align: left; }
.dt td { padding: 10px 16px; color: var(--t2); border-bottom: 1px solid rgba(255,255,255,0.02); max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Topbar & Header */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 20px 24px; background: var(--panel);
  box-shadow: 0 0 0 1px var(--border); border-radius: var(--radius);
  margin-bottom: 24px;
}
.tb-title { font-size: 18px; font-weight: 600; color: var(--t1); display: flex; align-items: center; gap: 10px; }
.tb-stats { display: flex; gap: 24px; }
.tbs-item { display: flex; flex-direction: column; gap: 4px; }
.tbs-lbl { font-size: 10px; color: var(--t3); text-transform: uppercase; letter-spacing: 0.05em; }
.tbs-val { font-family: var(--mono); font-size: 14px; color: var(--t1); }

/* Status Strip */
.status-strip {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 24px; background: var(--panel);
  box-shadow: 0 0 0 1px var(--border); border-radius: var(--radius);
  font-family: var(--mono); font-size: 12px; color: var(--t2);
  margin-bottom: 16px;
}
.status-strip span b { color: var(--t1); font-weight: 500; }

/* Streaming Rollout */
.ro-traj { display: flex; flex-direction: column; gap: 8px; padding: 16px 24px; }
.tr-row {
  display: grid; grid-template-columns: 40px 1fr auto auto 60px;
  align-items: center; gap: 16px; padding: 12px 16px;
  background: rgba(255,255,255,0.02); border-radius: var(--radius); border-left: 2px solid transparent;
}
.tr-win  { border-left-color: var(--g); background: rgba(16,185,129,0.05); }
.tr-loss { border-left-color: var(--r); background: rgba(239,68,68,0.05); }
.tr-n    { font-family: var(--mono); font-size: 11px; color: var(--t3); }
.tr-rsn  { font-family: var(--body); font-size: 13px; color: var(--t1); font-style: italic; line-height: 1.4; }
.tr-tool { font-family: var(--mono); font-size: 11px; color: var(--t1); background: rgba(255,255,255,0.1); padding: 4px 8px; border-radius: 4px; }
.tr-loc  { font-family: var(--mono); font-size: 11px; color: var(--t3); }
.tr-rew  { font-family: var(--mono); font-size: 13px; font-weight: 500; text-align: right; }
.tr-rew-pos { color: var(--g); }
.tr-rew-neg { color: var(--r); }
.rot-empty  { padding: 24px; text-align: center; color: var(--t3); font-size: 13px; font-style: italic; }

/* Benchmark & Rings */
.bm-root { padding: 24px; }
.bm-lane { margin-bottom: 16px; }
.bm-meta { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 12px; }
.bm-track { height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; }
.bm-bar { height: 100%; border-radius: 2px; background: var(--g); }

.diff-root { display: flex; flex-direction: column; gap: 16px; padding: 24px; }
.diff-stats-row { display: flex; gap: 16px; }
.dsr-item { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px; background: rgba(255,255,255,0.02); padding: 16px; border-radius: 8px; }
.dsr-val { font-family: var(--mono); font-size: 24px; color: var(--t1); font-weight: 500; }
.dsr-label { font-size: 10px; color: var(--t3); text-transform: uppercase; letter-spacing: 0.05em; }

/* API Section */
.api-block { background: #000; padding: 16px; border-radius: var(--radius); font-family: var(--mono); font-size: 12px; color: var(--t2); margin-top: 12px; overflow-x: auto; box-shadow: inset 0 0 0 1px var(--border); }
.api-block code { color: var(--g); }
.api-desc { font-size: 13px; color: var(--t2); margin-bottom: 8px; }

/* Legacy Fallbacks to keep old elements from breaking */
.err-banner { padding:20px; background:rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.2); border-radius:var(--radius); font-family:var(--mono); font-size:11px; color:var(--r); }
.empty-center { padding:40px 20px; text-align:center; color:var(--t3); }
.ro-dna { padding:16px; }
.dna-head { font-family:var(--mono); font-size:10px; color:var(--t3); text-transform:uppercase; margin-bottom:12px; }
.dna-r { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
.dna-lbl { width:100px; font-family:var(--mono); font-size:11px; color:var(--t2); }
.dna-track { flex:1; height:4px; background:rgba(255,255,255,0.1); border-radius:2px; }
.dna-fill { height:100%; border-radius:2px; background:var(--g); }
.df-n { background:var(--r); }
.dna-v { width:40px; text-align:right; font-family:var(--mono); font-size:11px; }

"""
'''

build_demo_code = '''
def _api_section_html():
    return """
    <div style="padding: 12px 24px 24px 24px;">
        <div class="api-desc">DataForge Arena provides a full OpenEnv REST API for programmatic evaluation.</div>
        
        <div class="api-desc" style="margin-top: 24px;"><b>Reset environment (cURL)</b></div>
        <div class="api-block">
<pre>curl -X POST https://vivek567-dataforge-arena.hf.space/reset \\
  -H "Content-Type: application/json" \\
  -d '{"tier": 1}'</pre>
        </div>
        
        <div class="api-desc" style="margin-top: 24px;"><b>Execute one repair step (cURL)</b></div>
        <div class="api-block">
<pre>curl -X POST https://vivek567-dataforge-arena.hf.space/step \\
  -H "Content-Type: application/json" \\
  -d '{"reasoning": "age 145 exceeds schema max 120", "tool_id": 3, "column": 2, "row_id": 7}'</pre>
        </div>

        <div class="api-desc" style="margin-top: 24px;"><b>Python Requests Example</b></div>
        <div class="api-block">
<pre>import requests

BASE = "https://vivek567-dataforge-arena.hf.space"
obs = requests.post(f"{BASE}/reset", json={"tier": 1}).json()

for step in range(5):
    action = {"reasoning": "...", "tool_id": 3, "column": 2, "row_id": 7}
    result = requests.post(f"{BASE}/step", json=action).json()
    print(f"Step {step+1} reward: {result['reward']:.3f}")</pre>
        </div>
    </div>
    """

def build_demo():
    choices = available_agent_choices()
    default = "Live GRPO Model \u00b7 Trained on 265 RL steps [GRPO]" if "Live GRPO Model" in choices else "Heuristic Surgeon \u00b7 Rule-based constraint-aware repairs [DETERMINISTIC]"

    agent_choices = [
        "Naive Baseline \u00b7 Greedy imputation, no schema awareness [BASELINE]",
        "Heuristic Surgeon \u00b7 Rule-based constraint-aware repairs [DETERMINISTIC]",
    ]
    if "Live GRPO Model" in choices:
        agent_choices.append("Live GRPO Model \u00b7 Trained on 265 RL steps [GRPO]")
    else:
        agent_choices.append("Live GRPO Model \u00b7 Checkpoint Required [UNAVAILABLE]")

    tier_choices = [
        "Tier 1 \u00b7 Nulls, type errors, range violations",
        "Tier 2 \u00b7 FK mismatches, temporal drift",
        "Tier 3 \u00b7 Full relational reasoning",
    ]

    with gr.Blocks(title="DataForge Arena", css=CSS, theme=gr.themes.Base()) as demo:
        state = gr.State(_new_state())
        
        # Header bar (keep existing _topbar_html() logic)
        topbar = gr.HTML(_topbar_html())
        
        with gr.Row(equal_height=False):
            # LEFT SIDEBAR
            with gr.Column(scale=1, min_width=320, elem_classes=["sidebar"]):
                gr.HTML("<div class='pane-label'>Agent Selection</div>")
                agent_pick = gr.Radio(choices=agent_choices, value=default, label="", interactive=True, elem_classes=["custom-radio"])
                
                gr.HTML("<div class='pane-label' style='margin-top:32px;'>Scenario Complexity</div>")
                tier_pick = gr.Radio(choices=tier_choices, value="Tier 1 \u00b7 Nulls, type errors, range violations", label="", interactive=True, elem_classes=["custom-radio"])
                
                gen_btn = gr.Button("Create New Scenario", variant="primary", elem_classes=["primary"])
                
                gr.HTML("<div class='pane-label' style='margin-top:40px;'>Benchmark Performance</div>")
                bench_html = gr.HTML(_benchmark_html())
                
                gr.HTML("<div class='pane-label' style='margin-top:32px;'>Training Convergence \u00b7 265 steps</div>")
                reward_spark = gr.LinePlot(x="step", y="total_reward", height=150, x_title="Step", y_title="Reward", tooltip=["step", "total_reward"])
            
            # MAIN CONTENT
            with gr.Column(scale=3, elem_classes=["main-content"]):
                # Status strip
                status_strip = gr.HTML("<div class='status-strip'><span>Tier: <b>1</b></span><span>Status: <b>Waiting</b></span><span>Target Rows: <b>--</b></span></div>")
                
                # Data view (collapsible)
                with gr.Accordion("Live Corrupted Dataset", open=True):
                    dirty_view = gr.Dataframe(label="", interactive=False, wrap=False)
                
                # Execute button
                exec_btn = gr.Button("\u25b6 Execute Repair Policy", variant="primary", size="lg", elem_classes=["primary"])
                
                # Streaming output
                rollout_out = gr.HTML(_empty_rollout())
                
                # Results row
                with gr.Row():
                    with gr.Column(scale=1):
                        acc_display = gr.HTML(_accuracy_display(None, None))
                    with gr.Column(scale=2):
                        repaired_view = gr.Dataframe(label="", interactive=False, wrap=False)
                
                diff_out = gr.HTML(_diff_html(None, None, None))
        
        # API section at bottom
        with gr.Accordion("Developer API", open=False):
            gr.HTML(_api_section_html())
        
        # --- Logic Wiring ---
        def load_dash():
            df = get_training_data()
            return _topbar_html(), _benchmark_html(), df
        
        def on_gen(t_val, session_state):
            tier = 1
            if "Tier 2" in t_val: tier = 2
            elif "Tier 3" in t_val: tier = 3

            s_out = _new_state()
            s_out["tier"] = tier
            dirty, gt, acc = generate_episode(tier)
            s_out["dirty"] = dirty
            s_out["gt"]    = gt
            
            stats = f"<div class='status-strip'><span>Tier: <b>{tier}</b></span><span>Rows: <b>{len(dirty)}</b></span><span>Initial Acc: <b>{acc*100:.1f}%</b></span></div>"
            return dirty.head(8), stats, s_out, _empty_rollout(), _accuracy_display(acc, acc), dirty.head(8), _diff_html(dirty, dirty, gt)

        gen_outs = [dirty_view, status_strip, state, rollout_out, acc_display, repaired_view, diff_out]
        gen_btn.click(fn=on_gen, inputs=[tier_pick, state], outputs=gen_outs)

        def simulate_with_repaired(agent_val, session_state):
            if "Naive Baseline" in agent_val: agent_type = "Naive Baseline"
            elif "Heuristic Surgeon" in agent_val: agent_type = "Heuristic Surgeon"
            else: agent_type = "Live GRPO Model"

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
                            if pd.isna(cell):     tr,tc,tid,rsn=ri,ci,0,"Null->IMPUTE_MEDIAN"; break
                            if str(cell).startswith("ERR_"): tr,tc,tid,rsn=ri,ci,0,"ERR->IMPUTE_MEDIAN"; break
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
                            {"role":"user","content":f"Observation: {obs.model_dump_json()}\\nOutput valid JSON only."}]
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

        demo.load(fn=load_dash, outputs=[topbar, bench_html, reward_spark])

    return demo
'''

with open('demo/app.py', 'w', encoding='utf-8') as f:
    f.write(head + css + build_demo_code + tail)
