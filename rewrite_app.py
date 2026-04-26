import sys

def main():
    with open('demo/app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    css_start = content.find('CSS = """')
    css_end = content.find('"""', css_start + 8) + 3

    build_demo_start = content.find('def build_demo():')
    build_demo_end = content.find('demo = build_demo()')

    new_css = '''CSS = """
:root {
  --bg: #0a0a0a;
  --panel: #111111;
  --border: rgba(255,255,255,0.08);
  --t1: #ffffff;
  --t2: #a0a0a0;
  --t3: #666666;
  --g: #10b981;
  --r: #ef4444;
  --a: #f59e0b;
  --bl: #3b82f6;
  --body: "Inter", sans-serif;
  --mono: "DM Mono", monospace;
  --radius: 8px;
}
body {
  background-color: var(--bg) !important;
  font-family: var(--body) !important;
  color: var(--t1) !important;
  margin: 0;
  padding: 0;
}
* {
  box-sizing: border-box;
}

/* Base Gradio Overrides */
.gradio-container {
  background-color: var(--bg) !important;
  max-width: 1400px !important;
}
.sidebar {
  background-color: var(--bg);
  padding-right: 24px;
}
.main-content {
  background-color: var(--bg);
  padding-left: 24px;
}

/* Card/Panel Styling */
.gradio-html, .gradio-dataframe, .gradio-accordion {
  background-color: var(--panel) !important;
  box-shadow: 0 0 0 1px var(--border) !important;
  border: none !important;
  border-radius: var(--radius) !important;
  margin-bottom: 16px !important;
}
.gradio-accordion > label {
  background-color: var(--panel) !important;
  color: var(--t1) !important;
  font-weight: 500 !important;
  font-family: var(--body) !important;
  padding: 16px 24px !important;
  border-bottom: 1px solid var(--border) !important;
}
.gradio-accordion .label-wrap {
  border: none !important;
}

/* Dataframe */
.table-wrap {
  border: none !important;
  background: transparent !important;
}
.dt {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--mono);
  font-size: 11px;
}
.dt th {
  padding: 10px 16px;
  background: rgba(255,255,255,0.02);
  color: var(--t3);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-bottom: 1px solid var(--border);
  text-align: left;
}
.dt td {
  padding: 10px 16px;
  color: var(--t2);
  border-bottom: 1px solid rgba(255,255,255,0.02);
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Primary Button (Claude style) */
button.primary {
  background: var(--t1) !important;
  color: var(--bg) !important;
  border: none !important;
  border-radius: var(--radius) !important;
  padding: 12px 24px !important;
  font-weight: 600 !important;
  font-family: var(--body) !important;
  font-size: 14px !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 10px rgba(255,255,255,0.1) !important;
  width: 100% !important;
  display: block !important;
}
button.primary:hover {
  background: #e0e0e0 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 15px rgba(255,255,255,0.15) !important;
}

/* Radio Buttons (Cards) */
.gradio-radio label {
  background: var(--panel) !important;
  box-shadow: 0 0 0 1px var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 16px !important;
  margin-bottom: 8px !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: flex-start !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  border: none !important;
}
.gradio-radio label.selected {
  box-shadow: 0 0 0 1px var(--border), inset 4px 0 0 var(--g) !important;
  background: rgba(16,185,129,0.03) !important;
}
.gradio-radio input[type="radio"] {
  display: none !important;
}
.gradio-radio span.ml-2 {
  font-family: var(--body) !important;
  font-weight: 500 !important;
  color: var(--t1) !important;
  font-size: 14px !important;
  margin: 0 !important;
}

/* Topbar & Layout Fixes */
.pane-label {
  font-family: var(--body);
  font-size: 12px;
  font-weight: 600;
  color: var(--t3);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 12px;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  background: var(--panel);
  box-shadow: 0 0 0 1px var(--border);
  border-radius: var(--radius);
  margin-bottom: 24px;
}
.tb-title {
  font-family: var(--body);
  font-size: 16px;
  font-weight: 600;
  color: var(--t1);
  display: flex;
  align-items: center;
  gap: 8px;
}
.tb-stats {
  display: flex;
  gap: 16px;
}
.tbs-item {
  display: flex;
  flex-direction: column;
}
.tbs-lbl {
  font-size: 10px;
  color: var(--t3);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.tbs-val {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--t1);
}

/* Streaming Rollout */
.ro-traj {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px 24px;
}
.tr-row {
  display: grid;
  grid-template-columns: 40px 1fr auto auto 60px;
  align-items: center;
  gap: 16px;
  padding: 12px 16px;
  background: rgba(255,255,255,0.02);
  border-radius: var(--radius);
  border-left: 2px solid transparent;
}
.tr-win  { border-left-color: var(--g); background: rgba(16,185,129,0.05); }
.tr-loss { border-left-color: var(--r); background: rgba(239,68,68,0.05); }
.tr-n    { font-family: var(--mono); font-size: 11px; color: var(--t3); }
.tr-rsn  { font-family: var(--body); font-size: 13px; color: var(--t2); font-style: italic; }
.tr-tool { font-family: var(--mono); font-size: 11px; color: var(--t1); background: rgba(255,255,255,0.1); padding: 4px 8px; border-radius: 4px; }
.tr-loc  { font-family: var(--mono); font-size: 11px; color: var(--t3); }
.tr-rew  { font-family: var(--mono); font-size: 13px; font-weight: 500; text-align: right; }
.tr-rew-pos { color: var(--g); }
.tr-rew-neg { color: var(--r); }
.rot-empty  { padding: 24px; text-align: center; color: var(--t3); font-size: 13px; font-family: var(--body); font-style: italic; }

/* Status Strip */
.status-strip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 24px;
  background: var(--panel);
  box-shadow: 0 0 0 1px var(--border);
  border-radius: var(--radius);
  margin-bottom: 16px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--t2);
}
.status-strip span b { color: var(--t1); font-weight: 500; }

/* Benchmark & Rings */
.bm-root {
  padding: 24px;
}
.bm-meta {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-family: var(--body);
  font-size: 13px;
}
.bm-track {
  height: 4px;
  background: rgba(255,255,255,0.1);
  border-radius: 2px;
  margin-bottom: 16px;
}
.bm-bar {
  height: 100%;
  border-radius: 2px;
  background: var(--g);
}

.diff-root { display: flex; flex-direction: column; gap: 16px; padding: 24px; }
.dsr-item { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.dsr-val { font-family: var(--mono); font-size: 24px; color: var(--t1); font-weight: 500; }
.dsr-label { font-family: var(--body); font-size: 10px; color: var(--t3); text-transform: uppercase; letter-spacing: 0.05em; }

/* API Section */
.api-block {
  background: #000;
  padding: 16px;
  border-radius: var(--radius);
  font-family: var(--mono);
  font-size: 12px;
  color: var(--t2);
  margin-top: 12px;
  overflow-x: auto;
}
.api-block code {
  color: var(--g);
}
.api-desc {
  font-family: var(--body);
  font-size: 13px;
  color: var(--t2);
  margin-bottom: 8px;
}
"""
'''

    new_api_func = '''
def _api_section_html():
    return """
    <div style="padding: 24px;">
        <div class="api-desc">DataForge Arena provides a full OpenEnv REST API for programmatic evaluation.</div>
        
        <div class="api-desc" style="margin-top: 20px;"><b>Reset environment (cURL)</b></div>
        <div class="api-block">
<pre>curl -X POST https://vivek567-dataforge-arena.hf.space/reset \\
  -H "Content-Type: application/json" \\
  -d '{"tier": 1}'</pre>
        </div>
        
        <div class="api-desc" style="margin-top: 20px;"><b>Execute one repair step (cURL)</b></div>
        <div class="api-block">
<pre>curl -X POST https://vivek567-dataforge-arena.hf.space/step \\
  -H "Content-Type: application/json" \\
  -d '{"reasoning": "age 145 exceeds schema max 120", "tool_id": 3, "column": 2, "row_id": 7}'</pre>
        </div>

        <div class="api-desc" style="margin-top: 20px;"><b>Python Requests Example</b></div>
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
'''

    new_build_demo = '''def build_demo():
    choices = available_agent_choices()
    default = "Live GRPO Model · Trained on 265 RL steps [GRPO]" if "Live GRPO Model" in choices else "Heuristic Surgeon · Rule-based constraint-aware repairs [DETERMINISTIC]"

    agent_choices = [
        "Naive Baseline · Greedy imputation, no schema awareness [BASELINE]",
        "Heuristic Surgeon · Rule-based constraint-aware repairs [DETERMINISTIC]",
    ]
    if "Live GRPO Model" in choices:
        agent_choices.append("Live GRPO Model · Trained on 265 RL steps [GRPO]")
    else:
        agent_choices.append("Live GRPO Model · Checkpoint Required [UNAVAILABLE]")

    tier_choices = [
        "Tier 1 · Nulls, type errors, range violations",
        "Tier 2 · FK mismatches, temporal drift",
        "Tier 3 · Full relational reasoning",
    ]

    with gr.Blocks(title="DataForge Arena") as demo:
        state = gr.State(_new_state())
        
        # Header bar (keep existing _topbar_html() logic)
        topbar = gr.HTML(_topbar_html())
        
        with gr.Row(equal_height=False):
            # LEFT SIDEBAR
            with gr.Column(scale=1, min_width=300, elem_classes=["sidebar"]):
                gr.HTML("<div class='pane-label'>Agent</div>")
                agent_pick = gr.Radio(choices=agent_choices, value=default, label="", interactive=True)
                
                gr.HTML("<div class='pane-label' style='margin-top:24px;'>Complexity</div>")
                tier_pick = gr.Radio(choices=tier_choices, value="Tier 1 · Nulls, type errors, range violations", label="", interactive=True)
                
                gen_btn = gr.Button("New Scenario", variant="primary", elem_classes=["primary"])
                
                gr.HTML("<div class='pane-label' style='margin-top:32px;'>Benchmark Performance</div>")
                bench_html = gr.HTML(_benchmark_html())
                
                gr.HTML("<div class='pane-label' style='margin-top:24px;'>Training Curve · 265 steps</div>")
                reward_spark = gr.LinePlot(x="step", y="total_reward", height=100, x_title="Step", y_title="Reward", tooltip=["step", "total_reward"])
            
            # MAIN CONTENT
            with gr.Column(scale=3, elem_classes=["main-content"]):
                # Status strip
                status_strip = gr.HTML("<div class='status-strip'><span>Tier: <b>1</b></span><span>Status: <b>Waiting</b></span><span>Accuracy: <b>--</b></span></div>")
                
                # Data view (collapsible)
                with gr.Accordion("Corrupted Data", open=True):
                    dirty_view = gr.Dataframe(label="", interactive=False, wrap=False)
                
                # Execute button
                exec_btn = gr.Button("▶ Execute Agent", variant="primary", size="lg", elem_classes=["primary"])
                
                # Streaming output
                rollout_out = gr.HTML(_empty_rollout())
                
                # Results row
                with gr.Row():
                    acc_display = gr.HTML(_accuracy_display(None, None))
                    repaired_view = gr.Dataframe(label="", interactive=False, wrap=False)
                
                diff_out = gr.HTML(_diff_html(None, None, None))
        
        # API section at bottom
        with gr.Accordion("Use via API", open=False):
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
            
            stats = f"<div class='status-strip'><span>Tier: <b>{tier}</b></span><span>Rows: <b>{len(dirty)}</b></span><span>Initial Acc: <b>{acc:.1%}</b></span></div>"
            return dirty.head(8), stats, s_out, _empty_rollout(), _accuracy_display(acc, acc), dirty.head(8), _diff_html(dirty, dirty, gt)

        gen_outs = [dirty_view, status_strip, state, rollout_out, acc_display, repaired_view, diff_out]
        
        gen_btn.click(fn=on_gen, inputs=[tier_pick, state], outputs=gen_outs)

        # After execution update repaired view — we need to also return repaired_view
        # Wrap simulate_agent to yield repaired data too
        def simulate_with_repaired(agent_val, session_state):
            # Parse agent type from the long choice string
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

    final_content = content[:css_start] + new_css + new_api_func + new_build_demo + content[build_demo_end:]
    
    with open('demo/app.py', 'w', encoding='utf-8') as f:
        f.write(final_content)

if __name__ == '__main__':
    main()
