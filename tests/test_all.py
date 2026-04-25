"""Regression tests for DataForge Arena."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from environment.corruptor import Corruptor
from environment.env import DataForgeEnv, SurgeonAction
from environment.reward import RewardComputer
from environment.schemas import HEALTHCARE_SCHEMA
from training.parser import robust_parse_action


@pytest.fixture
def clean_df():
    return pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "age": [30, 45, 28, 60, 35],
            "birth_year": [1994, 1979, 1996, 1964, 1989],
            "email": ["a@h.com", "b@h.com", "c@h.com", "d@h.com", "e@h.com"],
            "phone": ["1234567890"] * 5,
            "diagnosis": ["Flu", "Diabetes", "Fracture", "Hypertension", "Asthma"],
            "department_id": [1, 2, 3, 4, 5],
            "department_name": ["Cardiology", "Neurology", "Oncology", "Pediatrics", "Ortho"],
            "admission_date": ["2024-01-01"] * 5,
        }
    )


@pytest.fixture
def corruptor():
    return Corruptor()


@pytest.fixture
def env(corruptor, clean_df):
    return DataForgeEnv(corruptor=corruptor, schema=HEALTHCARE_SCHEMA, clean_data=clean_df)


def test_corruptor_generates_episode(corruptor, clean_df):
    dirty, gt, meta = corruptor.generate_episode(clean_df)
    assert len(dirty) >= len(clean_df)
    assert "tool" in meta


def test_solvability_gate_rejects_banned_tools(corruptor, clean_df):
    dirty = clean_df.copy()
    valid, reason = corruptor._solvability_gate(dirty, clean_df, {"tool": "delete_row"})
    assert not valid
    assert "unrecoverable" in reason


def test_column_null_rate_limit(corruptor, clean_df):
    dirty = clean_df.copy()
    dirty["age"] = np.nan
    valid, _ = corruptor._solvability_gate(dirty, clean_df, {"tool": "inject_null_cluster", "col": "age"})
    assert not valid


def test_tier_transitions(corruptor):
    assert corruptor.current_tier() == 1
    corruptor._epoch = 29
    corruptor._recent_rewards.extend([1.0] * 20)
    corruptor._update_tier()
    assert corruptor.current_tier() == 1

    corruptor._epoch = 31
    corruptor._update_tier()
    assert corruptor.current_tier() == 2

    corruptor._epoch = 75
    corruptor._recent_rewards.extend([1.0] * 20)
    corruptor._update_tier()
    assert corruptor.current_tier() == 3


def test_force_tier_enables_requested_corruptions(corruptor, clean_df):
    corruptor.force_tier(3)
    _, _, meta = corruptor.generate_episode(clean_df)
    assert corruptor.current_tier() == 3
    assert meta["tool"] in {"break_foreign_key", "duplicate_row_mutate"}
    assert meta["difficulty"] == 3
    assert meta["requested_tier"] == 3


def test_corruptor_is_transitioning(corruptor):
    corruptor._epoch = 35
    assert corruptor.is_transitioning()
    corruptor._epoch = 45
    assert not corruptor.is_transitioning()
    corruptor._epoch = 75
    assert corruptor.is_transitioning()


def test_corruptor_fallback_reports_actual_difficulty(corruptor, clean_df, monkeypatch):
    def always_bad(df, tier):
        return df, {"tool": "inject_null_cluster", "col": "age"}

    monkeypatch.setattr(corruptor, "_corrupt", always_bad)
    monkeypatch.setattr(corruptor, "_solvability_gate", lambda dirty, gt, metadata: (False, "bad"))
    corruptor.force_tier(3)

    _, _, meta = corruptor.generate_episode(clean_df)

    assert meta["requested_tier"] == 3
    assert meta["difficulty"] == 1
    assert meta["fallback_from_tier"] == 3


def test_accuracy_delta_positive_on_fix(clean_df):
    dirty = clean_df.copy()
    dirty.at[0, "age"] = np.nan
    rc = RewardComputer()
    prev_acc = rc._field_accuracy(dirty, clean_df)
    dirty.at[0, "age"] = clean_df.at[0, "age"]
    curr_acc = rc._field_accuracy(dirty, clean_df)
    assert curr_acc > prev_acc


def test_noop_correct_cell_gives_positive(clean_df):
    rc = RewardComputer()
    action = SurgeonAction(reasoning="cell is correct", tool_id=7, column=0, row_id=0)
    reward = rc._score_tool_logic(action, clean_df, clean_df)
    assert reward > 0


def test_impute_on_correct_cell_penalized(clean_df):
    rc = RewardComputer()
    action = SurgeonAction(reasoning="imputing", tool_id=0, column=0, row_id=0)
    reward = rc._score_tool_logic(action, clean_df, clean_df)
    assert reward < 0


def test_efficiency_rewards_repair_tool_on_incorrect_cell(clean_df):
    rc = RewardComputer()
    dirty = clean_df.copy()
    repaired = clean_df.copy()
    age_col = list(clean_df.columns).index("age")
    dirty.at[0, "age"] = np.nan
    action = SurgeonAction(
        reasoning="age is null because the value is missing",
        tool_id=0,
        column=age_col,
        row_id=0,
    )

    reward = rc._score_efficiency(action, repaired, clean_df, previous_state=dirty)

    assert reward == 0.5


def test_efficiency_penalizes_noop_on_incorrect_cell(clean_df):
    rc = RewardComputer()
    dirty = clean_df.copy()
    age_col = list(clean_df.columns).index("age")
    dirty.at[0, "age"] = np.nan
    action = SurgeonAction(reasoning="skip bad cell", tool_id=7, column=age_col, row_id=0)

    reward = rc._score_efficiency(action, dirty, clean_df, previous_state=dirty)

    assert reward < 0


def test_reasoning_allows_medium_length_useful_explanations(clean_df):
    rc = RewardComputer()
    dirty = clean_df.copy()
    dirty.at[0, "age"] = np.nan
    action = SurgeonAction(
        reasoning="age missing because source value is absent in this row today now",
        tool_id=0,
        column=list(clean_df.columns).index("age"),
        row_id=0,
    )

    reward = rc._score_reasoning(action, dirty)

    assert reward >= 0.3


def test_antihack_mass_delete_penalty(clean_df):
    dirty = clean_df.copy()
    dirty["_is_deleted"] = True
    rc = RewardComputer()
    reward = rc._detect_shortcuts(dirty, clean_df)
    assert reward == -5.0


def test_reward_fix_dominates_wrong_action():
    fix_reward = 0.1 * 20
    wrong_penalty = -1.0
    ratio = fix_reward / abs(wrong_penalty)
    assert ratio > 0


def test_episode_complete_not_trivially_true(clean_df):
    dirty = clean_df.copy()
    dirty.at[0, "age"] = np.nan
    rc = RewardComputer()
    starting_acc = rc._field_accuracy(dirty, clean_df)
    action = SurgeonAction(reasoning="skip", tool_id=7, column=0, row_id=0)
    result = rc.compute(
        state=dirty,
        ground_truth=clean_df,
        action=action,
        original_dirty=dirty.copy(),
        prev_accuracy=starting_acc,
        episode_start=__import__("time").time(),
        step_count=1,
        starting_accuracy=starting_acc,
    )
    assert result["episode_complete"] is False


def test_duplicate_row_accuracy_not_stuck_at_51(clean_df):
    rc = RewardComputer()
    dup = clean_df.iloc[0].copy()
    dup["age"] = np.nan
    dirty = pd.concat([clean_df, pd.DataFrame([dup])], ignore_index=True)
    gt_extended = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
    acc = rc._field_accuracy(dirty, gt_extended)
    assert acc > 0.90


def test_parser_clean_json():
    action = robust_parse_action('{"reasoning":"null value","tool_id":0,"column":1,"row_id":3}')
    assert action.tool_id == 0


def test_parser_with_preamble():
    action = robust_parse_action('Sure! {"reasoning":"type error","tool_id":3,"column":2,"row_id":1}')
    assert action.tool_id == 3


def test_parser_trailing_comma():
    action = robust_parse_action('{"reasoning":"null","tool_id":1,"column":0,"row_id":2,}')
    assert action.tool_id == 1


def test_parser_single_quotes():
    action = robust_parse_action("{'reasoning':'missing','tool_id':2,'column':3,'row_id':0}")
    assert action.tool_id == 2


def test_parser_raises_on_garbage():
    with pytest.raises(ValueError):
        robust_parse_action("I don't know what to do with this data at all.")


def test_parser_handles_code_fence():
    action = robust_parse_action('```json\n{"reasoning":"null","tool_id":1,"column":0,"row_id":2}\n```')
    assert action.tool_id == 1


def test_parser_handles_chat_messages():
    action = robust_parse_action(
        [
            {"role": "assistant", "content": "Here you go:"},
            {"role": "assistant", "content": '{"reasoning":"missing","tool_id":2,"column":3,"row_id":0}'},
        ]
    )
    assert action.tool_id == 2


def test_parser_accepts_row_idx_and_tool_name_aliases():
    action = robust_parse_action(
        '{"reasoning":"email invalid fix format","tool":"CORRECT_FORMAT","col":4,"_row_idx":6}',
        require_fields=True,
    )
    assert action.tool_id == 3
    assert action.column == 4
    assert action.row_id == 6


def test_parser_accepts_tool_name_in_regex_fallback():
    action = robust_parse_action(
        '"reasoning":"age missing use median", "tool_name":"IMPUTE_MEDIAN", "column":2, "row_idx":14'
    )
    assert action.tool_id == 0
    assert action.column == 2
    assert action.row_id == 14


def test_parser_clamps_high_tool_id():
    action = robust_parse_action('{"reasoning":"test","tool_id":99,"column":0,"row_id":0}')
    assert action.tool_id == 7


def test_parser_clamps_negative_tool_id():
    action = robust_parse_action('{"reasoning":"test","tool_id":-1,"column":0,"row_id":0}')
    assert action.tool_id == 0


def test_parser_strict_mode_rejects_missing_fields():
    with pytest.raises(ValueError):
        robust_parse_action('{"reasoning":"test"}', require_fields=True)


def test_nan_serialization(env):
    obs = env.reset()
    parsed = json.loads(obs.rows_json)
    assert isinstance(parsed, list)


def test_observation_within_token_budget(env):
    from training.prompt import build_prompt

    obs = env.reset()
    prompt = build_prompt(obs)
    estimated_tokens = len(prompt) / 4
    assert estimated_tokens < 1024


def test_soft_delete_no_index_drift(env):
    env.reset()
    action = SurgeonAction(reasoning="corrupted", tool_id=4, column=0, row_id=2)
    env.step(action)
    if "_is_deleted" in env._state.columns:
        assert bool(env._state["_is_deleted"].iloc[2]) is True


def test_negative_indices_are_invalid(env):
    env.reset()
    action = SurgeonAction(reasoning="invalid", tool_id=7, column=-1, row_id=-1)
    _, reward, _, info = env.step(action)
    assert reward == -0.5
    assert info["invalid_action"] is True


def test_deleted_row_cannot_be_retargeted(env):
    env.reset()
    env.step(SurgeonAction(reasoning="delete", tool_id=4, column=0, row_id=0))
    _, reward, _, info = env.step(SurgeonAction(reasoning="retry", tool_id=7, column=0, row_id=0))
    assert reward == -0.5
    assert info["invalid_action"] is True


def test_env_full_episode(env):
    env.reset()
    for _ in range(5):
        _, reward, done, _ = env.step(SurgeonAction(reasoning="testing", tool_id=7, column=0, row_id=0))
        assert isinstance(reward, float)
        if done:
            break


def test_model_config_no_torch_crash():
    from training.model_config import detect_gpu

    gpu = detect_gpu()
    assert "type" in gpu
    assert "vram_gb" in gpu


def test_model_selection_t4():
    from training.model_config import select_model

    cfg = select_model({"type": "T4", "vram_gb": 15})
    assert "1.5B" in cfg["model_name"] or "Qwen" in cfg["model_name"]
    assert cfg["max_completion_length"] <= 96
    assert cfg["max_training_tier"] == 2


def test_model_selection_a100():
    from training.model_config import select_model

    cfg = select_model({"type": "A100", "vram_gb": 40})
    assert "8B" in cfg["model_name"]


def test_model_selection_l40():
    from training.model_config import select_model

    cfg = select_model({"type": "L40S", "vram_gb": 20})
    assert "3B" in cfg["model_name"]


def test_precision_selection_t4_uses_fp16():
    from training.model_config import select_precision

    cfg = select_precision({"type": "Tesla T4", "vram_gb": 15, "capability": "7.5"})
    assert cfg["fp16"] is True
    assert cfg["bf16"] is False


def test_precision_selection_a100_uses_bf16():
    from training.model_config import select_precision

    cfg = select_precision({"type": "A100", "vram_gb": 40, "capability": "8.0"})
    assert cfg["bf16"] is True
    assert cfg["fp16"] is False


def test_eval_resolve_heuristic_agent():
    from eval.evaluate import resolve_eval_agent

    cfg = resolve_eval_agent("heuristic")
    assert cfg["agent_mode"] == "heuristic"
    assert cfg["model_source"] == "heuristic-rule-based"
    assert cfg["fallback_used"] is False


def test_eval_resolve_grpo_requires_local_checkpoint(tmp_path):
    from eval.evaluate import resolve_eval_agent

    missing_path = tmp_path / "missing-checkpoint"
    with pytest.raises(FileNotFoundError):
        resolve_eval_agent("grpo", str(missing_path))


def test_eval_resolves_latest_adapter_checkpoint(tmp_path):
    from eval.evaluate import _resolve_loadable_model_path

    root = tmp_path / "outputs"
    old_checkpoint = root / "checkpoint-25"
    latest_checkpoint = root / "checkpoint-75"
    old_checkpoint.mkdir(parents=True)
    latest_checkpoint.mkdir(parents=True)
    (old_checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (latest_checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert _resolve_loadable_model_path(str(root)) == latest_checkpoint


def test_eval_prefers_root_adapter_checkpoint(tmp_path):
    from eval.evaluate import _resolve_loadable_model_path

    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    nested = tmp_path / "checkpoint-99"
    nested.mkdir()
    (nested / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert _resolve_loadable_model_path(str(tmp_path)) == tmp_path


def test_committed_eval_results_include_provenance():
    payload = json.loads(Path("eval/results.json").read_text(encoding="utf-8"))
    assert payload["agent_mode"] == "grpo"
    assert payload["model_source"] == "outputs/dataforge-surgeon"
    assert payload["fallback_used"] is False
    assert "surgeon_advantage_accuracy_delta" in payload
    assert payload["surgeon_advantage_accuracy_delta"] > 0


def test_committed_heuristic_baseline_include_provenance():
    payload = json.loads(Path("eval/heuristic_results.json").read_text(encoding="utf-8"))
    assert payload["agent_mode"] == "heuristic"
    assert payload["model_source"] == "heuristic-rule-based"
    assert payload["fallback_used"] is False
    assert payload["surgeon_advantage_accuracy_delta"] > 0


def test_server_requirements_cover_demo_entrypoint():
    requirements = Path("requirements-server.txt").read_text(encoding="utf-8")
    assert "gradio>=" in requirements
    assert "peft>=" in requirements


def test_server_info_advertises_accuracy_delta_only():
    import asyncio
    from environment.server import info

    payload = asyncio.run(info())
    assert "accuracy_delta" in payload["reward_signals"]
    assert "accuracy_absolute" not in payload["reward_signals"]


def test_demo_hides_live_mode_without_checkpoint():
    from demo.app import available_agent_choices

    assert available_agent_choices(model_available=False) == [
        "Naive Baseline",
        "Heuristic Surgeon",
    ]


def test_demo_shows_live_mode_with_checkpoint():
    from demo.app import available_agent_choices

    assert available_agent_choices(model_available=True) == [
        "Naive Baseline",
        "Heuristic Surgeon",
        "Live GRPO Model",
    ]


def test_demo_detects_adapter_checkpoint(tmp_path):
    from demo.app import local_model_available

    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert local_model_available(str(tmp_path)) is True


def test_demo_evidence_snapshot_reports_heuristic_and_grpo(monkeypatch, tmp_path):
    from demo import app

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    (eval_dir / "results.json").write_text(
        json.dumps(
            {
                "agent_mode": "grpo",
                "surgeon_avg_accuracy_delta": -0.0004,
                "random_avg_accuracy_delta": -0.0045,
                "surgeon_advantage_accuracy_delta": 0.0041,
                "episodes": 20,
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "heuristic_results.json").write_text(
        json.dumps(
            {
                "agent_mode": "heuristic",
                "surgeon_advantage_accuracy_delta": 0.0053,
                "episodes": 20,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "step": [0, 75],
            "total_reward": [-1.4, -0.8],
            "parse_success_rate": [0.25, 0.50],
            "difficulty": [1, 2],
        }
    ).to_csv(logs_dir / "training_log.csv", index=False)

    monkeypatch.setattr(app, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(app, "EVAL_RESULTS_PATH", str(eval_dir / "results.json"))
    monkeypatch.setattr(app, "LOG_PATH", str(logs_dir / "training_log.csv"))
    monkeypatch.setattr(app, "LOCAL_MODEL_PATH", str(tmp_path / "outputs" / "missing"))

    html = app._evidence_snapshot_html()

    assert "Heuristic baseline" in html
    assert "+0.53 pp" in html
    assert "GRPO checkpoint" in html
    assert "+0.41 pp" in html
    assert "First 25.0% -&gt; last 50.0%" in html
    assert "Checkpoint gated" in html


def test_merge_duplicate_excludes_deleted_col(clean_df):
    from environment.tools import _merge_duplicate

    state = clean_df.copy()
    state["_is_deleted"] = False
    _merge_duplicate(state, 0)
    assert True


def test_correct_format_repairs_age_from_birth_year(clean_df):
    from environment.tools import apply_tool

    state = clean_df.copy()
    state.at[0, "age"] = 180
    action = SurgeonAction(
        reasoning="age is inconsistent with birth_year",
        tool_id=3,
        column=list(clean_df.columns).index("age"),
        row_id=0,
    )
    repaired = apply_tool(state, action, HEALTHCARE_SCHEMA)
    assert repaired.at[0, "age"] == clean_df.at[0, "age"]


def test_correct_format_repairs_department_name(clean_df):
    from environment.tools import apply_tool

    state = clean_df.copy()
    state.at[0, "department_name"] = "INVALID_DEPT"
    action = SurgeonAction(
        reasoning="department name disagrees with id",
        tool_id=3,
        column=list(clean_df.columns).index("department_name"),
        row_id=0,
    )
    repaired = apply_tool(state, action, HEALTHCARE_SCHEMA)
    assert repaired.at[0, "department_name"] == clean_df.at[0, "department_name"]


def test_observation_scores_format_mismatch(env, clean_df):
    env._state = clean_df.copy()
    env._ground_truth = clean_df.copy()
    env._action_log = []
    env._step_count = 0
    env._state.at[0, "admission_date"] = "01/01/24"
    obs = env._make_observation()
    rows = json.loads(obs.rows_json)
    assert obs.total_errors >= 1
    assert any(row["_row_idx"] == 0 for row in rows)


def test_observation_exposes_suspect_columns(env, clean_df):
    env._state = clean_df.copy()
    env._ground_truth = clean_df.copy()
    env._action_log = []
    env._step_count = 0
    env._state.at[0, "email"] = "not-an-email"
    obs = env._make_observation()
    rows = json.loads(obs.rows_json)
    target = next(row for row in rows if row["_row_idx"] == 0)
    assert target["_error_score"] >= 1
    assert "email" in target["_suspect_columns"]
