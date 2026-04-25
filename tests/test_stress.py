"""Stress and edge-case coverage for DataForge Arena."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from environment.corruptor import Corruptor
from environment.env import DataForgeEnv, SurgeonAction
from environment.reward import RewardComputer
from environment.schemas import DEPT_MAP, FINANCIAL_SCHEMA, HEALTHCARE_SCHEMA, SURGEON_TOOLS
from environment.tools import apply_tool
from environment.validation import (
    cell_has_error,
    expected_department_id,
    expected_department_name,
    infer_reference_year,
)
from training.parser import robust_parse_action


@pytest.fixture
def clean_hc():
    return pd.DataFrame(
        {
            "patient_id": list(range(1, 11)),
            "name": [f"P{i}" for i in range(10)],
            "age": [30 + i for i in range(10)],
            "birth_year": [1994 - i for i in range(10)],
            "email": [f"p{i}@test.com" for i in range(10)],
            "phone": ["1234567890"] * 10,
            "diagnosis": ["Flu"] * 10,
            "department_id": [(i % 10) + 1 for i in range(10)],
            "department_name": [DEPT_MAP[(i % 10) + 1] for i in range(10)],
            "admission_date": ["2024-03-15"] * 10,
        }
    )


@pytest.fixture
def rc():
    return RewardComputer()


@pytest.fixture
def corr():
    return Corruptor()


@pytest.fixture
def env(corr, clean_hc):
    return DataForgeEnv(corruptor=corr, schema=HEALTHCARE_SCHEMA, clean_data=clean_hc)


def test_full_episode_no_crash(env):
    env.reset()
    for _ in range(env.MAX_STEPS):
        obs, _, done, _ = env.step(SurgeonAction(reasoning="scan", tool_id=7, column=0, row_id=0))
        if done:
            break
    assert obs.step_count <= env.MAX_STEPS


def test_all_tools_valid_in_episode(env):
    for tool_id in range(8):
        env.reset()
        _, reward, done, info = env.step(SurgeonAction(reasoning="test", tool_id=tool_id, column=0, row_id=0))
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "step" in info


def test_reward_is_finite_for_all_tools(env):
    for tool_id in range(8):
        env.reset()
        _, reward, _, _ = env.step(SurgeonAction(reasoning="test", tool_id=tool_id, column=0, row_id=0))
        assert math.isfinite(reward)


def test_step_count_increments(env):
    env.reset()
    for expected in range(1, 6):
        obs, _, done, _ = env.step(SurgeonAction(reasoning="x", tool_id=7, column=0, row_id=0))
        if done:
            break
        assert obs.step_count == expected


def test_episode_ends_at_max_steps(env):
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(SurgeonAction(reasoning="noop", tool_id=7, column=0, row_id=0))
        steps += 1
        if steps > env.MAX_STEPS + 5:
            pytest.fail("Episode did not terminate within MAX_STEPS")
    assert steps <= env.MAX_STEPS


def test_tier3_episode_generates_without_crash(corr, clean_hc):
    corr.force_tier(3)
    dirty, _, meta = corr.generate_episode(clean_hc)
    assert dirty is not None
    assert meta["difficulty"] in {1, 3}


def test_multiple_resets_clear_rollout_state(env):
    env.reset()
    env.step(SurgeonAction(reasoning="x", tool_id=0, column=0, row_id=0))
    obs2 = env.reset()
    assert obs2.step_count == 0
    assert env._action_log == []
    assert env._step_count == 0


VALID_PARSEABLE = [
    '{"reasoning":"fix null","tool_id":0,"column":1,"row_id":2}',
    '{"reasoning":"type err","tool_id":3,"column":4,"row_id":0}',
    '{"reasoning":"dup","tool_id":5,"column":0,"row_id":9}',
    '{"reasoning":"del","tool_id":4,"column":0,"row_id":1}',
    '{"reasoning":"x","tool":"IMPUTE_MEDIAN","col":2,"row":1}',
    '{"reasoning":"x","tool_name":"NO_OP","col_idx":0,"row_idx":0}',
    '{"reasoning":"x","action":"DELETE_ROW","column_idx":0,"_row_idx":0}',
    "{'reasoning':'x','tool_id':1,'column':0,'row_id':0}",
    '{"reasoning":"fix","tool_id":2,"column":0,"row_id":0,}',
    '{"reasoning":"x","tool_id":"0","column":0,"row_id":0}',
    '{"reasoning":"x","tool_id":3.0,"column":1.0,"row_id":2.0}',
    '{"reasoning":"x","tool_id":99,"column":0,"row_id":0}',
    '{"reasoning":"x","tool_id":1,"column":-3,"row_id":-1}',
    '```json\n{"reasoning":"fix","tool_id":0,"column":0,"row_id":0}\n```',
]


INVALID_NOT_PARSEABLE = [
    "",
    "   ",
    "garbage text with no json",
    "null",
    "[]",
    "123",
    '{"key":"no action fields at all"}',
]


@pytest.mark.parametrize("raw", VALID_PARSEABLE)
def test_parser_handles_valid_input(raw):
    action = robust_parse_action(raw)
    assert 0 <= action.tool_id <= 7
    assert action.row_id >= 0
    assert action.column >= 0


@pytest.mark.parametrize("raw", INVALID_NOT_PARSEABLE)
def test_parser_raises_on_truly_invalid(raw):
    with pytest.raises(ValueError):
        robust_parse_action(raw, require_fields=True)


def test_parser_never_returns_out_of_range_tool_id():
    for bad_id in [-5, -1, 8, 100, 9999]:
        raw = f'{{"reasoning":"x","tool_id":{bad_id},"column":0,"row_id":0}}'
        action = robust_parse_action(raw)
        assert 0 <= action.tool_id <= 7


def test_parser_handles_all_tool_name_strings():
    for tool_id, info in SURGEON_TOOLS.items():
        raw = f'{{"reasoning":"x","tool_id":"{info["name"]}","column":0,"row_id":0}}'
        action = robust_parse_action(raw)
        assert action.tool_id == tool_id


def test_anti_hack_triggers_above_25_pct_deletion(rc, clean_hc):
    state = clean_hc.copy()
    state["_is_deleted"] = False
    state.at[0, "_is_deleted"] = True
    state.at[1, "_is_deleted"] = True
    state.at[2, "_is_deleted"] = True
    assert rc._detect_shortcuts(state, clean_hc) == -5.0


def test_anti_hack_silent_below_25_pct(rc, clean_hc):
    state = clean_hc.copy()
    state["_is_deleted"] = False
    state.at[0, "_is_deleted"] = True
    assert rc._detect_shortcuts(state, clean_hc) == 0.0


def test_accuracy_always_in_unit_interval(rc, clean_hc):
    dirty = clean_hc.copy()
    dirty["_is_deleted"] = False
    dirty.at[0, "_is_deleted"] = True
    dirty.at[0, "age"] = float("nan")
    acc = rc._field_accuracy(dirty, clean_hc)
    assert 0.0 <= acc <= 1.0


def test_field_accuracy_perfect_match(rc, clean_hc):
    assert rc._field_accuracy(clean_hc, clean_hc) == pytest.approx(1.0)


def test_field_accuracy_all_null(rc, clean_hc):
    dirty = clean_hc.copy().astype(object)
    for col in dirty.columns:
        dirty[col] = float("nan")
    assert rc._field_accuracy(dirty, clean_hc) == pytest.approx(0.0)


def test_reward_compute_returns_total_key(rc, clean_hc):
    state = clean_hc.copy()
    state.at[0, "age"] = float("nan")
    action = SurgeonAction(reasoning="null age impute", tool_id=0, column=2, row_id=0)
    import time

    result = rc.compute(
        state=state,
        ground_truth=clean_hc,
        action=action,
        original_dirty=state,
        prev_accuracy=0.95,
        episode_start=time.time(),
        step_count=1,
        starting_accuracy=0.95,
        previous_state=state,
    )
    assert "total" in result
    assert isinstance(result["total"], float)


def test_timeout_returns_negative_reward(rc, clean_hc):
    import time

    result = rc.compute(
        state=clean_hc,
        ground_truth=clean_hc,
        action=SurgeonAction(reasoning="x", tool_id=7, column=0, row_id=0),
        original_dirty=clean_hc,
        prev_accuracy=1.0,
        episode_start=time.time() - 35,
        step_count=1,
        starting_accuracy=1.0,
        previous_state=clean_hc,
    )
    assert result.get("timeout") is True
    assert result["total"] == -3.0


def test_impute_median_fills_null_numeric(clean_hc):
    state = clean_hc.copy()
    state.at[0, "age"] = float("nan")
    action = SurgeonAction(reasoning="null", tool_id=0, column=list(clean_hc.columns).index("age"), row_id=0)
    repaired = apply_tool(state, action, HEALTHCARE_SCHEMA)
    assert pd.notna(repaired.at[0, "age"])


def test_impute_mode_fills_null_string(clean_hc):
    state = clean_hc.copy()
    state.at[0, "diagnosis"] = float("nan")
    action = SurgeonAction(reasoning="missing diag", tool_id=1, column=list(clean_hc.columns).index("diagnosis"), row_id=0)
    repaired = apply_tool(state, action, HEALTHCARE_SCHEMA)
    assert pd.notna(repaired.at[0, "diagnosis"])


def test_forward_fill_uses_previous_row(clean_hc):
    state = clean_hc.copy()
    state.at[2, "diagnosis"] = float("nan")
    action = SurgeonAction(reasoning="ffill", tool_id=2, column=list(clean_hc.columns).index("diagnosis"), row_id=2)
    repaired = apply_tool(state, action, HEALTHCARE_SCHEMA)
    assert repaired.at[2, "diagnosis"] == clean_hc.at[1, "diagnosis"]


def test_delete_row_soft_deletes(clean_hc):
    repaired = apply_tool(
        clean_hc.copy(),
        SurgeonAction(reasoning="del", tool_id=4, column=0, row_id=0),
        HEALTHCARE_SCHEMA,
    )
    assert "_is_deleted" in repaired.columns
    assert bool(repaired.at[0, "_is_deleted"]) is True


def test_delete_already_deleted_row_is_noop(clean_hc):
    state = clean_hc.copy()
    state["_is_deleted"] = False
    state.at[0, "_is_deleted"] = True
    repaired = apply_tool(
        state,
        SurgeonAction(reasoning="del", tool_id=4, column=0, row_id=0),
        HEALTHCARE_SCHEMA,
    )
    assert repaired["_is_deleted"].sum() == 1


def test_correct_format_normalizes_date(clean_hc):
    state = clean_hc.copy()
    state.at[0, "admission_date"] = "03/15/2024"
    repaired = apply_tool(
        state,
        SurgeonAction(reasoning="date format", tool_id=3, column=list(clean_hc.columns).index("admission_date"), row_id=0),
        HEALTHCARE_SCHEMA,
    )
    assert repaired.at[0, "admission_date"] == "2024-03-15"


def test_correct_format_repairs_phone_digits(clean_hc):
    state = clean_hc.copy()
    state.at[0, "phone"] = "(123) 456-7890"
    repaired = apply_tool(
        state,
        SurgeonAction(reasoning="phone fmt", tool_id=3, column=list(clean_hc.columns).index("phone"), row_id=0),
        HEALTHCARE_SCHEMA,
    )
    digits = "".join(ch for ch in str(repaired.at[0, "phone"]) if ch.isdigit())
    assert len(digits) >= 10


def test_noop_and_flag_do_not_change_visible_values(clean_hc):
    for tool_id in [6, 7]:
        state = clean_hc.copy()
        before = state.copy()
        repaired = apply_tool(
            state,
            SurgeonAction(reasoning="x", tool_id=tool_id, column=0, row_id=0),
            HEALTHCARE_SCHEMA,
        )
        pd.testing.assert_frame_equal(
            repaired.drop(columns=["_is_deleted"], errors="ignore"),
            before.drop(columns=["_is_deleted"], errors="ignore"),
            check_dtype=False,
        )


def test_out_of_bounds_action_does_not_crash(clean_hc):
    repaired = apply_tool(
        clean_hc.copy(),
        SurgeonAction(reasoning="oob", tool_id=0, column=999, row_id=999),
        HEALTHCARE_SCHEMA,
    )
    assert isinstance(repaired, pd.DataFrame)


def test_cell_has_error_null_is_always_error():
    assert cell_has_error(float("nan"), "age", HEALTHCARE_SCHEMA)
    assert cell_has_error(float("nan"), "name", HEALTHCARE_SCHEMA)
    assert cell_has_error(None, "diagnosis", HEALTHCARE_SCHEMA)


def test_cell_has_error_err_prefix_always_error():
    for col in ["age", "name", "email"]:
        assert cell_has_error("ERR_42", col, HEALTHCARE_SCHEMA)


def test_cell_has_error_valid_values_not_error():
    row = pd.Series({"age": 30, "birth_year": 1994})
    assert not cell_has_error(30, "age", HEALTHCARE_SCHEMA, row=row)
    assert not cell_has_error("a@b.com", "email", HEALTHCARE_SCHEMA)
    assert not cell_has_error("2024-01-01", "admission_date", HEALTHCARE_SCHEMA)
    assert not cell_has_error("1234567890", "phone", HEALTHCARE_SCHEMA)


def test_cell_has_error_out_of_range_age():
    assert cell_has_error(150, "age", HEALTHCARE_SCHEMA)
    assert cell_has_error(-1, "age", HEALTHCARE_SCHEMA)
    assert not cell_has_error(25, "age", HEALTHCARE_SCHEMA)


def test_cell_has_error_invalid_email():
    assert cell_has_error("not-an-email", "email", HEALTHCARE_SCHEMA)
    assert cell_has_error("missing@dot", "email", HEALTHCARE_SCHEMA)
    assert not cell_has_error("valid@domain.com", "email", HEALTHCARE_SCHEMA)


def test_cell_has_error_invalid_date_formats():
    assert cell_has_error("01/01/2024", "admission_date", HEALTHCARE_SCHEMA)
    assert cell_has_error("2024/01/01", "admission_date", HEALTHCARE_SCHEMA)
    assert not cell_has_error("2024-01-01", "admission_date", HEALTHCARE_SCHEMA)


def test_infer_reference_year_consistent(clean_hc):
    ref = infer_reference_year(clean_hc)
    for _, row in clean_hc.iterrows():
        assert abs((row["age"] + row["birth_year"]) - ref) <= 2


def test_dept_map_round_trips():
    for dept_id, dept_name in DEPT_MAP.items():
        assert expected_department_name(dept_id) == dept_name
        assert expected_department_id(dept_name) == dept_id


def test_all_surgeon_tools_have_name():
    for tool_id, info in SURGEON_TOOLS.items():
        assert "name" in info
        assert isinstance(info["name"], str)
        assert len(info["name"]) > 0


def test_surgeon_tools_covers_0_through_7():
    assert set(SURGEON_TOOLS.keys()) == set(range(8))


def test_healthcare_schema_has_required_columns():
    required = {
        "patient_id",
        "age",
        "birth_year",
        "email",
        "admission_date",
        "department_id",
        "department_name",
    }
    assert required.issubset(HEALTHCARE_SCHEMA.keys())


def test_financial_schema_has_required_columns():
    required = {"transaction_id", "amount", "currency", "transaction_date", "status"}
    assert required.issubset(FINANCIAL_SCHEMA.keys())


def test_dept_map_covers_schema_range():
    lo, hi = HEALTHCARE_SCHEMA["department_id"]["range"]
    for dept_id in range(lo, hi + 1):
        assert dept_id in DEPT_MAP


def test_generated_episodes_pass_solvability_gate(corr, clean_hc):
    for _ in range(20):
        dirty, gt, meta = corr.generate_episode(clean_hc)
        valid, reason = corr._solvability_gate(dirty, gt, meta)
        assert valid, reason


def test_corruptor_tier1_never_produces_delete_row(corr, clean_hc):
    for _ in range(20):
        corr.force_tier(1)
        _, _, meta = corr.generate_episode(clean_hc)
        assert meta.get("tool") != "delete_row"


def test_corruptor_records_episodes(corr):
    for reward in [0.5, -0.2, 1.0]:
        corr.record_episode(reward)
    assert len(corr._recent_rewards) == 3
    assert corr._epoch == 3


def test_force_tier_resets_window(corr):
    corr._recent_rewards.extend([0.9] * 15)
    corr.force_tier(1)
    assert len(corr._recent_rewards) == 0
    assert corr._unlocked_tier == 1


def test_corruptor_wont_unlock_tier_without_reward_gate(corr):
    corr._epoch = 35
    corr._recent_rewards.extend([-2.0] * 20)
    corr._update_tier()
    assert corr.current_tier() == 1


def test_results_json_advantage_is_positive():
    payload = json.loads(Path("eval/results.json").read_text(encoding="utf-8"))
    assert payload["surgeon_advantage_accuracy_delta"] > 0


def test_heuristic_results_json_win_rate_positive():
    payload = json.loads(Path("eval/heuristic_results.json").read_text(encoding="utf-8"))
    assert payload["surgeon_win_rate"] > 0


def test_training_log_parses_cleanly():
    df = pd.read_csv("logs/training_log.csv")
    assert "step" in df.columns
    assert "total_reward" in df.columns
    assert "parse_success_rate" in df.columns
    assert len(df) >= 5


def test_training_log_parse_success_valid():
    df = pd.read_csv("logs/training_log.csv")
    rates = pd.to_numeric(df["parse_success_rate"], errors="coerce").dropna()
    assert len(rates) > 0
    assert rates.between(0, 1).all()


def test_training_log_difficulty_column_present():
    df = pd.read_csv("logs/training_log.csv")
    difficulties = pd.to_numeric(df["difficulty"], errors="coerce").dropna()
    assert difficulties.between(1, 3).all()
