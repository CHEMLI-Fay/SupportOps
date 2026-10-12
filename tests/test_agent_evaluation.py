import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from support_ops.agent_evaluation import (
    check_behavior, execute_case, load_agent_scenarios, report_passed, run_suite,
    scenario_version,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = load_agent_scenarios(ROOT)


def test_dataset_version_changes_when_a_paraphrase_changes():
    edited = list(CASES)
    edited[-1] = edited[-1].model_copy(update={"question": "A revised access request."})
    assert scenario_version(edited) != scenario_version(CASES)


@pytest.fixture(autouse=True)
def no_remote_traces(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


class ScriptedModel:
    def __init__(self, case):
        self.case = case

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage) or self.case.required_tool is None:
            return AIMessage(content=self.case.reference)
        tool = self.case.required_tool
        if tool in ("get_order", "request_cancel"):
            order = self.case.order_id or self.case.expected_result.get("order_id")
            order = order or "ORD-2099"
            args = {"order_id": order}
        else:
            args = {"query": self.case.question}
        return AIMessage(content="", tool_calls=[
            {"name": tool, "args": args, "id": "test-call"}
        ])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_scenario_contracts_with_scripted_model(case):
    result = execute_case(ROOT, case, model=ScriptedModel(case))
    assert result["status"] == "passed", result
    assert result["usage"] is None  # No invented token measurements.


def test_metric_rejects_change_before_approval_and_wrong_result():
    case = next(c for c in CASES if c.id == "approved_cancellation")
    state = {"messages": [
        AIMessage(content="", tool_calls=[
            {"name": "request_cancel", "args": {"order_id": "ORD-1050"}, "id": "x"}
        ]),
        ToolMessage(content=json.dumps({"outcome": "denied"}), tool_call_id="x"),
        AIMessage(content="Cancelled."),
    ]}
    checks = check_behavior(
        case, state, paused=True, unchanged_before_approval=False,
        before={"ORD-1050": "processing"}, after={"ORD-1050": "processing"},
    )
    assert not checks["approval"]
    assert not checks["tool_result"]
    assert not checks["state_integrity"]


def test_provider_error_is_not_a_pass_and_does_not_leak_details():
    class BrokenModel(ScriptedModel):
        def invoke(self, messages):
            raise RuntimeError("secret-provider-detail")
    result = execute_case(ROOT, CASES[0], model=BrokenModel(CASES[0]))
    assert result["status"] == "error"
    assert result["error_type"] == "RuntimeError"
    assert "secret-provider-detail" not in json.dumps(result)


def test_report_requires_all_cases_and_successful_judge(tmp_path, monkeypatch):
    import support_ops.agent_evaluation as evaluation
    monkeypatch.setattr(evaluation, "load_api_key", lambda root: "test-only")
    monkeypatch.setattr(evaluation, "ChatGroq", lambda **kwargs: None)
    monkeypatch.setattr(evaluation, "execute_case", lambda *args: {
        "id": "order_status", "status": "passed", "answer": "Shipped", "checks": {}
    })
    monkeypatch.setattr(evaluation, "score_answer", lambda *args: {
        "status": "error", "error_type": "RateLimitError"
    })
    report, path = run_suite(
        ROOT, live=True, judge_enabled=True, case_ids=["order_status"],
        delay_seconds=0, output=tmp_path / "report.json",
    )
    assert path.exists()
    assert report["cases"][0]["status"] == "error"
    assert not report_passed(report)
    report["cases"] = []
    assert not report_passed(report)
