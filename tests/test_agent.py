from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from support_ops.agent import build_support_agent, make_checkpointer
from support_ops.storage import SupportStore


ROOT = Path(__file__).resolve().parents[1]


class ScriptedModel:
    def __init__(self, tool_name: str, arguments: dict):
        self.tool_name = tool_name
        self.arguments = arguments

    def bind_tools(self, tools, tool_choice="auto"):
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="Tool result: " + messages[-1].content)
        return AIMessage(
            content="",
            tool_calls=[
                {"name": self.tool_name, "args": self.arguments, "id": "scripted-call"}
            ],
        )


def build(tmp_path, tool_name, arguments, checkpointer=None):
    store = SupportStore(tmp_path)
    store.initialize(ROOT / "data/orders.json")
    graph = build_support_agent(
        ROOT,
        "CUST-001",
        store=store,
        model=ScriptedModel(tool_name, arguments),
        checkpointer=checkpointer or InMemorySaver(),
    )
    return graph, store


def start(graph, question, thread_id=None):
    config = {"configurable": {"thread_id": thread_id or str(uuid4())}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)], "tool_calls_used": 0},
        config=config,
    )
    return result, config


def test_read_tools_are_customer_bound(tmp_path):
    graph, _ = build(tmp_path, "get_order", {"order_id": "ORD-2099"})
    result, _ = start(graph, "Status of ORD-2099?")
    assert result["last_tool_result"] == {"found": False, "error": "order_unavailable"}
    assert "customer_id" not in result["messages"][-1].content

    policy_graph, _ = build(tmp_path, "search_policy", {"query": "return policy"})
    result, _ = start(policy_graph, "What is the return policy?")
    assert result["last_tool_result"]["source_id"] == "POL-RETURNS"


def test_approval_denial_and_persistent_resume(tmp_path):
    checkpointer = make_checkpointer(tmp_path)
    graph, store = build(
        tmp_path, "request_cancel", {"order_id": "ORD-1050"}, checkpointer
    )
    pending, config = start(graph, "Cancel ORD-1050")
    assert "__interrupt__" in pending
    assert store.get_order("CUST-001", "ORD-1050")["status"] == "processing"

    resumed_graph, _ = build(
        tmp_path, "request_cancel", {"order_id": "ORD-1050"}, checkpointer
    )
    denied = resumed_graph.invoke(Command(resume=False), config=config)
    assert denied["last_tool_result"]["outcome"] == "denied"
    assert store.get_order("CUST-001", "ORD-1050")["status"] == "processing"

    pending, config = start(graph, "Cancel ORD-1050")
    assert "__interrupt__" in pending
    approved = resumed_graph.invoke(Command(resume=True), config=config)
    assert approved["last_tool_result"]["outcome"] == "cancelled"
    assert store.get_order("CUST-001", "ORD-1050")["status"] == "cancelled"
    replay = store.cancel_order("CUST-001", "ORD-1050", "CUST-001:scripted-call")
    assert replay["replayed"] is True


def test_ineligible_order_does_not_pause(tmp_path):
    graph, store = build(tmp_path, "request_cancel", {"order_id": "ORD-1042"})
    result, _ = start(graph, "Cancel ORD-1042")
    assert "__interrupt__" not in result
    assert result["last_tool_result"]["eligible"] is False
    assert store.get_order("CUST-001", "ORD-1042")["status"] == "shipped"
