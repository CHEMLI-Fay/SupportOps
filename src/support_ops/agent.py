"""Bounded Groq tool-calling agent with human approval for demo actions."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from support_ops.config import load_api_key, load_settings, project_file
from support_ops.policies import load_policies, search_policies
from support_ops.storage import SupportStore
from support_ops.tools import OrderLookup


MAX_TOOL_CALLS = 4


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_calls_used: int
    last_tool_result: NotRequired[dict]


class PolicySearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=2, description="Customer's policy question or topic.")


def make_checkpointer(root: Path) -> SqliteSaver:
    """Keep paused graph runs in a local SQLite file."""
    path = root.resolve() / "artifacts" / "checkpoints.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection)


def build_support_agent(
    root: Path,
    customer_id: str,
    *,
    store: SupportStore | None = None,
    model=None,
    checkpointer=None,
):
    """Build one customer's agent; model arguments never include customer ID."""
    if not re.fullmatch(r"CUST-[0-9]+", customer_id):
        raise ValueError("Invalid demo customer ID.")
    settings = load_settings(root)
    store = store or SupportStore(root)
    store.initialize(project_file(root, settings.files.orders))
    policies = load_policies(project_file(root, "data/policies.json"))
    prompt = project_file(root, "prompts/agent.txt").read_text(encoding="utf-8").strip()
    if model is None:
        load_dotenv(root / ".env", override=False)
        model = ChatGroq(
            model=settings.llm.model,
            api_key=load_api_key(root),
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
            reasoning_effort=settings.llm.reasoning_effort,
            model_kwargs={"include_reasoning": False},
        )

    def get_order(order_id: str) -> dict:
        return store.get_order(customer_id, order_id)

    def search_policy(query: str) -> dict:
        return search_policies(policies, query)

    def request_cancel(order_id: str) -> dict:
        return store.preview_cancel(customer_id, order_id)

    tools = [
        StructuredTool.from_function(
            func=get_order,
            name="get_order",
            description="Read the current customer's order. Does not modify it.",
            args_schema=OrderLookup,
        ),
        StructuredTool.from_function(
            func=search_policy,
            name="search_policy",
            description="Retrieve a written store policy with a source ID.",
            args_schema=PolicySearchInput,
        ),
        StructuredTool.from_function(
            func=request_cancel,
            name="request_cancel",
            description="Preview cancellation of an order. The app asks for human approval before any update.",
            args_schema=OrderLookup,
        ),
    ]
    by_name = {tool.name: tool for tool in tools}
    bound_model = model.bind_tools(tools, tool_choice="auto")

    def call_model(state: AgentState) -> dict:
        if state.get("tool_calls_used", 0) >= MAX_TOOL_CALLS:
            return {
                "messages": [AIMessage(content="Je ne peux pas terminer cette demande de façon fiable. Veuillez contacter le support.")],
            }
        response = bound_model.invoke([SystemMessage(content=prompt), *state["messages"]])
        if not isinstance(response, AIMessage) or response.invalid_tool_calls:
            return {
                "messages": [AIMessage(content="Je n'ai pas pu vérifier cette demande. Veuillez réessayer.")],
            }
        remaining = MAX_TOOL_CALLS - state.get("tool_calls_used", 0)
        if len(response.tool_calls) > remaining:
            return {
                "messages": [AIMessage(content="Je ne peux pas terminer cette demande de façon fiable. Veuillez contacter le support.")],
            }
        return {"messages": [response]}

    def route_after_model(state: AgentState) -> str:
        latest = state["messages"][-1]
        return "execute_tools" if isinstance(latest, AIMessage) and latest.tool_calls else END

    def execute_tools(state: AgentState) -> dict:
        response = state["messages"][-1]
        outputs = []
        last_result = None
        for call in response.tool_calls:
            name = call["name"]
            call_id = call.get("id")
            if not call_id or name not in by_name:
                result = {"error": "unsupported_tool"}
            else:
                try:
                    result = by_name[name].invoke(call["args"])
                except (ValidationError, ValueError, TypeError):
                    result = {"error": "invalid_tool_arguments"}
                if name == "request_cancel" and result.get("eligible"):
                    approval = interrupt(
                        {
                            "kind": "cancel_order",
                            "order_id": result["order_id"],
                            "message": "Approve cancellation of this processing order?",
                        }
                    )
                    if approval is True:
                        result = store.cancel_order(
                            customer_id, result["order_id"], f"{customer_id}:{call_id}"
                        )
                    else:
                        result = {
                            "found": True,
                            "order_id": result["order_id"],
                            "outcome": "denied",
                        }
            outputs.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=call_id or "missing_call_id",
                    name=name,
                )
            )
            last_result = result
        return {
            "messages": outputs,
            "tool_calls_used": state.get("tool_calls_used", 0) + len(response.tool_calls),
            "last_tool_result": last_result,
        }

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("execute_tools", execute_tools)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_model, {"execute_tools": "execute_tools", END: END})
    builder.add_edge("execute_tools", "agent")
    return builder.compile(checkpointer=checkpointer)
