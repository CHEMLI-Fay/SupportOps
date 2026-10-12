"""Bounded Groq tool-calling agent with human approval for demo actions."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
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


def cancellation_answer(result: dict, order_id: str | None, question: str) -> str:
    """Render an action outcome from validated data, without another LLM call."""
    lower = question.casefold()
    english = bool(re.search(r"\b(cancel|cancellation|order)\b", lower)) and not (
        "annul" in lower or "commande" in lower
    )
    label = (
        order_id
        if isinstance(order_id, str) and re.fullmatch(r"ORD-[0-9]+", order_id)
        else ("this order" if english else "cette commande")
    )
    if result.get("error") == "invalid_tool_arguments":
        return (
            "Please provide a valid order ID before requesting cancellation."
            if english
            else "Veuillez indiquer un numéro de commande valide pour demander l'annulation."
        )
    if result.get("found") is False:
        return (
            f"Order {label} is unavailable. No cancellation was made."
            if english
            else f"La commande {label} est introuvable ou inaccessible. Aucune annulation n'a été effectuée."
        )
    if result.get("outcome") == "cancelled":
        return (
            f"Order {label} was cancelled after your approval."
            if english
            else f"La commande {label} a été annulée après votre approbation."
        )
    if result.get("outcome") == "denied":
        return (
            f"You declined cancellation of order {label}. No change was made."
            if english
            else f"Vous avez refusé l'annulation de la commande {label}. Aucun changement n'a été effectué."
        )
    status = result.get("status")
    if status == "cancelled":
        return (
            f"Order {label} is already cancelled. No change was made."
            if english
            else f"La commande {label} est déjà annulée. Aucun changement n'a été effectué."
        )
    if status == "shipped":
        return (
            f"Order {label} has already shipped and cannot be cancelled here. No change was made."
            if english
            else f"La commande {label} est déjà expédiée et ne peut pas être annulée ici. Aucun changement n'a été effectué."
        )
    if status == "delivered":
        return (
            f"Order {label} has already been delivered and cannot be cancelled here. No change was made."
            if english
            else f"La commande {label} est déjà livrée et ne peut pas être annulée ici. Aucun changement n'a été effectué."
        )
    return (
        "I could not confirm this cancellation. No change was made."
        if english
        else "Je n'ai pas pu confirmer cette annulation. Aucun changement n'a été effectué."
    )


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_calls_used: int
    last_tool_result: NotRequired[dict]


class PolicySearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=2, description="Customer's policy question or topic.")


class DatabaseSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=2, description="Search the current customer's account, orders, or public product catalog.")


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

    def search_database(query: str) -> dict:
        return store.search_database(customer_id, query)

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
            func=search_database,
            name="search_database",
            description="Search the current demo customer's profile and order history, or the public product catalog. Use this for broad account, history, customer, and product questions.",
            args_schema=DatabaseSearchInput,
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
        if len(response.tool_calls) > 1 and any(
            call["name"] == "request_cancel" for call in response.tool_calls
        ):
            return {
                "messages": [
                    AIMessage(content="Veuillez demander l'annulation d'une seule commande à la fois.")
                ]
            }
        return {"messages": [response]}

    def route_after_model(state: AgentState) -> str:
        latest = state["messages"][-1]
        return "execute_tools" if isinstance(latest, AIMessage) and latest.tool_calls else END

    def execute_tools(state: AgentState) -> dict:
        response = state["messages"][-1]
        outputs = []
        last_result = None
        cancel_result = None
        cancel_order_id = None
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
            if name == "request_cancel":
                cancel_result = result
                cancel_order_id = result.get("order_id") or call["args"].get("order_id")
            last_result = result
        if cancel_result is not None:
            question = next(
                (
                    str(message.content)
                    for message in reversed(state["messages"])
                    if isinstance(message, HumanMessage)
                ),
                "",
            )
            outputs.append(
                AIMessage(
                    content=cancellation_answer(
                        cancel_result, cancel_order_id, question
                    )
                )
            )
        return {
            "messages": outputs,
            "tool_calls_used": state.get("tool_calls_used", 0) + len(response.tool_calls),
            "last_tool_result": last_result,
        }

    def route_after_tools(state: AgentState) -> str:
        return END if isinstance(state["messages"][-1], AIMessage) else "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("execute_tools", execute_tools)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_model, {"execute_tools": "execute_tools", END: END})
    builder.add_conditional_edges(
        "execute_tools", route_after_tools, {"agent": "agent", END: END}
    )
    return builder.compile(checkpointer=checkpointer)
