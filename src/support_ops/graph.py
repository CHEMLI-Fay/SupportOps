"""A small, deterministic support-request workflow."""

import re
from typing import Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from support_ops.tools import Order, make_order_tool


Category = Literal["order_status", "missing_order_id", "multiple_order_ids", "unsupported"]


class SupportState(TypedDict):
    question: str
    category: NotRequired[Category]
    order_id: NotRequired[str]
    order: NotRequired[dict]
    answer: NotRequired[str]


ORDER_ID_PATTERN = re.compile(r"\bORD-[0-9]+\b", re.IGNORECASE)
STATUS_PATTERN = re.compile(
    r"\b(statut|status|suivi|tracking)\b|où en est|where is", re.IGNORECASE
)
OTHER_INTENT_PATTERN = re.compile(
    r"\b(rembours\w*|refund\w*|annul\w*|cancel\w*|retour\w*|return\w*|modifi\w*|chang\w*)\b",
    re.IGNORECASE,
)

STATUS_IN_FRENCH = {
    "processing": "en préparation",
    "shipped": "expédiée",
    "delivered": "livrée",
    "cancelled": "annulée",
}


def classify_request(question: str) -> dict:
    """Recognize a narrow order-status intent and extract one order ID."""
    if not STATUS_PATTERN.search(question) or OTHER_INTENT_PATTERN.search(question):
        return {"category": "unsupported"}
    order_ids = list(dict.fromkeys(match.upper() for match in ORDER_ID_PATTERN.findall(question)))
    if not order_ids:
        return {"category": "missing_order_id"}
    if len(order_ids) > 1:
        return {"category": "multiple_order_ids"}
    return {"category": "order_status", "order_id": order_ids[0]}


def build_order_graph(orders: dict[str, Order], customer_id: str):
    """Build a read-only graph for one customer context."""
    order_tool = make_order_tool(orders, customer_id)

    def classify(state: SupportState) -> dict:
        return classify_request(state["question"])

    def route_after_classification(state: SupportState) -> Category:
        return state["category"]

    def lookup_order(state: SupportState) -> dict:
        return {"order": order_tool.invoke({"order_id": state["order_id"]})}

    def route_after_lookup(state: SupportState) -> str:
        return "answer" if state["order"]["found"] else "unavailable"

    def answer_order(state: SupportState) -> dict:
        status = STATUS_IN_FRENCH[state["order"]["status"]]
        return {"answer": f"Votre commande {state['order_id']} est actuellement {status}."}

    def answer_unavailable(state: SupportState) -> dict:
        return {"answer": "Je ne peux pas accéder à cette commande."}

    def ask_order_id(state: SupportState) -> dict:
        return {"answer": "Quel est votre numéro de commande (par exemple ORD-1042) ?"}

    def ask_one_order_id(state: SupportState) -> dict:
        return {"answer": "Indiquez un seul numéro de commande à la fois."}

    def answer_unsupported(state: SupportState) -> dict:
        return {"answer": "Je peux actuellement vérifier le statut d'une commande. Quelle est votre question sur son statut ?"}

    builder = StateGraph(SupportState)
    builder.add_node("classify_request", classify)
    builder.add_node("lookup_order", lookup_order)
    builder.add_node("answer", answer_order)
    builder.add_node("unavailable", answer_unavailable)
    builder.add_node("ask_order_id", ask_order_id)
    builder.add_node("ask_one_order_id", ask_one_order_id)
    builder.add_node("unsupported", answer_unsupported)
    builder.add_edge(START, "classify_request")
    builder.add_conditional_edges(
        "classify_request",
        route_after_classification,
        {
            "order_status": "lookup_order",
            "missing_order_id": "ask_order_id",
            "multiple_order_ids": "ask_one_order_id",
            "unsupported": "unsupported",
        },
    )
    builder.add_conditional_edges(
        "lookup_order",
        route_after_lookup,
        {"answer": "answer", "unavailable": "unavailable"},
    )
    for final_node in ("answer", "unavailable", "ask_order_id", "ask_one_order_id", "unsupported"):
        builder.add_edge(final_node, END)
    return builder.compile()
