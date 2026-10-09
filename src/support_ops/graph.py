"""A small, deterministic order-status workflow."""

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from support_ops.tools import Order, make_order_tool


class SupportState(TypedDict):
    order_id: str
    order: NotRequired[dict]
    answer: NotRequired[str]


STATUS_IN_FRENCH = {
    "processing": "en préparation",
    "shipped": "expédiée",
    "delivered": "livrée",
    "cancelled": "annulée",
}


def build_order_graph(orders: dict[str, Order], customer_id: str):
    """Build a read-only graph for one customer context."""
    order_tool = make_order_tool(orders, customer_id)

    def lookup_order(state: SupportState) -> dict:
        return {"order": order_tool.invoke({"order_id": state["order_id"]})}

    def route_after_lookup(state: SupportState) -> str:
        return "answer" if state["order"]["found"] else "unavailable"

    def answer_order(state: SupportState) -> dict:
        status = STATUS_IN_FRENCH[state["order"]["status"]]
        return {"answer": f"Votre commande {state['order_id']} est actuellement {status}."}

    def answer_unavailable(state: SupportState) -> dict:
        return {"answer": "Je ne peux pas accéder à cette commande."}

    builder = StateGraph(SupportState)
    builder.add_node("lookup_order", lookup_order)
    builder.add_node("answer", answer_order)
    builder.add_node("unavailable", answer_unavailable)
    builder.add_edge(START, "lookup_order")
    builder.add_conditional_edges(
        "lookup_order",
        route_after_lookup,
        {"answer": "answer", "unavailable": "unavailable"},
    )
    builder.add_edge("answer", END)
    builder.add_edge("unavailable", END)
    return builder.compile()
