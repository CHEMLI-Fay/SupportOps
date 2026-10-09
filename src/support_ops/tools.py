"""Read-only tools backed by synthetic orders."""

import json
from pathlib import Path
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    order_id: str = Field(pattern=r"^ORD-[0-9]+$")
    customer_id: str = Field(pattern=r"^CUST-[0-9]+$")
    product: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    status: Literal["processing", "shipped", "delivered", "cancelled"]
    estimated_delivery: str | None


class OrderLookup(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    order_id: str = Field(pattern=r"^ORD-[0-9]+$", description="Order identifier, e.g. ORD-1042.")


def load_orders(path: Path) -> dict[str, Order]:
    try:
        records = TypeAdapter(list[Order]).validate_python(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, ValidationError):
        raise ValueError("Invalid order fixtures: check data/orders.json.") from None
    orders = {order.order_id: order for order in records}
    if len(orders) != len(records):
        raise ValueError("Order fixtures contain duplicate order identifiers.")
    return orders


def make_order_tool(orders: dict[str, Order], customer_id: str) -> StructuredTool:
    """Bind a demo customer outside the model-controlled tool arguments.

    This demonstrates an authorization boundary, not a real login system.
    """
    def get_order(order_id: str) -> dict:
        order = orders.get(order_id)
        if order is None or order.customer_id != customer_id:
            return {"found": False, "error": "order_unavailable"}
        return {"found": True, **order.model_dump(exclude={"customer_id"})}

    return StructuredTool.from_function(
        func=get_order,
        name="get_order",
        description="Read the current customer's order status. No order is modified.",
        args_schema=OrderLookup,
    )
