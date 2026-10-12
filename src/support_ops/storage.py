"""Local SQLite storage for synthetic orders and idempotent demo actions."""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from support_ops.tools import load_orders


class SupportStore:
    def __init__(self, root: Path):
        self.path = root.resolve() / "artifacts" / "supportops.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self, orders_path: Path) -> None:
        """Create tables and seed missing orders without overwriting prior actions."""
        orders = load_orders(orders_path)
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    product TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    estimated_delivery TEXT
                );
                CREATE TABLE IF NOT EXISTS actions (
                    request_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (order_id) REFERENCES orders(order_id)
                );
                """
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO orders
                (order_id, customer_id, product, quantity, status, estimated_delivery)
                VALUES (:order_id, :customer_id, :product, :quantity, :status, :estimated_delivery)
                """,
                [order.model_dump() for order in orders.values()],
            )

    def get_order(self, customer_id: str, order_id: str) -> dict:
        """Return only an owned order; foreign and missing IDs look identical."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT order_id, product, quantity, status, estimated_delivery "
                "FROM orders WHERE order_id = ? AND customer_id = ?",
                (order_id, customer_id),
            ).fetchone()
        return {"found": True, **dict(row)} if row else {"found": False, "error": "order_unavailable"}

    def list_orders(self, customer_id: str) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT order_id, product, quantity, status, estimated_delivery "
                "FROM orders WHERE customer_id = ? ORDER BY order_id",
                (customer_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def reset_demo(self, orders_path: Path) -> None:
        """Restore synthetic seed orders for a new local demonstration."""
        orders = load_orders(orders_path)
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM actions")
            connection.execute("DELETE FROM orders")
            connection.executemany(
                """
                INSERT INTO orders
                (order_id, customer_id, product, quantity, status, estimated_delivery)
                VALUES (:order_id, :customer_id, :product, :quantity, :status, :estimated_delivery)
                """,
                [order.model_dump() for order in orders.values()],
            )

    def preview_cancel(self, customer_id: str, order_id: str) -> dict:
        order = self.get_order(customer_id, order_id)
        if not order["found"]:
            return order
        return {
            "found": True,
            "order_id": order_id,
            "status": order["status"],
            "eligible": order["status"] == "processing",
            "reason": (
                f"status_{order['status']}" if order["status"] != "processing" else None
            ),
        }

    def cancel_order(self, customer_id: str, order_id: str, request_id: str) -> dict:
        """Commit an approved cancellation at most once for a request ID."""
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT customer_id, order_id, outcome FROM actions WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if previous:
                if previous["customer_id"] != customer_id or previous["order_id"] != order_id:
                    raise ValueError("Action request ID was reused for another order.")
                return {
                    "order_id": order_id,
                    "outcome": previous["outcome"],
                    "request_id": request_id,
                    "replayed": True,
                }
            row = connection.execute(
                "SELECT status FROM orders WHERE order_id = ? AND customer_id = ?",
                (order_id, customer_id),
            ).fetchone()
            if row is None:
                return {"found": False, "error": "order_unavailable"}
            if row["status"] != "processing":
                return {
                    "found": True,
                    "order_id": order_id,
                    "outcome": "not_eligible",
                    "status": row["status"],
                    "reason": f"status_{row['status']}",
                }
            connection.execute(
                "UPDATE orders SET status = 'cancelled' "
                "WHERE order_id = ? AND customer_id = ? AND status = 'processing'",
                (order_id, customer_id),
            )
            connection.execute(
                "INSERT INTO actions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    customer_id,
                    order_id,
                    "cancel_order",
                    "cancelled",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return {
            "found": True,
            "order_id": order_id,
            "outcome": "cancelled",
            "request_id": request_id,
            "replayed": False,
        }
