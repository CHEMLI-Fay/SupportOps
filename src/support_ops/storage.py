"""Local SQLite storage for the fictional support operations database."""

import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from support_ops.records import load_customers, load_products
from support_ops.tools import Order, load_orders


class SupportStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / "artifacts" / "supportops.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _order_values(order: Order) -> dict:
        first = order.items[0]
        return {
            "order_id": order.order_id,
            "customer_id": order.customer_id,
            "product": first.product_name,
            "quantity": sum(item.quantity for item in order.items),
            "status": order.status,
            "estimated_delivery": order.estimated_delivery,
            "created_at": order.created_at,
            "updated_at": order.updated_at,
            "tracking_number": order.tracking_number,
            "payment_method": order.payment_method,
            "subtotal": order.subtotal,
            "shipping_fee": order.shipping_fee,
            "total": order.total,
            "currency": order.currency,
            "shipping_address": order.shipping_address,
        }

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                product TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                status TEXT NOT NULL,
                estimated_delivery TEXT,
                created_at TEXT,
                updated_at TEXT,
                tracking_number TEXT,
                payment_method TEXT,
                subtotal REAL,
                shipping_fee REAL,
                total REAL,
                currency TEXT,
                shipping_address TEXT
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
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                tier TEXT NOT NULL,
                joined_at TEXT NOT NULL,
                preferred_language TEXT NOT NULL,
                address TEXT NOT NULL,
                notes TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS products (
                sku TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                currency TEXT NOT NULL,
                stock INTEGER NOT NULL,
                rating REAL NOT NULL,
                reviews INTEGER NOT NULL,
                image TEXT NOT NULL,
                description TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS order_items (
                order_id TEXT NOT NULL,
                sku TEXT NOT NULL,
                product_name TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price REAL NOT NULL,
                PRIMARY KEY (order_id, sku),
                FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
                FOREIGN KEY (sku) REFERENCES products(sku)
            );
            """
        )
        existing = {
            row["name"] for row in connection.execute("PRAGMA table_info(orders)")
        }
        additions = {
            "created_at": "TEXT",
            "updated_at": "TEXT",
            "tracking_number": "TEXT",
            "payment_method": "TEXT",
            "subtotal": "REAL",
            "shipping_fee": "REAL",
            "total": "REAL",
            "currency": "TEXT",
            "shipping_address": "TEXT",
        }
        for name, sql_type in additions.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE orders ADD COLUMN {name} {sql_type}")

    def initialize(self, orders_path: Path) -> None:
        """Create the local schema and seed or enrich records without resetting status."""
        orders = load_orders(orders_path)
        data_dir = orders_path.resolve().parent
        customers = load_customers(data_dir / "customers.json")
        products = load_products(data_dir / "products.json")

        with closing(self._connect()) as connection, connection:
            self._create_schema(connection)
            connection.executemany(
                """
                INSERT INTO customers VALUES
                (:customer_id, :name, :email, :phone, :tier, :joined_at,
                 :preferred_language, :address, :notes)
                ON CONFLICT(customer_id) DO UPDATE SET
                    name=excluded.name, email=excluded.email, phone=excluded.phone,
                    tier=excluded.tier, joined_at=excluded.joined_at,
                    preferred_language=excluded.preferred_language,
                    address=excluded.address, notes=excluded.notes
                """,
                [customer.model_dump() for customer in customers.values()],
            )
            connection.executemany(
                """
                INSERT INTO products VALUES
                (:sku, :name, :category, :price, :currency, :stock,
                 :rating, :reviews, :image, :description)
                ON CONFLICT(sku) DO UPDATE SET
                    name=excluded.name, category=excluded.category,
                    price=excluded.price, currency=excluded.currency,
                    stock=excluded.stock, rating=excluded.rating,
                    reviews=excluded.reviews, image=excluded.image,
                    description=excluded.description
                """,
                [product.model_dump() for product in products.values()],
            )
            for order in orders.values():
                values = self._order_values(order)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO orders
                    (order_id, customer_id, product, quantity, status,
                     estimated_delivery, created_at, updated_at, tracking_number,
                     payment_method, subtotal, shipping_fee, total, currency,
                     shipping_address)
                    VALUES
                    (:order_id, :customer_id, :product, :quantity, :status,
                     :estimated_delivery, :created_at, :updated_at, :tracking_number,
                     :payment_method, :subtotal, :shipping_fee, :total, :currency,
                     :shipping_address)
                    """,
                    values,
                )
                # Refresh descriptive fixture fields but preserve the current status.
                connection.execute(
                    """
                    UPDATE orders SET customer_id=:customer_id, product=:product,
                        quantity=:quantity, estimated_delivery=:estimated_delivery,
                        created_at=:created_at, updated_at=:updated_at,
                        tracking_number=:tracking_number, payment_method=:payment_method,
                        subtotal=:subtotal, shipping_fee=:shipping_fee, total=:total,
                        currency=:currency, shipping_address=:shipping_address
                    WHERE order_id=:order_id
                    """,
                    values,
                )
                connection.executemany(
                    """
                    INSERT INTO order_items
                    (order_id, sku, product_name, quantity, unit_price)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(order_id, sku) DO UPDATE SET
                        product_name=excluded.product_name,
                        quantity=excluded.quantity,
                        unit_price=excluded.unit_price
                    """,
                    [
                        (
                            order.order_id,
                            item.sku,
                            item.product_name,
                            item.quantity,
                            item.unit_price,
                        )
                        for item in order.items
                    ],
                )

    def _items_for(self, connection: sqlite3.Connection, order_id: str) -> list[dict]:
        rows = connection.execute(
            """
            SELECT i.sku, i.product_name, i.quantity, i.unit_price,
                   p.category, p.image
            FROM order_items i JOIN products p ON p.sku=i.sku
            WHERE i.order_id=? ORDER BY i.sku
            """,
            (order_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_order(self, customer_id: str, order_id: str) -> dict:
        """Return one customer's order with its item details."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT order_id, product, quantity, status, estimated_delivery,
                       created_at, updated_at, tracking_number, payment_method,
                       subtotal, shipping_fee, total, currency, shipping_address
                FROM orders WHERE order_id=? AND customer_id=?
                """,
                (order_id, customer_id),
            ).fetchone()
            if not row:
                return {"found": False, "error": "order_unavailable"}
            result = {"found": True, **dict(row)}
            result["items"] = self._items_for(connection, order_id)
            return result

    def get_order_by_id(self, order_id: str) -> dict | None:
        """Return a record for the internal, read-only database detail view."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT customer_id FROM orders WHERE order_id=?", (order_id,)
            ).fetchone()
        return self.get_order(row["customer_id"], order_id) if row else None

    def list_orders(self, customer_id: str | None = None) -> list[dict]:
        query = """
            SELECT order_id, customer_id, product, quantity, status,
                   estimated_delivery, created_at, updated_at, tracking_number,
                   payment_method, subtotal, shipping_fee, total, currency
            FROM orders
        """
        params: tuple = ()
        if customer_id is not None:
            query += " WHERE customer_id=?"
            params = (customer_id,)
        query += " ORDER BY created_at DESC, order_id"
        with closing(self._connect()) as connection:
            return [dict(row) for row in connection.execute(query, params).fetchall()]

    def list_customers(self) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT c.customer_id, c.name, c.email, c.phone, c.tier,
                       c.joined_at, c.preferred_language, c.address, c.notes,
                       COUNT(o.order_id) AS order_count,
                       COALESCE(SUM(o.total), 0) AS lifetime_spend
                FROM customers c LEFT JOIN orders o ON o.customer_id=c.customer_id
                GROUP BY c.customer_id ORDER BY c.name
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def get_customer(self, customer_id: str) -> dict | None:
        return next(
            (
                customer
                for customer in self.list_customers()
                if customer["customer_id"] == customer_id
            ),
            None,
        )

    def list_products(self) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT sku, name, category, price, currency, stock,
                       rating, reviews, image, description
                FROM products ORDER BY category, name
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def list_order_items(self) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT i.order_id, o.customer_id, i.sku, i.product_name,
                       i.quantity, i.unit_price, p.category
                FROM order_items i JOIN orders o ON o.order_id=i.order_id
                JOIN products p ON p.sku=i.sku
                ORDER BY i.order_id, i.sku
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def search_database(self, customer_id: str, query: str) -> dict:
        """Search only the active demo customer's profile and orders, plus catalog items."""
        customer = self.get_customer(customer_id)
        if customer is None:
            return {"found": False, "error": "customer_unavailable"}
        all_orders = self.list_orders(customer_id)
        all_products = self.list_products()
        order_ids = re.findall(r"ORD-[0-9]+", query.upper())
        if order_ids:
            selected_orders = [
                order for order in all_orders if order["order_id"] in order_ids
            ]
        else:
            selected_orders = all_orders[:8]
        terms = [
            term.casefold()
            for term in re.findall(r"[\w-]{3,}", query)
            if term.casefold() not in {"order", "orders", "commande", "commandes", "what", "when", "your", "have", "with", "from", "this", "that", "pour", "dans", "avec", "quel", "quelle", "mes", "mon", "les", "des"}
        ]
        products = [
            product
            for product in all_products
            if not terms
            or any(
                term in (product["name"] + " " + product["category"] + " " + product["description"]).casefold()
                for term in terms
            )
        ]
        if not products and any(term in query.casefold() for term in ("catalog", "inventory", "stock", "catalogue", "produit", "products")):
            products = all_products
        return {
            "found": True,
            "customer": {
                key: customer[key]
                for key in (
                    "customer_id", "name", "email", "phone", "tier", "joined_at",
                    "preferred_language", "order_count", "lifetime_spend",
                )
            },
            "orders": selected_orders,
            "products": products[:6],
            "source_ids": [order["order_id"] for order in selected_orders],
        }

    def reset_demo(self, orders_path: Path) -> None:
        """Restore all synthetic database records and order statuses."""
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM actions")
            connection.execute("DELETE FROM order_items")
            connection.execute("DELETE FROM orders")
        self.initialize(orders_path)

    def preview_cancel(self, customer_id: str, order_id: str) -> dict:
        order = self.get_order(customer_id, order_id)
        if not order["found"]:
            return order
        return {
            "found": True,
            "order_id": order_id,
            "status": order["status"],
            "eligible": order["status"] == "processing",
            "reason": f"status_{order['status']}" if order["status"] != "processing" else None,
        }

    def cancel_order(self, customer_id: str, order_id: str, request_id: str) -> dict:
        """Commit an approved cancellation at most once for a request ID."""
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT customer_id, order_id, outcome FROM actions WHERE request_id=?",
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
                "SELECT status FROM orders WHERE order_id=? AND customer_id=?",
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
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """
                UPDATE orders SET status='cancelled', updated_at=?
                WHERE order_id=? AND customer_id=? AND status='processing'
                """,
                (now[:10], order_id, customer_id),
            )
            connection.execute(
                "INSERT INTO actions VALUES (?, ?, ?, ?, ?, ?)",
                (request_id, customer_id, order_id, "cancel_order", "cancelled", now),
            )
        return {
            "found": True,
            "order_id": order_id,
            "outcome": "cancelled",
            "request_id": request_id,
            "replayed": False,
        }
