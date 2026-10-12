"""Northstar Support: a polished local support operations workspace."""

import csv
import html
import io
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from support_ops.agent import build_support_agent, make_checkpointer
from support_ops.config import find_project_root, load_settings, project_file
from support_ops.policies import load_policies
from support_ops.observability import run_config
from support_ops.storage import SupportStore


ROOT = find_project_root(Path(__file__).parent)
load_dotenv(ROOT / ".env", override=False)
SETTINGS = load_settings(ROOT)
NAV_ITEMS = ["Overview", "Inbox", "Orders", "Customers", "Catalog", "Knowledge", "Database"]
PAGE_DESCRIPTIONS = {
    "Overview": "A clear view of today's support operations.",
    "Inbox": "Resolve customer questions with verified account context.",
    "Orders": "Review order status, products, payments, and shipping.",
    "Customers": "Browse fictional customer profiles and history.",
    "Catalog": "Review products, pricing, stock, and product imagery.",
    "Knowledge": "Browse the policy library used by the assistant.",
    "Database": "Open the local demo database and inspect its records.",
}
STATUS_LABELS = {
    "processing": "Processing",
    "shipped": "Shipped",
    "delivered": "Delivered",
    "cancelled": "Cancelled",
}


st.set_page_config(
    page_title="Northstar Support",
    page_icon=str(ROOT / "assets" / "northstar-mark.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = project_file(ROOT, "assets/app.css").read_text(encoding="utf-8")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def resources():
    store = SupportStore(ROOT)
    store.initialize(project_file(ROOT, SETTINGS.files.orders))
    checkpointer = make_checkpointer(ROOT)
    customers = store.list_customers()
    return store, checkpointer, customers


@st.cache_resource(show_spinner=False)
def agent_for(customer_id: str):
    store, checkpointer, _ = resources()
    return build_support_agent(
        ROOT, customer_id, store=store, checkpointer=checkpointer
    )


def safe_text(value) -> str:
    return html.escape(str(value if value is not None else "—"))


def money(amount, currency="EUR") -> str:
    try:
        amount = float(amount or 0)
    except (TypeError, ValueError):
        amount = 0.0
    return f"{amount:,.2f} {currency}"


def status_html(status: str) -> str:
    label = STATUS_LABELS.get(status, status.replace("_", " ").title())
    return (
        f'<span class="status-pill status-{safe_text(status)}">'
        f"{safe_text(label)}</span>"
    )


def metric_card(label: str, value: str, foot: str = ""):
    st.markdown(
        f'<div class="info-card"><div class="info-card-title">{safe_text(label)}</div>'
        f'<div class="info-card-value">{safe_text(value)}</div>'
        f'<div class="info-card-foot">{safe_text(foot)}</div></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, kicker: str = "NORTHSTAR SUPPORT"):
    st.markdown(f'<div class="page-kicker">{safe_text(kicker)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-subtitle">{safe_text(subtitle)}</div>', unsafe_allow_html=True)


def money_total(rows, key="total"):
    return sum(float(row.get(key) or 0) for row in rows)


def reset_conversation():
    st.session_state.thread_id = str(uuid4())
    st.session_state.chat_messages = []
    st.session_state.pending_approval = None


def navigate(page):
    st.session_state.navigation = page


def source_summary(result: dict | None) -> list[str]:
    if not isinstance(result, dict):
        return []
    labels = []
    if result.get("order_id"):
        labels.append(f"Order {result['order_id']}")
    if result.get("source_id"):
        labels.append(str(result["source_id"]))
    labels.extend(str(value) for value in result.get("source_ids", []))
    if result.get("customer", {}).get("customer_id"):
        labels.append(f"Customer {result['customer']['customer_id']}")
    if result.get("products"):
        labels.extend(f"Product {p['sku']}" for p in result["products"] if p.get("sku"))
    return list(dict.fromkeys(labels))


def run_agent(prompt_or_command):
    agent = agent_for(st.session_state.active_customer)
    config = run_config(ROOT, st.session_state.thread_id)
    return agent.invoke(prompt_or_command, config=config)


def record_agent_result(result):
    if "__interrupt__" in result:
        st.session_state.pending_approval = result["__interrupt__"][0].value
        return
    st.session_state.pending_approval = None
    reply = next(
        (
            message
            for message in reversed(result.get("messages", []))
            if isinstance(message, AIMessage)
            and not message.tool_calls
            and message.content
        ),
        None,
    )
    content = str(reply.content) if reply else "I could not complete that request."
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": content,
            "sources": source_summary(result.get("last_tool_result")),
        }
    )


def submit_question(question: str):
    clean = question.strip()
    if not clean:
        return
    st.session_state.chat_messages.append({"role": "user", "content": clean, "sources": []})
    try:
        result = run_agent(
            {"messages": [HumanMessage(content=clean)], "tool_calls_used": 0}
        )
        record_agent_result(result)
    except Exception:
        st.session_state.pending_approval = None
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": "I couldn't complete that request. Please check the connection and try again.",
                "sources": [],
            }
        )


def render_sources(sources: list[str]):
    if not sources:
        return
    st.markdown(
        " ".join(
            f'<span class="source-chip">{safe_text(source)}</span>'
            for source in sources
        ),
        unsafe_allow_html=True,
    )


def render_sidebar(store, customers):
    st.sidebar.markdown(
        """
        <div class="brand-block">
          <img class="brand-mark" src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5NiA5NiIgZmlsbD0ibm9uZSI+PHJlY3Qgd2lkdGg9Ijk2IiBoZWlnaHQ9Ijk2IiByeD0iMjYiIGZpbGw9IiMxMDIzM2QiLz48cGF0aCBkPSJNNDggMTcgNTYgMzkgNzkgNDggNTYgNTYgNDggNzkgNDAgNTYgMTcgNDggNDAgMzkgNDggMTdaIiBmaWxsPSIjOWZlM2Q0Ii8+PGNpcmNsZSBjeD0iNDgiIGN5PSI0OCIgcj0iNiIgZmlsbD0iIzEwMjMzZCIvPjwvc3ZnPg=="/>
          <div><div class="brand-name">Northstar Support</div>
          <div class="brand-caption">CUSTOMER OPERATIONS</div></div>
        </div>
        <div class="sidebar-section">Workspace</div>
        """,
        unsafe_allow_html=True,
    )
    if "navigation" not in st.session_state:
        st.session_state.navigation = "Overview"
    if "active_customer" not in st.session_state:
        st.session_state.active_customer = SETTINGS.demo.customer_id
    if "thread_id" not in st.session_state:
        reset_conversation()
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "pending_approval" not in st.session_state:
        st.session_state.pending_approval = None

    st.sidebar.radio(
        "Workspace",
        NAV_ITEMS,
        key="navigation",
        label_visibility="collapsed",
    )
    st.sidebar.markdown('<div class="sidebar-section">Quick access</div>', unsafe_allow_html=True)
    st.sidebar.button(
        "Open demo database",
        key="open_database",
        width="stretch",
        on_click=navigate,
        args=("Database",),
    )

    customer_names = {c["customer_id"]: c["name"] for c in customers}
    customer_ids = list(customer_names)
    if st.session_state.active_customer not in customer_ids:
        st.session_state.active_customer = customer_ids[0]
    st.sidebar.markdown('<div class="sidebar-section">Demo account</div>', unsafe_allow_html=True)
    st.sidebar.selectbox(
        "Current customer",
        customer_ids,
        key="active_customer",
        format_func=lambda customer_id: (
            f"{customer_names.get(customer_id, customer_id)} · {customer_id}"
        ),
        on_change=reset_conversation,
        help="Fictional customer context for this local application.",
    )
    st.sidebar.caption("All customer and order data in this workspace is fictional.")
    with st.sidebar.expander("Demo controls"):
        st.caption("Restore the original sample order statuses and receipts.")
        if st.button("Restore sample data", key="reset_data", width="stretch"):
            store.reset_demo(project_file(ROOT, SETTINGS.files.orders))
            st.cache_resource.clear()
            reset_conversation()
            st.rerun()


def render_overview(store, customer_id):
    orders = store.list_orders()
    customers = store.list_customers()
    products = store.list_products()
    processing = sum(order["status"] == "processing" for order in orders)
    shipped = sum(order["status"] == "shipped" for order in orders)
    shipped_value = money_total([order for order in orders if order["status"] == "shipped"])

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="hero-panel">
              <div class="hero-label">SUPPORT OPERATIONS · LIVE WORKSPACE</div>
              <h2>Every customer detail, in one place.</h2>
              <p>Explore orders, resolve questions with verified records, and keep every action under control.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.image(
            project_file(ROOT, "assets/overview-collection.png"),
            width="stretch",
        )

    st.write("")
    cols = st.columns(4)
    with cols[0]:
        metric_card("Total orders", str(len(orders)), "Across the sample database")
    with cols[1]:
        metric_card("Awaiting fulfilment", str(processing), "Orders in processing")
    with cols[2]:
        metric_card("In transit", str(shipped), f"{money(shipped_value)} order value")
    with cols[3]:
        metric_card("Customers", str(len(customers)), f"{len(products)} products in catalog")

    st.write("")
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown('<div class="section-head"><h2>Recent orders</h2><span class="section-meta">Latest activity</span></div>', unsafe_allow_html=True)
        recent = store.list_orders()[:6]
        st.dataframe(
            [
                {
                    "Order": order["order_id"],
                    "Customer": order["customer_id"],
                    "Product": order["product"],
                    "Placed": order["created_at"],
                    "Status": STATUS_LABELS.get(order["status"], order["status"]),
                    "Total": money(order["total"], order["currency"]),
                }
                for order in recent
            ],
            hide_index=True,
            width="stretch",
        )
    with right:
        st.markdown('<div class="section-head"><h2>Workspace shortcuts</h2></div>', unsafe_allow_html=True)
        st.button("Open support inbox", width="stretch", on_click=navigate, args=("Inbox",))
        st.button("Browse all orders", width="stretch", on_click=navigate, args=("Orders",))
        st.button("Open customer database", width="stretch", on_click=navigate, args=("Database",))
        selected_customer = store.get_customer(customer_id)
        if selected_customer:
            st.markdown(
                f'<div class="profile-card"><div class="info-card-title">ACTIVE DEMO CUSTOMER</div>'
                f'<div class="info-card-value">{safe_text(selected_customer["name"])}</div>'
                f'<div class="info-card-foot">{safe_text(selected_customer["tier"])} member · '
                f'{selected_customer["order_count"]} orders · Joined {safe_text(selected_customer["joined_at"])}</div></div>',
                unsafe_allow_html=True,
            )


def render_inbox(store, customer_id):
    customer = store.get_customer(customer_id)
    orders = store.list_orders(customer_id)
    left, main, right = st.columns([0.85, 1.75, 0.95], gap="large")
    with left:
        st.markdown('<div class="section-head"><h2>Conversations</h2></div>', unsafe_allow_html=True)
        if st.button("New conversation", width="stretch"):
            reset_conversation()
            st.rerun()
        st.markdown('<div class="order-row"><strong>Current customer</strong><br><span class="section-meta">AI assisted · verified data</span></div>', unsafe_allow_html=True)
        st.write("")
        for order in orders[:4]:
            st.markdown(
                f'<div class="order-row"><strong>{safe_text(order["product"])}</strong><br>'
                f'<span class="section-meta">{safe_text(order["order_id"])} · {safe_text(STATUS_LABELS.get(order["status"], order["status"]))}</span></div>',
                unsafe_allow_html=True,
            )
            st.write("")
    with main:
        st.markdown('<div class="section-head"><h2>Support assistant</h2><span class="section-meta">Northstar AI · verified records</span></div>', unsafe_allow_html=True)
        if not st.session_state.chat_messages:
            st.markdown(
                """
                <div class="profile-card">
                  <div class="info-card-title">HOW CAN I HELP?</div>
                  <div class="info-card-value">Ask a question about this account.</div>
                  <div class="info-card-foot">Order status, order history, product stock, or store policy.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        for message in st.session_state.chat_messages:
            avatar = str(ROOT / "assets" / "northstar-mark.svg") if message["role"] == "assistant" else "user"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                render_sources(message.get("sources", []))

        pending = st.session_state.pending_approval
        if pending:
            order = store.get_order(customer_id, pending["order_id"])
            st.markdown(
                f'<div class="approval-card"><strong>Approval required</strong><br>'
                f'<span class="section-meta">Cancel order {safe_text(pending["order_id"])} · '
                f'{safe_text(order.get("product", "Order"))} · {money(order.get("total"), order.get("currency", "EUR"))}</span></div>',
                unsafe_allow_html=True,
            )
            approve_col, deny_col = st.columns(2)
            with approve_col:
                approve = st.button("Approve cancellation", type="primary", width="stretch")
            with deny_col:
                deny = st.button("Decline", width="stretch")
            if approve or deny:
                try:
                    result = run_agent(Command(resume=bool(approve)))
                    record_agent_result(result)
                except Exception:
                    st.session_state.pending_approval = None
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": "The request could not be completed. Please try again.", "sources": []}
                    )
                st.rerun()

        quick_left, quick_right = st.columns(2)
        with quick_left:
            if st.button("Check latest order", width="stretch"):
                if orders:
                    submit_question(f"What is the status of my latest order {orders[0]['order_id']}?")
                    st.rerun()
        with quick_right:
            if st.button("View return policy", width="stretch"):
                submit_question("What is the return policy?")
                st.rerun()

        prompt = st.chat_input(
            "Ask about this customer, orders, products, or policy",
            disabled=bool(st.session_state.pending_approval),
            key="support_chat_input",
        )
        if prompt:
            submit_question(prompt)
            st.rerun()

    with right:
        st.markdown('<div class="section-head"><h2>Customer profile</h2></div>', unsafe_allow_html=True)
        if customer:
            initials = "".join(part[0] for part in customer["name"].split()[:2]).upper()
            st.markdown(
                f'<div class="profile-card"><div class="profile-avatar">{safe_text(initials)}</div>'
                f'<div class="info-card-value">{safe_text(customer["name"])}</div>'
                f'<div class="info-card-foot">{safe_text(customer["tier"])} member</div><hr>'
                f'<div class="info-card-title">EMAIL</div><div>{safe_text(customer["email"])}</div>'
                f'<div class="info-card-title" style="margin-top:.7rem">PHONE</div><div>{safe_text(customer["phone"])}</div>'
                f'<div class="info-card-title" style="margin-top:.7rem">MEMBER SINCE</div><div>{safe_text(customer["joined_at"])}</div></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            metric_card("Lifetime spend", money(customer["lifetime_spend"]), f'{customer["order_count"]} orders')
        else:
            st.info("Select a demo customer to view their profile.")


def render_order_detail(store, order_id):
    order = store.get_order_by_id(order_id)
    if not order:
        st.info("Choose an order to view its details.")
        return
    status_col, total_col, placed_col, delivery_col = st.columns(4)
    with status_col:
        st.markdown(f'<div class="info-card-title">STATUS</div>{status_html(order["status"])}', unsafe_allow_html=True)
    with total_col:
        metric_card("Order total", money(order["total"], order["currency"]), "Including shipping")
    with placed_col:
        metric_card("Order placed", order["created_at"], "Creation date")
    with delivery_col:
        metric_card("Estimated delivery", order["estimated_delivery"] or "Not available", "Latest estimate")

    st.write("")
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.markdown("### Items")
        for item in order["items"]:
            image_path = project_file(ROOT, item["image"]) if item.get("image") else None
            item_col, details_col = st.columns([0.28, 0.72])
            with item_col:
                if image_path:
                    st.image(image_path, width=130)
            with details_col:
                st.markdown(f"**{safe_text(item['product_name'])}**")
                st.caption(f'{safe_text(item["sku"])} · Quantity {item["quantity"]} · {money(item["unit_price"], order["currency"])} each')
    with right:
        st.markdown("### Fulfilment")
        st.markdown(
            f'<div class="profile-card"><div class="info-card-title">TRACKING NUMBER</div>'
            f'<div class="info-card-value">{safe_text(order["tracking_number"] or "Not shipped yet")}</div>'
            f'<div class="info-card-title" style="margin-top:1rem">SHIP TO</div>'
            f'<div>{safe_text(order["shipping_address"])}</div>'
            f'<div class="info-card-title" style="margin-top:1rem">PAYMENT</div>'
            f'<div>{safe_text(order["payment_method"])}</div>'
            f'<div class="info-card-title" style="margin-top:1rem">LAST UPDATED</div>'
            f'<div>{safe_text(order["updated_at"])}</div></div>',
            unsafe_allow_html=True,
        )


def render_orders(store, customer_id):
    page_header("Orders", "Review every order record, fulfilment status, and payment detail.")
    all_orders = store.list_orders()
    search_col, status_col, customer_col = st.columns([1.5, 1, 1])
    with search_col:
        search = st.text_input("Search orders", placeholder="Order ID, product, or tracking number")
    with status_col:
        status_filter = st.selectbox("Status", ["All statuses", *STATUS_LABELS.values()])
    with customer_col:
        customer_filter = st.selectbox(
            "Customer",
            ["All customers", *[c["customer_id"] for c in store.list_customers()]],
        )
    filtered = all_orders
    if status_filter != "All statuses":
        filtered = [o for o in filtered if STATUS_LABELS.get(o["status"]) == status_filter]
    if customer_filter != "All customers":
        filtered = [o for o in filtered if o["customer_id"] == customer_filter]
    if search:
        needle = search.casefold()
        filtered = [
            o for o in filtered
            if needle in " ".join(str(o.get(k, "")) for k in ("order_id", "product", "tracking_number")).casefold()
        ]
    st.dataframe(
        [
            {
                "Order": order["order_id"],
                "Customer": order["customer_id"],
                "Product": order["product"],
                "Order date": order["created_at"],
                "Delivery": order["estimated_delivery"] or "—",
                "Status": STATUS_LABELS.get(order["status"], order["status"]),
                "Total": money(order["total"], order["currency"]),
            }
            for order in filtered
        ],
        hide_index=True,
        width="stretch",
    )
    order_ids = [order["order_id"] for order in filtered]
    if order_ids:
        selected = st.selectbox("Open order record", order_ids)
        render_order_detail(store, selected)
    else:
        st.info("No orders match these filters.")


def render_customers(store):
    page_header("Customers", "Fictional customer profiles, membership, and purchase history.")
    customers = store.list_customers()
    search_col, tier_col = st.columns([2, 1])
    with search_col:
        search = st.text_input("Search customers", placeholder="Name, email, or customer ID")
    tiers = ["All tiers", "Gold", "Silver", "Standard"]
    with tier_col:
        tier = st.selectbox("Membership", tiers)
    filtered = [
        customer for customer in customers
        if (tier == "All tiers" or customer["tier"] == tier)
        and (not search or search.casefold() in " ".join(str(customer.get(key, "")) for key in ("name", "email", "customer_id")).casefold())
    ]
    st.dataframe(
        [
            {
                "Customer": customer["name"],
                "ID": customer["customer_id"],
                "Membership": customer["tier"],
                "Orders": customer["order_count"],
                "Lifetime spend": money(customer["lifetime_spend"]),
                "Joined": customer["joined_at"],
            }
            for customer in filtered
        ],
        hide_index=True,
        width="stretch",
    )
    customer_ids = [customer["customer_id"] for customer in filtered]
    if not customer_ids:
        st.info("No customer matches these filters.")
        return
    selected = st.selectbox(
        "Open customer profile",
        customer_ids,
        format_func=lambda value: next(c["name"] for c in filtered if c["customer_id"] == value),
    )
    customer = store.get_customer(selected)
    profile_col, history_col = st.columns([1, 1.5], gap="large")
    with profile_col:
        st.markdown(
            f'<div class="profile-card"><div class="profile-avatar">{"".join(x[0] for x in customer["name"].split()[:2])}</div>'
            f'<div class="info-card-value">{safe_text(customer["name"])}</div>'
            f'<div class="info-card-foot">{safe_text(customer["tier"])} member since {safe_text(customer["joined_at"])}</div><hr>'
            f'<div>{safe_text(customer["email"])}</div><div>{safe_text(customer["phone"])}</div>'
            f'<div style="margin-top:.7rem">{safe_text(customer["address"])}</div></div>',
            unsafe_allow_html=True,
        )
    with history_col:
        st.markdown("### Order history")
        st.dataframe(
            [
                {
                    "Order": order["order_id"],
                    "Product": order["product"],
                    "Status": STATUS_LABELS.get(order["status"], order["status"]),
                    "Date": order["created_at"],
                    "Total": money(order["total"], order["currency"]),
                }
                for order in store.list_orders(selected)
            ],
            hide_index=True,
            width="stretch",
        )


def render_catalog(store):
    page_header("Product catalog", "Products, stock levels, customer ratings, and local product photography.")
    products = store.list_products()
    search_col, category_col = st.columns([2, 1])
    with search_col:
        search = st.text_input("Search catalog", placeholder="Product name or SKU")
    with category_col:
        categories = ["All categories", *sorted({p["category"] for p in products})]
        category = st.selectbox("Category", categories)
    filtered = [
        p for p in products
        if (category == "All categories" or p["category"] == category)
        and (not search or search.casefold() in f'{p["name"]} {p["sku"]}'.casefold())
    ]
    cards = st.columns(3, gap="large")
    for index, product in enumerate(filtered):
        with cards[index % 3]:
            image = project_file(ROOT, product["image"])
            st.image(image, width="stretch")
            st.markdown(
                f'<div class="info-card"><div class="info-card-title">{safe_text(product["category"])} · {safe_text(product["sku"])}</div>'
                f'<div class="info-card-value">{safe_text(product["name"])}</div>'
                f'<div class="info-card-foot">{safe_text(product["description"])}</div>'
                f'<div style="margin-top:.7rem"><strong>{money(product["price"], product["currency"])}</strong>'
                f' <span class="section-meta">· Stock {product["stock"]} · {product["rating"]:.1f} rating ({product["reviews"]} reviews)</span></div></div>',
                unsafe_allow_html=True,
            )


def render_knowledge():
    page_header("Knowledge library", "Published store policies the assistant can cite.")
    policies = load_policies(project_file(ROOT, "data/policies.json"))
    for policy in policies:
        with st.container(border=True):
            heading, source = st.columns([4, 1])
            with heading:
                st.markdown(f"### {policy.title}")
            with source:
                st.markdown(f'<span class="source-chip">{safe_text(policy.id)}</span>', unsafe_allow_html=True)
            st.write(policy.content)
            st.caption("Source record · Local policy fixture")


def render_database(store):
    page_header("Demo database", "A read-only view of the fictional records available to the workspace.")
    orders = store.list_orders()
    customers = store.list_customers()
    products = store.list_products()
    items = store.list_order_items()
    columns = st.columns(4)
    with columns[0]:
        metric_card("Orders", str(len(orders)), "Order records")
    with columns[1]:
        metric_card("Customers", str(len(customers)), "Fictional profiles")
    with columns[2]:
        metric_card("Products", str(len(products)), "Catalog records")
    with columns[3]:
        metric_card("Order lines", str(len(items)), "Product rows")
    st.write("")
    tabs = st.tabs(["Orders", "Order items", "Customers", "Products", "Policies"])
    with tabs[0]:
        st.dataframe(orders, hide_index=True, width="stretch")
    with tabs[1]:
        st.dataframe(items, hide_index=True, width="stretch")
    with tabs[2]:
        st.dataframe(customers, hide_index=True, width="stretch")
    with tabs[3]:
        st.dataframe(products, hide_index=True, width="stretch")
    with tabs[4]:
        policies = load_policies(project_file(ROOT, "data/policies.json"))
        st.dataframe(
            [
                {"Source": policy.id, "Policy": policy.title, "Keywords": ", ".join(policy.keywords), "Content": policy.content}
                for policy in policies
            ],
            hide_index=True,
            width="stretch",
        )
    csv_col, note_col = st.columns([1, 3])
    with csv_col:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(orders[0].keys()) if orders else [])
        if orders:
            writer.writeheader()
            writer.writerows(orders)
        st.download_button(
            "Export orders CSV",
            data=buffer.getvalue(),
            file_name="northstar-demo-orders.csv",
            mime="text/csv",
        )
    with note_col:
        st.caption("This local workspace contains synthetic data. The database pages are read-only.")


def main():
    store, _, customers = resources()
    render_sidebar(store, customers)
    page = st.session_state.navigation

    left, right = st.columns([4, 1])
    with left:
        st.markdown(
            f'<div class="page-kicker">NORTHSTAR SUPPORT / {safe_text(page.upper())}</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="section-meta" style="text-align:right;padding-top:.3rem">{date.today().strftime("%d %B %Y")}</div>',
            unsafe_allow_html=True,
        )

    if page == "Overview":
        render_overview(store, st.session_state.active_customer)
    elif page == "Inbox":
        render_inbox(store, st.session_state.active_customer)
    elif page == "Orders":
        render_orders(store, st.session_state.active_customer)
    elif page == "Customers":
        render_customers(store)
    elif page == "Catalog":
        render_catalog(store)
    elif page == "Knowledge":
        render_knowledge()
    elif page == "Database":
        render_database(store)


if __name__ == "__main__":
    main()
