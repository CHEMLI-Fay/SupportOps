"""Streamlit chat for the fictional SupportOps workflow."""

import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from support_ops.agent import build_support_agent, make_checkpointer
from support_ops.config import find_project_root, load_settings, project_file
from support_ops.storage import SupportStore


ROOT = find_project_root(Path(__file__).parent)
load_dotenv(ROOT / ".env", override=False)
SETTINGS = load_settings(ROOT)

st.set_page_config(page_title="SupportOps", page_icon="📦", layout="centered")


@st.cache_resource
def resources():
    store = SupportStore(ROOT)
    store.initialize(project_file(ROOT, SETTINGS.files.orders))
    return store, make_checkpointer(ROOT)


@st.cache_resource
def agent_for(customer_id: str):
    store, checkpointer = resources()
    return build_support_agent(
        ROOT, customer_id, store=store, checkpointer=checkpointer
    )


def new_conversation():
    st.session_state.thread_id = str(uuid4())
    st.session_state.chat = []
    st.session_state.pending = None


def run_input(value):
    agent = agent_for(st.session_state.customer_id)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    return agent.invoke(value, config=config)


def record_result(result):
    if "__interrupt__" in result:
        st.session_state.pending = result["__interrupt__"][0].value
        return
    st.session_state.pending = None
    replies = [
        message for message in result["messages"]
        if isinstance(message, AIMessage) and not message.tool_calls and message.content
    ]
    if replies:
        st.session_state.chat.append(("assistant", str(replies[-1].content)))
    else:
        st.session_state.chat.append(("assistant", "Je n'ai pas pu terminer cette demande."))


if "customer_id" not in st.session_state:
    st.session_state.customer_id = SETTINGS.demo.customer_id
if "thread_id" not in st.session_state:
    new_conversation()

st.title("📦 SupportOps")
st.caption("Ask about a fictional order or store policy. Cancellation requires your approval.")

with st.sidebar:
    st.subheader("Demo context")
    st.selectbox(
        "Customer",
        ["CUST-001", "CUST-002"],
        key="customer_id",
        on_change=new_conversation,
        help="This selector is a demo context, not authentication.",
    )
    store, _ = resources()
    st.write("Orders visible to this demo customer")
    st.dataframe(store.list_orders(st.session_state.customer_id), hide_index=True)
    if st.button("New conversation"):
        new_conversation()
        st.rerun()
    if st.button("Reset demo orders"):
        store.reset_demo(project_file(ROOT, SETTINGS.files.orders))
        new_conversation()
        st.rerun()
    st.divider()
    st.caption("Try: 'Quel est le statut de ORD-1042 ?'")
    st.caption("Try: 'Quelle est la politique de retour ?'")
    st.caption("Try: 'Annule ma commande ORD-1050.'")
    st.caption(
        "LangSmith tracing: "
        + ("on" if os.getenv("LANGSMITH_TRACING", "").lower() == "true" else "off")
    )

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.write(content)

if st.session_state.pending:
    pending = st.session_state.pending
    st.warning(f"Approve cancellation of {pending['order_id']}?")
    left, right = st.columns(2)
    with left:
        approve = st.button("Approve cancellation", type="primary")
    with right:
        deny = st.button("Deny")
    if approve or deny:
        try:
            record_result(run_input(Command(resume=bool(approve))))
        except Exception:
            st.error("The request failed. Check the terminal and try again.")
        st.rerun()

prompt = st.chat_input(
    "Ask about an order or policy",
    disabled=bool(st.session_state.pending),
)
if prompt:
    st.session_state.chat.append(("user", prompt))
    try:
        with st.spinner("Checking the request..."):
            result = run_input(
                {"messages": [HumanMessage(content=prompt)], "tool_calls_used": 0}
            )
        record_result(result)
    except Exception:
        st.session_state.chat.append(
            ("assistant", "La demande a échoué. Vérifiez la connexion et la configuration.")
        )
    st.rerun()
