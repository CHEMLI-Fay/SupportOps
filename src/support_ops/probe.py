"""A bounded two-call probe for Groq tool calling."""

import json
from pathlib import Path
from time import perf_counter

from groq import APIConnectionError, APIStatusError
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langsmith import tracing_context

from support_ops.config import load_api_key, load_prompt, load_settings, project_file
from support_ops.tools import load_orders, make_order_tool


def run_order_probe(root: Path) -> dict:
    """Make two model calls and one read-only lookup; return publishable fields only.

    The first call uses automatic tool selection. A missing or unexpected tool
    call fails the probe instead of silently manufacturing a successful result.
    """
    settings = load_settings(root)
    prompt = load_prompt(root, settings)
    orders = load_orders(project_file(root, settings.files.orders))
    tool = make_order_tool(orders, settings.demo.customer_id)
    expected_args = {"order_id": settings.demo.order_id}
    demo_order = orders.get(settings.demo.order_id)
    if demo_order is None or demo_order.customer_id != settings.demo.customer_id:
        raise ValueError("The configured demo order must belong to the demo customer.")
    llm = ChatGroq(
        model=settings.llm.model,
        api_key=load_api_key(root),
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        timeout=settings.llm.timeout_seconds,
        max_retries=settings.llm.max_retries,
        reasoning_effort=settings.llm.reasoning_effort,
        model_kwargs={"include_reasoning": False},
    )
    messages = [SystemMessage(content=prompt), HumanMessage(content=settings.demo.question)]
    started = perf_counter()
    try:
        # Keep this first experiment local even if tracing is enabled in the shell.
        with tracing_context(enabled=False):
            first = llm.bind_tools([tool], tool_choice="auto").invoke(messages)
            if first.invalid_tool_calls or len(first.tool_calls) != 1:
                raise ValueError("Probe failed: expected exactly one valid tool call.")
            call = first.tool_calls[0]
            if call["name"] != tool.name or call["args"] != expected_args:
                raise ValueError("Probe failed: the model selected the wrong tool or order.")
            result = tool.invoke(call["args"])
            messages.extend([
                first,
                ToolMessage(content=json.dumps(result), tool_call_id=call["id"]),
            ])
            final = llm.invoke(messages)
    except APIStatusError as error:
        raise RuntimeError(
            f"Groq request failed (HTTP {error.status_code}). "
            "Check credentials, model access and account quotas. "
            "The provider response was omitted to keep notebook output safe."
        ) from None
    except APIConnectionError:
        raise RuntimeError("Groq connection failed or timed out. Check connectivity and retry later.") from None

    if final.response_metadata.get("finish_reason") == "length":
        raise ValueError("Probe failed: output reached the token limit; review the configured budget.")
    if final.tool_calls or not isinstance(final.content, str) or not final.content.strip():
        raise ValueError("Probe failed: expected a non-empty final text answer.")
    usage = [message.usage_metadata for message in (first, final)]
    total_tokens = (
        sum(item["total_tokens"] for item in usage) if all(item is not None for item in usage) else None
    )
    return {
        "model": settings.llm.model,
        "model_calls": 2,
        "tool_name": call["name"],
        "tool_arguments": call["args"],
        "tool_result": result,
        "answer": final.content.strip(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "total_tokens": total_tokens,
        "tool_selection_check": "passed",
        "answer_review": "pending_manual_review",
    }
