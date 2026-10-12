"""Trace context and reproducible version labels; never returns credentials."""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

from support_ops.config import load_settings


def version_labels(root: Path) -> dict:
    settings = load_settings(root)
    files = ["config/settings.toml", "prompts/agent.txt", "data/agent_scenarios.json",
             "data/orders.json", "data/customers.json", "data/products.json", "data/policies.json"]
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    implementation = hashlib.sha256()
    for path in sorted((root / "src/support_ops").glob("*.py")):
        implementation.update(path.name.encode())
        implementation.update(path.read_bytes())
    return {
        "model": settings.llm.model,
        "prompt_version": hashlib.sha256((root / "prompts/agent.txt").read_bytes()).hexdigest()[:12],
        "evaluation_version": digest.hexdigest()[:12],
        "implementation_version": implementation.hexdigest()[:12],
        "data_kind": "synthetic",
    }


def tracing_status(root: Path) -> dict:
    load_dotenv(root / ".env", override=False)
    enabled = os.getenv("LANGSMITH_TRACING", "").lower() in {"true", "1"}
    configured = bool(os.getenv("LANGSMITH_API_KEY", "").strip())
    return {
        "enabled": enabled,
        "key_configured": configured,
        "project": os.getenv("LANGSMITH_PROJECT", "SupportOps"),
        "ready": enabled and configured,
    }


def run_config(root: Path, thread_id: str, source: str = "streamlit") -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "support_request",
        "tags": ["supportops", source],
        "metadata": {**version_labels(root), "session_id": thread_id, "source": source},
    }
