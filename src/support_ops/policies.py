"""Small, explicit policy search over synthetic store rules."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(pattern=r"^POL-[A-Z]+$")
    title: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)
    content: str = Field(min_length=1)


def load_policies(path: Path) -> list[Policy]:
    try:
        policies = TypeAdapter(list[Policy]).validate_python(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, ValidationError):
        raise ValueError("Invalid data/policies.json: check its schema and values.") from None
    if len({policy.id for policy in policies}) != len(policies):
        raise ValueError("Policy identifiers must be unique.")
    return policies


def search_policies(policies: list[Policy], query: str) -> dict:
    """Return the most relevant local rule with its source ID."""
    text = query.casefold().strip()
    if not text:
        return {"found": False, "error": "policy_unavailable"}
    ranked = []
    for policy in policies:
        score = sum(1 for keyword in policy.keywords if keyword.casefold() in text)
        if score:
            ranked.append((score, policy))
    if not ranked:
        return {"found": False, "error": "policy_unavailable"}
    ranked.sort(key=lambda item: (-item[0], item[1].id))
    policy = ranked[0][1]
    return {
        "found": True,
        "source_id": policy.id,
        "title": policy.title,
        "content": policy.content,
    }
