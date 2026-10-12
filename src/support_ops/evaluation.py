"""Evaluate the local graph against reviewed synthetic scenarios."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from support_ops.graph import Category


NodeName = Literal[
    "classify_request",
    "lookup_order",
    "answer",
    "unavailable",
    "ask_order_id",
    "ask_one_order_id",
    "unsupported",
]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    question: str = Field(min_length=1)
    category: Category
    path: list[NodeName] = Field(min_length=1)
    answer: str = Field(min_length=1)


class ScenarioSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    scenarios: list[Scenario] = Field(min_length=1)


def load_scenarios(path: Path) -> list[Scenario]:
    """Load and validate the versioned scenario set."""
    try:
        suite = ScenarioSuite.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError):
        raise ValueError("Invalid data/scenarios.json: check its schema and values.") from None
    ids = [scenario.id for scenario in suite.scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Scenario identifiers must be unique.")
    return suite.scenarios


def evaluate_scenarios(graph, scenarios: list[Scenario]) -> list[dict]:
    """Run each question and report category, path, and answer mismatches."""
    results = []
    for scenario in scenarios:
        state = {"question": scenario.question}
        path = []
        for update in graph.stream(state, stream_mode="updates"):
            node_name, changes = next(iter(update.items()))
            path.append(node_name)
            state.update(changes)
        actual = {
            "category": state.get("category"),
            "path": path,
            "answer": state.get("answer"),
        }
        expected = scenario.model_dump(include={"category", "path", "answer"})
        mismatches = [field for field in expected if actual[field] != expected[field]]
        results.append(
            {
                "id": scenario.id,
                "passed": not mismatches,
                "category": actual["category"],
                "path": path,
                "answer": actual["answer"],
                "mismatches": mismatches,
            }
        )
    return results
