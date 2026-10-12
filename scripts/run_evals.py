"""Run local DeepEval checks and optional live Groq evaluations."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")

from deepeval.metrics import ExactMatchMetric, GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from support_ops.agent import build_support_agent
from support_ops.config import find_project_root, load_api_key, load_settings, project_file
from support_ops.evaluation import evaluate_scenarios, load_scenarios
from support_ops.graph import build_order_graph
from support_ops.storage import SupportStore
from support_ops.tools import load_orders


class GroqJudge(DeepEvalBaseLLM):
    """Use the configured Groq model as DeepEval's optional LLM judge."""

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            return self.model.with_structured_output(schema, method="json_schema").invoke(prompt)
        return str(self.model.invoke(prompt).content)

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            return await self.model.with_structured_output(
                schema, method="json_schema"
            ).ainvoke(prompt)
        return str((await self.model.ainvoke(prompt)).content)

    def get_model_name(self):
        return "Groq support judge"


def run_offline(root: Path) -> dict:
    settings = load_settings(root)
    orders = load_orders(project_file(root, settings.files.orders))
    graph = build_order_graph(orders, settings.demo.customer_id)
    scenarios = load_scenarios(project_file(root, "data/scenarios.json"))
    results = evaluate_scenarios(graph, scenarios)
    scored = []
    for scenario, result in zip(scenarios, results, strict=True):
        case = LLMTestCase(
            input=scenario.question,
            actual_output=result["answer"],
            expected_output=scenario.answer,
        )
        metric = ExactMatchMetric()
        metric.measure(case)
        scored.append(
            {
                "id": scenario.id,
                "passed": result["passed"] and metric.is_successful(),
                "path": result["path"],
                "answer_score": metric.score,
            }
        )
    return {"passed": sum(row["passed"] for row in scored), "total": len(scored), "cases": scored}


def final_answer(state: dict) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return str(message.content)
    raise ValueError("Agent did not produce a final answer.")


def run_live(root: Path, judge_enabled: bool) -> dict:
    settings = load_settings(root)
    with TemporaryDirectory(prefix="supportops-eval-") as directory:
        store = SupportStore(Path(directory))
        store.initialize(project_file(root, settings.files.orders))
        graph = build_support_agent(
            root, settings.demo.customer_id, store=store, checkpointer=InMemorySaver()
        )
        cases = [
            {
                "id": "status",
                "question": "Quel est le statut de ma commande ORD-1042 ?",
                "reference": "La commande ORD-1042 est expédiée.",
                "required_tool": "get_order",
            },
            {
                "id": "return_policy",
                "question": "Quelle est la politique de retour ?",
                "reference": "POL-RETURNS: un article inutilisé peut être retourné dans les 14 jours après livraison. Contacter un agent humain pour commencer le retour.",
                "required_tool": "search_policy",
            },
        ]
        judge = None
        if judge_enabled:
            model = ChatGroq(
                model=settings.llm.model,
                api_key=load_api_key(root),
                temperature=0,
                max_tokens=settings.llm.max_tokens,
                timeout=settings.llm.timeout_seconds,
                max_retries=0,
                reasoning_effort=settings.llm.reasoning_effort,
                model_kwargs={"include_reasoning": False},
            )
            judge = GroqJudge(model)
        results = []
        for case in cases:
            config = {"configurable": {"thread_id": str(uuid4())}}
            state = graph.invoke(
                {"messages": [HumanMessage(content=case["question"])], "tool_calls_used": 0},
                config=config,
            )
            tool_names = [
                call["name"]
                for message in state["messages"]
                if isinstance(message, AIMessage)
                for call in message.tool_calls
            ]
            if case["required_tool"] not in tool_names:
                raise AssertionError(f"{case['id']}: required tool was not called.")
            answer = final_answer(state)
            record = {"id": case["id"], "tools": tool_names, "answer": answer}
            if judge is not None:
                metric = GEval(
                    name="Grounded support answer",
                    criteria="Judge whether the answer is faithful to the reference facts, answers the user's question, and makes no unsupported promises. Do not require identical wording.",
                    evaluation_params=[
                        SingleTurnParams.INPUT,
                        SingleTurnParams.ACTUAL_OUTPUT,
                        SingleTurnParams.EXPECTED_OUTPUT,
                    ],
                    model=judge,
                    threshold=0.7,
                    async_mode=False,
                )
                metric.measure(
                    LLMTestCase(
                        input=case["question"],
                        actual_output=answer,
                        expected_output=case["reference"],
                    )
                )
                record["judge_score"] = metric.score
                record["judge_passed"] = metric.is_successful()
                record["judge_reason"] = metric.reason
            results.append(record)
        cancel_config = {"configurable": {"thread_id": str(uuid4())}}
        pending = graph.invoke(
            {
                "messages": [HumanMessage(content="Annule ma commande ORD-1050.")],
                "tool_calls_used": 0,
            },
            config=cancel_config,
        )
        if "__interrupt__" not in pending:
            raise AssertionError("Cancellation did not pause for approval.")
        if store.get_order(settings.demo.customer_id, "ORD-1050")["status"] != "processing":
            raise AssertionError("Order changed before approval.")
        completed = graph.invoke(Command(resume=True), config=cancel_config)
        if store.get_order(settings.demo.customer_id, "ORD-1050")["status"] != "cancelled":
            raise AssertionError("Approved cancellation did not update the order.")
        results.append(
            {
                "id": "approved_cancellation",
                "approval_pause": True,
                "answer": final_answer(completed),
                "status_after": "cancelled",
            }
        )
        repeated = graph.invoke(
            {
                "messages": [HumanMessage(content="Annule ma commande ORD-1050.")],
                "tool_calls_used": 0,
            },
            config={"configurable": {"thread_id": str(uuid4())}},
        )
        repeated_answer = final_answer(repeated)
        if "__interrupt__" in repeated or "déjà annulée" not in repeated_answer:
            raise AssertionError("Repeated cancellation did not report the saved status.")
        results.append(
            {
                "id": "repeated_cancellation",
                "approval_pause": False,
                "answer": repeated_answer,
                "status_after": store.get_order(settings.demo.customer_id, "ORD-1050")["status"],
            }
        )
        return {"model": settings.llm.model, "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call the Groq support agent.")
    parser.add_argument("--judge", action="store_true", help="Use Groq as a DeepEval G-Eval judge.")
    args = parser.parse_args()
    if args.judge and not args.live:
        parser.error("--judge requires --live")
    root = find_project_root(Path(__file__).parent)
    offline = run_offline(root)
    print(json.dumps({"offline": offline}, ensure_ascii=True, indent=2))
    if offline["passed"] != offline["total"]:
        raise SystemExit(1)
    if args.live:
        live = run_live(root, args.judge)
        print(json.dumps({"live": live}, ensure_ascii=True, indent=2))
        if args.judge and any(
            case.get("judge_passed") is False for case in live["cases"]
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
