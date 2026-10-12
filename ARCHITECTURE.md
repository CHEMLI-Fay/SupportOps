# SupportOps: target workflow

This document records the implementation plan before the full workflow is coded. The project uses fictional customers, orders, policies, and simulated actions.

## User journey

```text
Streamlit or notebook
  -> customer context (demo only)
  -> LangGraph agent -> Groq LLM chooses one allowed tool
       -> get_order     -> customer-bound order lookup
       -> search_policy -> local store policy lookup
       -> request_cancel -> preview -> human approval -> idempotent local update
  -> grounded answer
  -> optional LangSmith trace
```

The model may choose tools and write the final answer. Python validates tool names, arguments, ownership, approval, and action eligibility. A per-turn tool limit prevents unbounded loops. A missing or failed tool result must never be presented as a completed action. No real commerce API or payment service is used.

## Data and state

- `data/orders.json` and `data/policies.json` are synthetic seed data.
- A local SQLite database under `artifacts/` stores order status and action receipts. The directory stays out of Git.
- A SQLite LangGraph checkpointer stores paused runs so approval can be resumed by thread ID.
- The customer ID comes from application context, never from model-controlled tool arguments. The Streamlit customer selector is only a demo and is not authentication.
- A cancellation can execute only for an owned order in `processing` state, after explicit approval. A unique request ID makes retries idempotent.

## Observability and evaluation

- LangSmith tracing is opt-in through `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`. Local use works without a LangSmith account.
- The existing deterministic scenario suite stays as a regression baseline.
- DeepEval scores the reviewed deterministic scenarios. Python tests cover tool and action invariants; an optional Groq-backed judge checks answer quality. LLM judge scores are reported separately from exact safety checks.
- The final validation includes local workflow paths, approval/resume, idempotency, notebook execution, Streamlit startup, and live Groq calls when a key and quota are available.

## Delivery order

1. Add policy and processing-order fixtures, package dependencies, and configuration.
2. Add SQLite order/action storage and policy lookup.
3. Build the bounded LangGraph agent with Groq tool calling and approval interruption.
4. Add a clear notebook walkthrough and a small Streamlit chat interface.
5. Add DeepEval cases and optional LangSmith tracing.
6. Run the local and live checks, review outputs, document the observed results, and only then publish the final increment.

## Completion criteria

The project is ready for user review when a fresh environment can install the documented dependencies; the notebook visibly explains and runs the full graph; the Streamlit app can ask about order status and policy; an approved cancellation updates only an eligible demo order once; denial changes nothing; automated checks pass; and live LLM behavior has been observed and recorded. Any provider or account limitation must be reported explicitly.
