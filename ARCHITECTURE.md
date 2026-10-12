# SupportOps: workflow architecture

The project uses fictional customers, orders, policies, and simulated actions.

## User journey

```text
Streamlit or notebook
  -> customer context (demo only)
  -> LangGraph agent -> Groq LLM chooses one allowed tool
       -> get_order     -> customer-bound order lookup
       -> search_database -> customer-bound profile/history or public catalog search
       -> search_policy -> local store policy lookup
       -> request_cancel -> preview -> human approval -> idempotent local update
  -> grounded answer
  -> optional LangSmith trace
```

The model may choose tools and write answers for read-only questions. Python produces the final cancellation outcome from verified tool data, and also validates tool names, arguments, ownership, approval, and action eligibility. A per-turn tool limit prevents unbounded loops. A missing or failed tool result must never be presented as a completed action. No real commerce API or payment service is used.

## Data and state

- `data/orders.json`, `data/customers.json`, `data/products.json`, and `data/policies.json` are synthetic seed data.
- A local SQLite database under `artifacts/` stores order status and action receipts. The directory stays out of Git.
- A SQLite LangGraph checkpointer stores paused runs so approval can be resumed by thread ID.
- The customer ID comes from application context, never from model-controlled tool arguments. The Streamlit customer selector is only a demo and is not authentication.
- A cancellation can execute only for an owned order in `processing` state, after explicit approval. A unique request ID makes retries idempotent.

## Observability and evaluation

- LangSmith tracing is opt-in through `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`. Local use works without a LangSmith account.
- The existing deterministic scenario suite stays as a regression baseline.
- `observability.py` labels app runs with conversation, model, prompt, configuration/data and implementation versions.
- `agent_evaluation.py` runs 54 versioned agent cases against isolated databases, persists a report after each case, and can publish the dataset and experiment with the LangSmith SDK.
- Experiment feedback separates objective behavior checks from the DeepEval judge score. API failures remain errors; no missing score is treated as success.
- `notebooks/03_agentops.ipynb` explains traces, metrics, quota errors, measured timing/usage and experiment comparison outside the customer interface.
- DeepEval scores the reviewed deterministic scenarios. Python tests cover tool and action invariants; an optional Groq-backed judge checks answer quality. LLM judge scores are reported separately from exact safety checks.
- The notebooks contain executable source cells. Their outputs are cleared before publication so local traces and session data stay in `artifacts/`.

## Cancellation outcome

The order tool reports the actual status when cancellation is unavailable, including an already cancelled order. After an eligible request is approved or declined, Python builds the final answer from the verified outcome. The action receipt makes a repeated approval idempotent.

## Evaluation interpretation

The 54 live cases cover nine behavior categories and five additional phrasings of each. Paraphrases are regression coverage for the same fixtures; they are not independent customer samples. A local scripted-model test verifies tool contracts, while a live run requires Groq. DeepEval's judge scores answer quality separately from the deterministic access, approval, and state checks.
