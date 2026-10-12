# Northstar Support

Northstar Support is a local support-operations workspace built as an agent-engineering project. It combines a Streamlit interface, a Groq-powered LangGraph assistant, a fictional SQLite customer database, human approval for simulated cancellations, DeepEval quality checks, and optional LangSmith observability.

All customers, orders, addresses, payment records, and products are fictional. The customer selector is a demo context, not authentication. No real commerce, payment, or CRM system is connected.

## What it demonstrates

- Bounded LLM tool calling with LangGraph and Groq.
- Customer-bound order and profile access.
- Pydantic validation for tool arguments and fixture data.
- Human-in-the-loop approval before a simulated cancellation.
- Idempotent SQLite actions and resumable LangGraph checkpoints.
- Streamlit operations views for orders, customers, products, policies, and database records.
- DeepEval deterministic checks and optional LLM-as-a-judge scoring.
- LangSmith traces, version fingerprints, datasets, and evaluation experiments.

## Architecture

```text
Streamlit or notebook
        |
        v
LangGraph support agent + Groq
        |
        +--> get_order ---------> customer-bound order lookup
        +--> search_database ---> profile/history or public catalog search
        +--> search_policy -----> local policy lookup
        +--> request_cancel ----> eligibility -> human approval -> SQLite update
        |
        v
Grounded answer + optional LangSmith trace
```

The model can select an allowed tool, but Python controls the customer identity, validates arguments, enforces the tool budget, checks authorization, and applies cancellation rules. Cancellation results are rendered from verified database state rather than freely rewritten by the model.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed workflow and safety boundaries.

## Requirements

- Windows PowerShell or an equivalent shell.
- Python 3.12 or newer.
- A Groq API key only for live model calls.
- A LangSmith API key only for remote tracing and experiments.

## Install and run

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
```

Set local credentials in `.env`. Never commit this file:

```dotenv
GROQ_API_KEY=your_groq_key
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=SupportOps
```

Start the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Try these questions in the Inbox:

```text
What is the status of ORD-1042?
How many orders do I have and what is my membership tier?
Which products are currently in stock?
What is the return policy?
Cancel my order ORD-1050.
```

Cancellation pauses for explicit approval. Use **Restore sample data** in **Demo controls** to reset the synthetic database and action receipts.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_evals.py
```

The repository contains 54 live regression scenarios: nine reviewed behaviors plus five natural paraphrases for each behavior. The offline baseline remains the deterministic routing suite; the live suite exercises the real Groq agent.

Run a live evaluation with the optional DeepEval judge:

```powershell
.\.venv\Scripts\python.exe scripts/run_evals.py --live --judge
```

Run the live suite and publish a LangSmith dataset/experiment:

```powershell
.\.venv\Scripts\python.exe scripts/run_evals.py --live --judge --langsmith --delay 20
```

Live calls consume provider quota. Reports are written under `artifacts/evaluations/` and include per-case checks, latency, reported tokens, judge scores, and aggregate summaries. The `artifacts/` directory is intentionally ignored by Git. The LangSmith dataset name changes when any live question or expected behavior changes.

## AgentOps walkthrough

Open [notebooks/03_agentops.ipynb](notebooks/03_agentops.ipynb) with the project virtual environment. It explains LangGraph execution, LangSmith tracing, DeepEval scoring, latency, token usage, quota errors, reports, and experiment comparison. The earlier workflow is [notebooks/02_full_workflow.ipynb](notebooks/02_full_workflow.ipynb). Live calls are disabled by default in notebooks.

Notebook outputs are cleared in Git. Run cells locally to generate your own plots and reports without publishing session traces.

## Reading LangSmith traces

Open the `SupportOps` project in LangSmith and filter for `support_request`. A normal trace contains the model call, selected tool, tool result, and final answer. An approved cancellation also contains the interruption and resumed run. Metadata includes model, prompt, data, evaluation, implementation, session, and source fingerprints. API keys are never included.

Evaluation experiments appear under LangSmith datasets and experiments. Each example records objective behavior feedback, `case_passed`, and `deepeval_grounding` when judging is enabled.

## Repository map

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit application and user workflows |
| `assets/` | CSS, brand mark, and product imagery |
| `src/support_ops/agent.py` | Groq agent and approval workflow |
| `src/support_ops/storage.py` | SQLite schema, access control, and actions |
| `src/support_ops/tools.py` | Validated order models and tools |
| `src/support_ops/records.py` | Validated customer and product records |
| `src/support_ops/observability.py` | LangSmith configuration and version labels |
| `src/support_ops/agent_evaluation.py` | Scenarios, DeepEval, reports, and LangSmith experiments |
| `scripts/run_evals.py` | Evaluation CLI |
| `data/` | Synthetic records, policies, and scenarios |
| `prompts/agent.txt` | Agent instructions and safety rules |
| `notebooks/` | Workflow and AgentOps walkthroughs |
| `tests/` | Regression tests |

## Limitations

This is a local learning and demonstration application, not a production CRM. The customer selector does not authenticate a real user. Data is synthetic, search is intentionally limited, and cancellation effects are simulated in SQLite. Live evaluation results depend on provider availability, quota, model behavior, and network access.

## License

This repository is currently an educational project without a published license.
