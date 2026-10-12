# SupportOps

SupportOps is a small agent engineering project for a fictional online store. A Groq LLM can answer order and policy questions by choosing tools in a LangGraph workflow. A cancellation pauses for human approval before a local SQLite order is changed. The project includes a visual notebook, a Streamlit chat, optional LangSmith traces, and DeepEval checks.

All orders, customers, policies, and actions are synthetic. The customer selector is a demo context, not authentication. No real store, payment, or shipping service is connected.

## How it works

```text
Customer question
      |
      v
LangGraph agent (Groq LLM, up to 4 tool calls)
      |
      +--> get_order -------> customer-bound SQLite lookup
      +--> search_policy ---> local policy with source ID
      +--> request_cancel --> check ownership and status
                                |
                                v
                         human approval pause
                           |            |
                          deny        approve
                           |            |
                         no change   one SQLite update
      |
      v
Answer in the customer's language
```

The model chooses a tool and writes an answer. Python owns the customer context, validates tool arguments, checks cancellation eligibility, and applies an approved update. Only `processing` orders can be cancelled. A request ID prevents the same approved action from running twice. The workflow saves pending approvals in a local SQLite LangGraph checkpointer, so the same conversation can resume after a rerun.

The earlier deterministic graph remains as a baseline for exact routing cases. It uses explicit Python rules and does not call Groq. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and boundaries.

Each part has one job:

| Part | Where | What it does |
| --- | --- | --- |
| LangGraph + Groq | `src/support_ops/agent.py` | Decides which tool to use, pauses for approval, and answers |
| Streamlit | `app.py` | Shows the chat, demo orders, and approval buttons |
| DeepEval | `scripts/run_evals.py` | Runs repeatable checks and optional LLM judging |
| LangSmith | `.env` tracing settings | Records live runs for inspection; it does not decide answers |

This separation lets the notebook and Streamlit use the same workflow while evaluations and traces observe it.

## Quick start on Windows

Use Python 3.12 or newer. In PowerShell, from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
```

Put your Groq key in the local `.env` file as `GROQ_API_KEY=...`. Do not commit `.env`. The offline notebook sections and exact checks work without an API key.

Run the chat app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Try `Quel est le statut de ma commande ORD-1042 ?`, `Quelle est la politique de retour ?`, and `Annule ma commande ORD-1050.` The last request should show approval buttons. The sidebar displays the demo customer's orders and has a reset button to restore the synthetic data.

## Learn in the notebooks

Open `notebooks/02_full_workflow.ipynb` in VS Code. Select the interpreter **`.venv\Scripts\python.exe`**, restart the kernel, and run all cells. It displays the agent graph as an image, tool results, a denied and approved cancellation, and offline evaluation results. Its default `RUN_LIVE_AGENT = False` avoids API calls; set it to `True` to inspect real Groq tool choices. The live cancellation cell pauses; change `APPROVE_LIVE_CANCELLATION` only when you want to resume that synthetic action.

`notebooks/01_groq_tool_calling.ipynb` explains the initial single-tool probe and deterministic routing graph. Its live section is also off by default.

## Evaluation and tracing

Run the offline checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\run_evals.py
```

The offline DeepEval run scores the reviewed deterministic scenarios with exact matches. The tests also cover customer isolation, policy sources, approval and denial, persistent resume, and idempotency. These checks establish local behavior; they do not measure general LLM accuracy.

Run the optional live agent checks, then add a Groq-backed DeepEval judge if desired:

```powershell
.\.venv\Scripts\python.exe scripts\run_evals.py --live
.\.venv\Scripts\python.exe scripts\run_evals.py --live --judge
```

The live command checks that Groq calls the required tools for status and policy questions and pauses before cancellation. `--judge` scores answer grounding against short reference facts. Judge scores are separate from exact tool and safety checks, and live runs use API quota.

To send LangGraph traces to LangSmith, add these fields to `.env`:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=SupportOps
```

Then run the app or a live evaluation and open the **SupportOps** project in LangSmith. Tracing is optional; setting it to `false` keeps local execution available. An existing shell environment variable takes precedence over `.env`, so restart the terminal or notebook kernel after changing one.

## Files

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit customer chat and approval controls |
| `src/support_ops/agent.py` | Groq tool loop and LangGraph approval pause |
| `src/support_ops/storage.py` | Customer-bound SQLite orders and idempotent actions |
| `src/support_ops/policies.py` | Search over written synthetic policies |
| `src/support_ops/graph.py` | Earlier deterministic routing baseline |
| `scripts/run_evals.py` | Offline DeepEval and optional live agent checks |
| `data/` | Fictional orders, policies, and reviewed scenarios |
| `notebooks/` | Step-by-step visual walkthroughs |
| `config/settings.toml` and `prompts/` | Model settings and instructions |

`artifacts/` holds local SQLite data and is ignored by Git. Use **Reset demo orders** in Streamlit to restore the initial processing order.

## Validation observed on 12 October 2026

- Six Python tests passed; all ten reviewed DeepEval scenarios passed.
- Both notebooks executed from start to finish with live mode off and each saved a graph image, with no cell errors.
- The Streamlit app rendered its title, chat input, and controls without an exception.
- In a live Groq run, the status question called `get_order`, the return question called `search_policy`, and cancellation paused before approval. After approval, the synthetic order changed to `cancelled`.
- The optional Groq-backed judge scored the two checked answers 1.0 each in that run. This is one observation, not a benchmark.
- LangSmith reported a `SupportOps` project with recent runs after tracing was enabled.

## Limits

This is a learning demo, not a deployed customer-support system. The sidebar customer selector provides no identity verification. Policy retrieval is keyword based, so unsupported questions may return no source. Live answer quality depends on the selected model and must be reviewed; the exact tests primarily protect tool use and state transitions.

## References

- [LangGraph interrupts and human approval](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangChain Groq integration](https://docs.langchain.com/oss/python/integrations/chat/groq)
- [LangSmith tracing with LangGraph](https://docs.langchain.com/langsmith/trace-with-langgraph)
- [DeepEval evaluation guide](https://deepeval.com/docs/evaluation-introduction)
