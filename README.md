# SupportOps

A small customer-support learning project for a fictional online store. Its current scope is order lookup and controlled tool calling.

## Current scope

The project demonstrates two paths over fictional order data:

```text
Groq probe:  question -> model selects get_order -> validated lookup -> model answer
Local graph: question -> classify_request -> lookup or guidance -> answer
```

The probe uses `openai/gpt-oss-120b` through Groq. The LangGraph path is deterministic and does not call the model. Both paths use the same read-only order tool. The customer identity is bound in Python, outside model-controlled arguments. This is a demonstration boundary, not real authentication.

## Project layout

```text
AgentOps/
  config/settings.toml          # Non-secret settings
  data/orders.json              # Fictional orders
  notebooks/01_groq_tool_calling.ipynb
  prompts/support.txt
  src/support_ops/             # Configuration, tool, probe and graph
  .env.example
  .gitignore
  pyproject.toml
  README.md
```

`config.py` loads and validates settings. `tools.py` validates order data and controls access. `probe.py` performs the two-call Groq experiment. `graph.py` builds the local workflow and classifies requests using explicit Python rules. The notebook imports these modules and displays each step.

## Installation

Use Python 3.12 or newer. From the `AgentOps` directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The project uses `pyproject.toml` for its direct dependencies. `.venv` is local and excluded from Git. Select `.venv\Scripts\python.exe` as the notebook kernel; the older `.venv-py311` is not compatible with this project.

## Configuration

Create the local environment file only if it does not already exist:

```powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
```

Replace the placeholder in `.env` with your Groq API key. Keep the key out of notebook cells and Git. Existing environment variables take precedence over `.env`; restart the kernel after changing a key that was already loaded.

Model settings, file paths and the demo scenario live in `config/settings.toml`. If you change the demo order ID, update the question to match it. All configured file paths stay inside the checkout.

The local checks need no API key. A successful live probe makes two Groq requests and uses account quota.

## Notebook and validation

The local graph now accepts a customer question and routes it by an explicit rule:

```text
question -> classify_request
              | order status + one ID  -> lookup_order -> answer or unavailable
              | order status + no ID   -> ask_order_id
              | order status + two IDs -> ask_one_order_id
              + other request         -> unsupported
```

The classifier recognizes a small set of French and English status phrases. It does not use Groq, infer a missing order ID, or claim to understand arbitrary requests. The order lookup remains restricted to the bound demo customer. The notebook displays each category, path and answer.

Open `notebooks/01_groq_tool_calling.ipynb` in VS Code, select `.venv\Scripts\python.exe` as the kernel, restart it, and run all cells in order. The default `RUN_LIVE_API = False` runs local checks and the LangGraph example without calling Groq. The notebook shows an accessible order, a missing order and an order belonging to another customer.

To repeat the model experiment, set `RUN_LIVE_API = True` in the notebook and run the live section. Review its answer for language, correct status and invented facts. A non-empty answer is not proof of correctness.

You can also run the live probe from the project root without writing notebook outputs:

```powershell
.\.venv\Scripts\python.exe -c "import json; from support_ops.config import find_project_root; from support_ops.probe import run_order_probe; print(json.dumps(run_order_probe(find_project_root()), ensure_ascii=False, indent=2))"
```

This command consumes Groq quota. The probe returns selected result fields and omits raw provider error bodies.

## Data provenance

`data/orders.json` contains two fictional records written for this project. One belongs to the demo customer; the other checks the access boundary. No external dataset is required.

## Validation status and limits

The notebook's eleven code cells completed in sequence with live calls disabled, and its graph image was generated. Local checks covered the order-access boundary, status and unavailable branches, missing and multiple IDs, an unsupported request, and an English status question. Earlier checks covered four order statuses and invalid identifiers. Two previously observed Groq probes selected `get_order` and answered the configured question correctly in French. They took 0.839 and 0.899 seconds and used 646 tokens each. These are individual observations, not an accuracy or latency benchmark.

The live probe covers one order-status question and still needs human review. The project has no real authentication, store-policy retrieval, persistent sessions, human approval or business actions.

## Planned progression

1. Build a small reviewed scenario set for request routing and answer quality.
2. Add store-policy retrieval and stronger error handling.
3. Add human approval before any simulated action, then compare agent designs on shared cases.

## References

- [LangChain Groq integration](https://docs.langchain.com/oss/python/integrations/chat/groq)
- [Groq tool calling](https://console.groq.com/docs/tool-use/overview)
