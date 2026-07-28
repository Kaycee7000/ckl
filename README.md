# CKL — Causal Knowledge + Simulation Library (scaffold)

This repository contains a scaffold for a modular causal reasoning and simulation platform. It includes:

- Historical data ingestion and point-in-time queries
- Mathematical modeling (deterministic & probabilistic engines)
- Causal reasoning (SCM, do-operator, counterfactuals)
- Multi-scenario simulation, clustering, and sensitivity analysis
- Historical validation and probabilistic scoring (CRPS, log score, coverage)
- Knowledge extraction, structured artifact library, and confidence scoring
- Continuous learning primitives (residual monitoring, proposals, gatekeeper)
- A lightweight MCP-compatible stdio interface for basic tool calls

Project layout (key folders)

- `src/historical_data_engine` — DuckDB storage, ingestion, `PointInTimeQuery`
- `src/modeling_engine` — Monte Carlo runner, Bayesian updater, dynamics solver
- `src/causal_engine` — Simple SCM, `do()` operator, counterfactuals
- `src/simulation_engine` — LHS sampling, `ScenarioRunner`, clustering, sensitivity
- `src/validation_engine` — Walk-forward backtester and probabilistic metrics
- `src/knowledge_engine` — Subgraph mining, variable abstraction, confidence scoring
- `src/knowledge_library` — Artifact JSON Schema + local repository and TF-IDF retrieval
- `src/continuous_learning` — Residual monitor, proposer, gatekeeper
- `src/mcp_interface` — MCP tool handlers, request schemas, and stdio server
- `tests/` — unit tests covering core scaffolds
- `requirements.txt` — Python dependencies

Implementation Roadmap

Phase 1: Foundations (Sprints 1–3)
	- Epic 1: Historical Data Engine
	- Epic 2: Mathematical Modeling Engine
	- Epic 9: Basic MCP Server Interface

Phase 2: Intelligence & Simulation (Sprints 4–6)
	- Epic 3: Causal Reasoning Engine
	- Epic 4: Multi-Scenario Simulator
	- Epic 5: Historical Validation Engine

Phase 3: Knowledge & Learning (Sprints 7–9)
	- Epic 6: Knowledge Extraction Engine
	- Epic 7: Structured Knowledge Library
	- Epic 8: Continuous Learning Engine

Quickstart

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the test suite:

```bash
pytest -q
```

Start the simple stdio MCP server (reads JSON lines from stdin):

```bash
python src/mcp_interface/stdio_server.py
```

Example MCP request (send as one JSON line to the server's stdin):

```json
{"tool":"simulate_scenario","payload":{"domain":"test","params":[1,2],"horizon":3}}
```

Notes & Next Steps

- The repository is a scaffold with minimal implementations and unit tests. For production use, replace local stores with managed services (Postgres/JSONB, vector DB), add authentication, persistence for monitors, and scale-out orchestration.
- Suggested next improvements: FastMCP integration (SSE), embedding-based semantic search for `knowledge_library`, production backtester hooks, and PyMC-based Bayesian updaters.

If you want, I can generate a printable roadmap file (Markdown) or open a PR with these changes. 

# ckl