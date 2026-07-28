# CKL — Causal Knowledge & Simulation Library

A lightweight scaffold for building causal reasoning, simulation, and continuous-learning workflows in Python.

Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project layout](#project-layout)
- [Quickstart](#quickstart)
- [Examples](#examples)
- [Development](#development)
- [Contributing](#contributing)

## Overview

CKL provides modular components for historical data ingestion, probabilistic and deterministic modeling, causal reasoning (SCM), multi-scenario simulation, and lightweight MCP-compatible tooling for integrations and automation.

## Features

- Historical data ingestion and point-in-time queries
- Probabilistic modeling and Monte Carlo simulation
- Simple structural causal models (SCM) and counterfactual operators
- Scenario generation, clustering, and sensitivity analysis
- Validation utilities and probabilistic scoring (CRPS, log score, coverage)
- Knowledge extraction and a small artifact library with JSON Schema
- Continuous-learning primitives: residual monitoring, proposal system, gatekeeper
- MCP stdio server for simple tool-based integrations

## Project layout

- `src/historical_data_engine` — ingestion, DuckDB-backed storage, `PointInTimeQuery`
- `src/modeling_engine` — Monte Carlo runner, Bayesian helpers, dynamics
- `src/causal_engine` — SCM implementation, `do()` operator, counterfactual helpers
- `src/simulation_engine` — scenario sampling, clustering, sensitivity analysis
- `src/validation_engine` — backtest logic and scoring metrics
- `src/knowledge_engine` — abstraction, mining, confidence scoring
- `src/knowledge_library` — artifact schema and local repository
- `src/continuous_learning` — residual monitor, proposer, gatekeeper
- `src/mcp_interface` — MCP handlers, request schemas, stdio/sse servers
- `tests/` — unit tests covering core components

## Quickstart

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the test suite:

```bash
pytest -q
```

Start the local stdio MCP server (reads JSON lines from stdin):

```bash
python src/mcp_interface/stdio_server.py
```

## Examples

Example MCP request (send one JSON line to the server's stdin):

```json
{"tool":"simulate_scenario","payload":{"domain":"test","params":[1,2],"horizon":3}}
```

Run a quick scenario runner from Python (example usage):

```python
from src.simulation_engine.scenario import ScenarioRunner

runner = ScenarioRunner()
result = runner.run({'horizon': 10, 'params': {}})
print(result.summary())
```

## Development

- Follow the Quickstart to set up a virtualenv.
- Run `pytest` frequently and add tests for new behavior.
- Keep changes small and focused; open a PR against `main` with a descriptive title.

## Contributing

Contributions are welcome. Open issues for bugs or feature requests and submit PRs with tests and documentation updates.

----

This README was updated to include a concise Quickstart and examples. If you'd like additional sections (badges, CI, or a generated roadmap file), tell me what to add and I can update the file or open a PR.