"""Simple stdio-based MCP server: read JSON lines from stdin, dispatch to handlers, write responses to stdout.

This file supports both module-style execution (`python -m src.mcp_interface.stdio_server`)
and direct script execution (`python src/mcp_interface/stdio_server.py`). It first
attempts the relative import expected when run as a module, and falls back to an
absolute import after adding the `src/` folder to `sys.path` when run as a script.
"""
import sys
import os
import json

try:
    # when run as a module (python -m src.mcp_interface.stdio_server)
    from . import handlers
except Exception:
    # running as a script; ensure the repository `src/` root is on sys.path
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from mcp_interface import handlers

from knowledge_library.repository import ArtifactRepository


def main_loop(repo_root: str = "."):
    repo = ArtifactRepository(repo_root)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            tool = req.get("tool")
            payload = req.get("payload", {})
            if tool == "simulate_scenario":
                out = handlers.simulate_scenario(payload, artifact_repo=repo)
            elif tool == "explain_causal_chain":
                out = handlers.explain_causal_chain(payload, artifact_repo=repo)
            elif tool == "query_knowledge_library":
                out = handlers.query_knowledge_library(payload, artifact_repo=repo)
            elif tool == "validate_historical_analogue":
                out = handlers.validate_historical_analogue(payload, artifact_repo=repo)
            else:
                out = {"error": "unknown_tool"}
        except Exception as e:
            out = {"error": str(e)}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main_loop()
