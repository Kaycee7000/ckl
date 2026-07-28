"""Simple stdio-based MCP server: read JSON lines from stdin, dispatch to handlers, write responses to stdout."""
import sys
import json
from . import handlers
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
