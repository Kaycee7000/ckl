cat << 'EOF' > /app/src/mcp_interface/sse_server.py
import os

try:
    from fastmcp import FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

from knowledge_library.repository import ArtifactRepository
from mcp_interface import handlers

mcp = FastMCP("Simulation Intelligence MCP Server")
ARTIFACT_REPO = ArtifactRepository('.')

@mcp.tool()
def simulate_scenario(payload: dict) -> str:
    """Execution handler for simulate_scenario"""
    try:
        return str(handlers.simulate_scenario(payload, artifact_repo=ARTIFACT_REPO))
    except TypeError:
        return str(handlers.simulate_scenario(payload))

@mcp.tool()
def explain_causal_chain(payload: dict) -> str:
    """Execution handler for explain_causal_chain"""
    try:
        return str(handlers.explain_causal_chain(payload, artifact_repo=ARTIFACT_REPO))
    except TypeError:
        return str(handlers.explain_causal_chain(payload))

@mcp.tool()
def query_knowledge_library(payload: dict) -> str:
    """Execution handler for query_knowledge_library"""
    try:
        return str(handlers.query_knowledge_library(payload, artifact_repo=ARTIFACT_REPO))
    except TypeError:
        return str(handlers.query_knowledge_library(payload))

@mcp.tool()
def validate_historical_analogue(payload: dict) -> str:
    """Execution handler for validate_historical_analogue"""
    try:
        return str(handlers.validate_historical_analogue(payload, artifact_repo=ARTIFACT_REPO))
    except TypeError:
        return str(handlers.validate_historical_analogue(payload))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    try:
        mcp.run(transport="sse", host=host, port=port)
    except Exception:
        import uvicorn
        app = mcp.starlette_app() if hasattr(mcp, "starlette_app") else mcp.app
        uvicorn.run(app, host=host, port=port)
EOF
