import os
from mcp.server.fastmcp import FastMCP
from knowledge_library.repository import ArtifactRepository
from mcp_interface import handlers

# Initialize FastMCP server
mcp = FastMCP("Simulation Intelligence MCP Server")

# Initialize shared artifact repo
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
    # FastMCP runs a fully compliant SSE/HTTP transport server out of the box
    import uvicorn
    # If FastMCP exposes an app or custom runner, run via its settings or Starlette app
    app = mcp.starlette_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
