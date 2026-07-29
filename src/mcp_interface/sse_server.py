import json
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from knowledge_library.repository import ArtifactRepository
from mcp_interface import handlers


app = FastAPI(
    title="Simulation Intelligence MCP Server",
    description="HTTP/SSE transport wrapper for remote AI agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HANDLERS = {
    "simulate_scenario": handlers.simulate_scenario,
    "explain_causal_chain": handlers.explain_causal_chain,
    "query_knowledge_library": handlers.query_knowledge_library,
    "validate_historical_analogue": handlers.validate_historical_analogue,
}

# shared artifact repo
ARTIFACT_REPO = ArtifactRepository('.')


@app.get("/sse")
async def sse_endpoint(request: Request):
    """MCP SSE handshake endpoint."""

    async def event_generator():
        # registration event pointing clients to POST endpoint
        yield {"event": "endpoint", "data": "/messages"}
        while True:
            if await request.is_disconnected():
                break
            await asyncio.sleep(15)
            yield {"event": "ping", "data": "{}"}

    return EventSourceResponse(event_generator())


@app.post("/messages")
async def post_mcp_message(request: Request):
    """Processes standard Model Context Protocol (MCP) JSON-RPC requests."""
    try:
        data = await request.json()
        method = data.get("method")
        msg_id = data.get("id")
        params = data.get("params", {})

        # 1. Handle MCP Initialization
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "Simulation Intelligence MCP Server", "version": "1.0.0"}
                }
            }

        # 2. Acknowledge initialization notification
        if method == "notifications/initialized":
            return {"jsonrpc": "2.0", "result": {}}

        # 3. Handle Tool Listing
        if method == "tools/list":
            tools_list = [
                {
                    "name": name,
                    "description": f"Execution handler for {name}",
                    "inputSchema": {"type": "object", "properties": {}}
                }
                for name in HANDLERS.keys()
            ]
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": tools_list}
            }

        # 4. Handle Tool Execution (tools/call)
        if method == "tools/call":
            tool_name = params.get("name")
            payload = params.get("arguments", {})

            if tool_name not in HANDLERS:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown tool '{tool_name}'"}
                }

            handler = HANDLERS[tool_name]
            try:
                result = handler(payload, artifact_repo=ARTIFACT_REPO)
            except TypeError:
                result = handler(payload)

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": str(result)}]
                }
            }

        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "Method not found"}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
