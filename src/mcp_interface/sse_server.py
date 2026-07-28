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
    """Processes incoming JSON tool requests from remote clients."""
    try:
        data = await request.json()
        tool_name = data.get("tool")
        payload = data.get("payload", {})

        if tool_name not in HANDLERS:
            raise HTTPException(status_code=400, detail=f"Unknown tool '{tool_name}'")

        handler = HANDLERS[tool_name]
        # call handler with shared artifact repo when signature accepts it
        try:
            result = handler(payload, artifact_repo=ARTIFACT_REPO)
        except TypeError:
            # fallback if handler signature differs
            result = handler(payload)

        return {"jsonrpc": "2.0", "id": data.get("id"), "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
