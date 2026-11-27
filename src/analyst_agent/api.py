"""FastAPI app: POST /ask (API key, rate limited), GET /health, GET /ready.

Run with:  uvicorn analyst_agent.api:app

At start-up the app opens, once:
* the MCP server over the database (a stdio subprocess, or MCP_URL if set),
* the checkpointer (Postgres when DATABASE_URL is set, else memory),
* Langfuse tracing when its keys are set,
and builds the LangGraph agent with the models from models.py.

Settings (environment or .env): DB_PATH, DOCS_PATH, MCP_URL, DATABASE_URL, MODEL*, API_KEYS,
AUTH_DISABLED, RATE_LIMIT_PER_MINUTE, DAILY_QUERY_CAP, LANGFUSE_*.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from analyst_agent.agent import AgentAnswer
from analyst_agent.checkpoint import open_checkpointer
from analyst_agent.graph import GraphAgent
from analyst_agent.mcp_tools import MCPTools
from analyst_agent.models import ModelConfig
from analyst_agent.observability import setup_tracing
from analyst_agent.security import KeyStore, RateLimiter

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

log = logging.getLogger("analyst_agent.api")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

# HTTP status for each run status that has no answer.
FAILURE_CODES = {"timeout": 504, "llm_error": 502, "tool_error": 502}
PUBLIC_ERRORS = {
    "timeout": "The answer took too long. Please try again.",
    "llm_error": "The language model is unavailable. Please try again later.",
    "tool_error": "The database is unavailable. Please try again later.",
    "sql_error": "I could not write a working query for this question. Try rephrasing it.",
    "max_steps": "I could not finish within the step limit. Try a simpler question.",
}


# ------------------------------------------------------------------ runtime


@dataclass
class Runtime:
    """Everything a request needs, created once per process."""

    agent: GraphAgent
    keys: KeyStore
    limiter: RateLimiter
    callbacks: list = field(default_factory=list)
    info: dict = field(default_factory=dict)  # shown by /ready, no secrets


@asynccontextmanager
async def open_runtime():
    """Build the Runtime from the environment. Used as the app's lifespan."""
    async with AsyncExitStack() as stack:
        db_path = ROOT / os.getenv("DB_PATH", "data/demo.db")
        docs_env = os.getenv("DOCS_PATH", "data/demo_docs.md")
        docs_path = ROOT / docs_env if docs_env else None
        mcp_url = os.getenv("MCP_URL")
        tools = MCPTools.http(mcp_url) if mcp_url else MCPTools.stdio(db_path, docs_path)
        await stack.enter_async_context(tools)
        checkpointer, kind = await stack.enter_async_context(
            open_checkpointer(os.getenv("DATABASE_URL"))
        )
        models = ModelConfig.from_env()
        callbacks = setup_tracing()
        yield Runtime(
            agent=GraphAgent(models.clients(), tools, checkpointer=checkpointer),
            keys=KeyStore.from_env(),
            limiter=RateLimiter.from_env(),
            callbacks=callbacks,
            info={
                "checkpointer": kind,
                "mcp": "http" if mcp_url else "stdio",
                "models": models.describe(),
                "tracing": bool(callbacks),
            },
        )


# ------------------------------------------------------------------ schemas


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Pass the thread_id of an earlier answer to ask a follow-up question.
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class AskResponse(BaseModel):
    request_id: str
    thread_id: str
    status: str
    answer: AgentAnswer
    steps: int
    repairs_used: int
    latency_ms: float
    cost_usd: float | None


# ------------------------------------------------------------------ app


def create_app(runtime_factory=open_runtime) -> FastAPI:
    """`runtime_factory` is an async context manager yielding a Runtime (tests pass a fake)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with runtime_factory() as runtime:
            app.state.runtime = runtime
            yield

    app = FastAPI(title="Analyst Agent", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_and_log(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if 0 < len(incoming) <= 64 else uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.user = None
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log.info(json.dumps({
            "event": "request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": round((time.perf_counter() - start) * 1000, 1),
            "user": request.state.user,
        }))
        return response

    def runtime_of(request: Request) -> Runtime:
        return request.app.state.runtime

    def authenticate(
        request: Request,
        response: Response,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> str:
        runtime = runtime_of(request)
        if not runtime.keys.configured:
            raise HTTPException(503, "No API keys are configured on this server.")
        key = x_api_key
        if not key and authorization and authorization.lower().startswith("bearer "):
            key = authorization[7:].strip()
        user = runtime.keys.identify(key)
        if user is None:
            raise HTTPException(401, "Missing or invalid API key.",
                                headers={"WWW-Authenticate": "Bearer"})
        request.state.user = user
        wait = runtime.limiter.check(user)
        if wait is not None:
            raise HTTPException(429, "Rate limit reached.", headers={"Retry-After": str(wait)})
        return user

    @app.get("/health")
    def health() -> dict:
        """Liveness: the process answers."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> dict:
        """Readiness: the MCP tools and the checkpointer are open."""
        return {"status": "ready", **runtime_of(request).info}

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request, user: str = Depends(authenticate)):
        runtime = runtime_of(request)
        # Threads are private to their key: the stored id is prefixed with the key's name.
        public_thread = body.thread_id or uuid.uuid4().hex
        result = await runtime.agent.arun(
            body.question, thread_id=f"{user}:{public_thread}", callbacks=runtime.callbacks
        )
        log.info(json.dumps({
            "event": "answer",
            "request_id": request.state.request_id,
            "status": result.status,
            "llm_calls": result.steps,
            "repairs": result.repairs_used,
            "cost_usd": result.cost_usd,
            "ms": round(result.latency_ms, 1),
            "error": result.error,  # full error in the log only
        }))
        if result.answer is None:
            # Provider and stack details stay in the log; the client gets the status and an id.
            detail = {
                "status": result.status,
                "error": PUBLIC_ERRORS.get(result.status, "No answer could be produced."),
                "request_id": request.state.request_id,
            }
            raise HTTPException(FAILURE_CODES.get(result.status, 422), detail)
        return AskResponse(
            request_id=request.state.request_id,
            thread_id=public_thread,
            status=result.status,
            answer=result.answer,
            steps=result.steps,
            repairs_used=result.repairs_used,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
        )

    return app


app = create_app()
