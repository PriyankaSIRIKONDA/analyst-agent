"""FastAPI app: POST /ask and GET /health.

Run with:  uvicorn analyst_agent.api:app --reload
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from analyst_agent.agent import Agent, AgentAnswer
from analyst_agent.db import Database
from analyst_agent.llm import LiteLLMClient
from analyst_agent.tools import DocsIndex, ToolBox

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

app = FastAPI(title="Analyst Agent")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    status: str
    answer: AgentAnswer
    steps: int
    repairs_used: int
    latency_ms: float


def build_agent() -> Agent:
    """Create the agent from environment settings (DB_PATH, DOCS_PATH, MODEL)."""
    db = Database(ROOT / os.getenv("DB_PATH", "data/demo.db"))
    docs = DocsIndex.from_file(ROOT / os.getenv("DOCS_PATH", "data/demo_docs.md"))
    return Agent(LiteLLMClient(), ToolBox(db, docs))


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer one question with the agent."""
    result = build_agent().run(request.question)
    if result.answer is None:
        detail = result.error or f"No answer (status: {result.status})"
        raise HTTPException(status_code=502, detail=detail)
    return AskResponse(
        status=result.status,
        answer=result.answer,
        steps=result.steps,
        repairs_used=result.repairs_used,
        latency_ms=result.latency_ms,
    )
