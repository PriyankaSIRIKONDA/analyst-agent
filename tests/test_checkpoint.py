"""The checkpointer: memory by default, Postgres when TEST_DATABASE_URL is set (CI sets it)."""

import asyncio
import os
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from analyst_agent.checkpoint import open_checkpointer
from analyst_agent.graph import GraphAgent
from analyst_agent.llm import FakeLLM, tool_turn
from analyst_agent.mcp_tools import MCPTools


def test_no_url_means_memory():
    async def go():
        async with open_checkpointer(None) as (saver, kind):
            return saver, kind

    saver, kind = asyncio.run(go())
    assert kind == "memory" and isinstance(saver, InMemorySaver)


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="needs TEST_DATABASE_URL")
def test_postgres_thread_survives_a_restart(db, docs_file):
    url = os.environ["TEST_DATABASE_URL"]
    thread = uuid.uuid4().hex

    def llms(sql, text):
        return {
            "router": FakeLLM([tool_turn(("route", {"route": "sql"}))]),
            "sql_writer": FakeLLM([tool_turn(("write_sql", {"sql": sql, "tables_used": []}))]),
            "explainer": FakeLLM([tool_turn(("final_answer", {"answer": text}))]),
        }

    async def one_process(models, question):
        # a fresh pool, saver and graph each time, as after an API restart
        async with open_checkpointer(url) as (saver, kind), \
                MCPTools.in_process(db.path, docs_file) as tools:
            assert kind == "postgres"
            await GraphAgent(models, tools, checkpointer=saver).arun(question, thread_id=thread)

    asyncio.run(one_process(llms("SELECT COUNT(*) FROM items", "20"), "How many items?"))
    second = llms("SELECT 1", "ok")
    asyncio.run(one_process(second, "And now?"))
    assert "How many items?" in second["sql_writer"].calls[0][1]["content"]
