"""Conversation memory: the LangGraph checkpointer.

With ``DATABASE_URL`` set (``postgresql://...``), checkpoints go to Postgres, so a thread
survives API restarts and is shared by every API instance. Without it, they stay in memory,
which is what the tests and a quick local run use.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import InMemorySaver


@asynccontextmanager
async def open_checkpointer(database_url: str | None, pool_size: int = 10):
    """Yield (checkpointer, kind). The Postgres tables are created on first use."""
    if not database_url:
        yield InMemorySaver(), "memory"
        return

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo=database_url,
        max_size=pool_size,
        # settings AsyncPostgresSaver needs
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    await pool.open(wait=True, timeout=30)
    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        yield saver, "postgres"
    finally:
        await pool.close()
