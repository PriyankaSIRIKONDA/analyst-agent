"""The LangGraph engine with scripted models (FakeLLM) and the real MCP server, in process."""

import asyncio

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from analyst_agent.graph import (
    OFF_TOPIC_ANSWER,
    GraphAgent,
    GraphLimits,
    dry_run_sql,
    rank_tables,
    turn_input,
)
from analyst_agent.llm import FakeLLM, LLMResponse, tool_turn
from analyst_agent.mcp_tools import MCPTools


def route(value="sql"):
    return tool_turn(("route", {"route": value}))


def write(sql, tables=("items",)):
    return tool_turn(("write_sql", {"sql": sql, "tables_used": list(tables)}))


def answer(text):
    return tool_turn(("final_answer", {"answer": text}))


def fake(router=(), writer=(), explainer=()):
    return {
        "router": FakeLLM(list(router)),
        "sql_writer": FakeLLM(list(writer)),
        "explainer": FakeLLM(list(explainer)),
    }


def ask(db, docs, llms, questions, limits=None, checkpointer=None, thread="t"):
    """Run the questions one after another in one thread; return the results."""

    async def go():
        async with MCPTools.in_process(db.path, docs) as tools:
            agent = GraphAgent(llms, tools, limits=limits, checkpointer=checkpointer)
            return [await agent.arun(q, thread_id=thread) for q in questions]

    return asyncio.run(go())


def test_happy_path_answers_with_sql_and_tables(db, docs_file):
    llms = fake([route()], [write("SELECT COUNT(*) AS n FROM items")], [answer("20 items.")])
    (result,) = ask(db, docs_file, llms, ["How many items?"])
    assert result.status == "ok"
    assert result.answer.answer == "20 items."
    assert result.answer.sql == "SELECT COUNT(*) AS n FROM items"
    assert result.answer.tables_used == ["items"]
    assert result.steps == 3  # model calls: router, writer, explainer
    assert result.route == "sql"
    names = [c["name"] for c in result.tool_calls]
    assert names == ["list_tables", "describe_table", "search_docs", "run_sql", "run_sql"]
    # the writer saw the schema and the docs; the explainer saw the rows
    writer_prompt = llms["sql_writer"].calls[0][1]["content"]
    assert "TABLE items" in writer_prompt and "is in euros" in writer_prompt
    assert "[[20]]" in llms["explainer"].calls[0][1]["content"]


def test_models_are_forced_to_call_their_tool(db, docs_file):
    llms = fake([route()], [write("SELECT 1")], [answer("1")])
    ask(db, docs_file, llms, ["q"])
    assert llms["router"].tool_choices[0]["function"]["name"] == "route"
    assert llms["sql_writer"].tool_choices[0]["function"]["name"] == "write_sql"
    assert llms["explainer"].tool_choices[0]["function"]["name"] == "final_answer"


def test_off_topic_question_skips_the_database(db, docs_file):
    llms = fake([route("off_topic")])
    (result,) = ask(db, docs_file, llms, ["Write me a poem"])
    assert result.status == "off_topic"
    assert result.answer.answer == OFF_TOPIC_ANSWER and result.answer.sql == ""
    assert result.tool_calls == [] and llms["sql_writer"].calls == []


def test_checker_sends_a_bad_query_back_to_the_writer(db, docs_file):
    llms = fake(
        [route()],
        [write("SELECT nope FROM items"), write("SELECT COUNT(*) FROM items")],
        [answer("20")],
    )
    (result,) = ask(db, docs_file, llms, ["How many?"])
    assert result.status == "ok" and result.repairs_used == 1
    retry_prompt = llms["sql_writer"].calls[1][1]["content"]
    assert "SELECT nope FROM items" in retry_prompt and "no such column" in retry_prompt


def test_write_queries_never_reach_the_database(db, docs_file):
    llms = fake([route()], [write("DELETE FROM items"), write("SELECT COUNT(*) FROM items")],
                [answer("20")])
    (result,) = ask(db, docs_file, llms, ["Delete everything"])
    assert result.repairs_used == 1
    # the static check rejected it before any MCP call with that SQL
    assert all("DELETE" not in str(c["arguments"]) for c in result.tool_calls)
    assert db.run_sql("SELECT COUNT(*) FROM items")["rows"] == [[20]]


def test_repair_budget_stops_the_run(db, docs_file):
    bad = [write("SELECT nope FROM items")] * 3
    llms = fake([route()], bad)
    (result,) = ask(db, docs_file, llms, ["?"], limits=GraphLimits(max_repairs=2))
    assert result.status == "sql_error" and result.answer is None
    assert result.repairs_used == 2 and "no such column" in result.error


def test_step_limit_stops_the_run(db, docs_file):
    bad = [write("SELECT nope FROM items")] * 10
    llms = fake([route()], bad)
    limits = GraphLimits(max_steps=5, max_repairs=10)
    (result,) = ask(db, docs_file, llms, ["?"], limits=limits)
    assert result.status == "max_steps" and "5 steps" in result.error


def test_failing_model_is_retried_then_reported(db, docs_file):
    llms = fake([ConnectionError("down"), ConnectionError("down")])
    limits = GraphLimits(retry_attempts=2)
    (result,) = ask(db, docs_file, llms, ["?"], limits=limits)
    assert result.status == "llm_error" and "router" in result.error
    assert len(llms["router"].calls) == 2  # first try + one retry


def test_transient_failure_recovers_on_retry(db, docs_file):
    llms = fake([ConnectionError("blip"), route()], [write("SELECT 1 AS x")], [answer("1")])
    (result,) = ask(db, docs_file, llms, ["?"])
    assert result.status == "ok"


def test_node_timeout_is_reported(db, docs_file):
    class Slow:
        async def acomplete(self, *args, **kwargs):
            await asyncio.sleep(5)

    limits = GraphLimits(retry_attempts=1, timeouts={"router": 0.1})
    (result,) = ask(db, docs_file, {"router": Slow()}, ["?"], limits=limits)
    assert result.status == "timeout" and "router" in result.error


def test_follow_up_question_sees_the_thread_history(db, docs_file):
    llms = fake(
        [route(), route()],
        [write("SELECT COUNT(*) FROM items"), write("SELECT SUM(price) FROM items")],
        [answer("20 items."), answer("315 euros.")],
    )
    first, second = ask(db, docs_file, llms, ["How many items?", "And their total price?"],
                        checkpointer=InMemorySaver())
    assert first.status == second.status == "ok"
    prompt = llms["sql_writer"].calls[1][1]["content"]
    assert "Earlier in this conversation" in prompt and "How many items?" in prompt
    assert "SELECT COUNT(*) FROM items" in prompt
    # per-turn fields were reset: the second turn did not inherit repairs or tool calls
    assert second.repairs_used == 0 and second.tool_calls[0]["name"] == "list_tables"


def test_threads_do_not_share_history(db, docs_file):
    saver = InMemorySaver()
    ask(db, docs_file, fake([route()], [write("SELECT 1")], [answer("one")]), ["first"],
        checkpointer=saver, thread="a")
    llms = fake([route()], [write("SELECT 2")], [answer("two")])
    ask(db, docs_file, llms, ["second"], checkpointer=saver, thread="b")
    assert "Earlier in this conversation" not in llms["sql_writer"].calls[0][1]["content"]


def test_usage_is_summed_over_model_calls(db, docs_file):
    def priced(turn):
        return LLMResponse(tool_calls=turn.tool_calls, prompt_tokens=10, completion_tokens=2,
                           cost_usd=0.001)

    llms = fake([priced(route())], [priced(write("SELECT 1"))], [priced(answer("1"))])
    (result,) = ask(db, docs_file, llms, ["?"])
    assert result.prompt_tokens == 30 and result.completion_tokens == 6
    assert result.cost_usd == pytest.approx(0.003)


def test_wrong_tables_used_are_taken_from_the_sql(db, docs_file):
    llms = fake([route()], [write("SELECT COUNT(*) FROM items", tables=["made_up"])], [answer("x")])
    (result,) = ask(db, docs_file, llms, ["?"])
    assert result.answer.tables_used == ["items"]


def test_helpers():
    assert dry_run_sql("SELECT 1") == "SELECT * FROM (SELECT 1) AS checked LIMIT 0"
    assert rank_tables("orders per customer", ["a", "orders", "customers"], 2) == [
        "orders", "customers"
    ]
    assert rank_tables("x", ["a", "b"], 5) == ["a", "b"]
    assert turn_input("q")["status"] == "running" and "history" not in turn_input("q")
