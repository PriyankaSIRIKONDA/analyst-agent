"""Agent loop tests, using FakeLLM so no network or API key is needed."""

import json

from analyst_agent.agent import Agent, AgentAnswer
from analyst_agent.llm import FakeLLM, LLMResponse, tool_turn
from analyst_agent.tools import ToolBox

GOOD_SQL = "SELECT COUNT(*) FROM items"
FINAL = (
    "final_answer",
    {"answer": "There are 20 items.", "sql": GOOD_SQL, "tables_used": ["items"]},
)


def test_happy_path(toolbox):
    llm = FakeLLM(
        [
            tool_turn(("list_tables", {})),
            tool_turn(("run_sql", {"sql": GOOD_SQL})),
            tool_turn(FINAL),
        ]
    )
    result = Agent(llm, toolbox).run("How many items?")

    assert result.status == "ok"
    assert isinstance(result.answer, AgentAnswer)
    assert result.answer.sql == GOOD_SQL
    assert result.answer.tables_used == ["items"]
    assert result.steps == 3
    assert result.repairs_used == 0
    assert [c["name"] for c in result.tool_calls] == ["list_tables", "run_sql", "final_answer"]
    assert result.latency_ms > 0


def test_sql_error_is_sent_back_and_repaired(toolbox):
    llm = FakeLLM(
        [
            tool_turn(("run_sql", {"sql": "SELECT COUNT(*) FROM itemz"})),
            tool_turn(("run_sql", {"sql": GOOD_SQL})),
            tool_turn(FINAL),
        ]
    )
    result = Agent(llm, toolbox).run("How many items?")

    assert result.status == "ok"
    assert result.repairs_used == 1
    # The second LLM call must have seen the database error message.
    second_call_messages = llm.calls[1]
    assert "no such table: itemz" in second_call_messages[-1]["content"]


def test_repair_budget_is_limited(toolbox):
    bad = tool_turn(("run_sql", {"sql": "SELECT * FROM nope"}))
    llm = FakeLLM([bad, bad, bad, bad, tool_turn(FINAL)])
    result = Agent(llm, toolbox, max_repairs=3).run("q")

    assert result.status == "sql_error"
    assert result.repairs_used == 3
    assert result.answer is None
    assert result.steps == 4  # stopped at the 4th error, never reached final_answer


def test_no_repair_ablation_stops_at_first_error(toolbox):
    bad = tool_turn(("run_sql", {"sql": "SELECT * FROM nope"}))
    llm = FakeLLM([bad, tool_turn(FINAL)])
    result = Agent(llm, toolbox, max_repairs=0).run("q")

    assert result.status == "sql_error"
    assert result.repairs_used == 0
    assert result.steps == 1


def test_stops_after_max_steps(toolbox):
    llm = FakeLLM([tool_turn(("list_tables", {})) for _ in range(20)])
    result = Agent(llm, toolbox, max_steps=8).run("q")

    assert result.status == "max_steps"
    assert result.steps == 8
    assert result.answer is None


def test_invalid_final_answer_is_rejected_then_corrected(toolbox):
    llm = FakeLLM(
        [
            tool_turn(("final_answer", {"answer": "20"})),  # missing sql and tables_used
            tool_turn(FINAL),
        ]
    )
    result = Agent(llm, toolbox).run("q")

    assert result.status == "ok"
    assert result.steps == 2
    assert "Invalid final_answer" in llm.calls[1][-1]["content"]


def test_plain_text_reply_gets_a_reminder(toolbox):
    llm = FakeLLM([LLMResponse(content="It is 20."), tool_turn(FINAL)])
    result = Agent(llm, toolbox).run("q")

    assert result.status == "ok"
    assert "final_answer" in llm.calls[1][-1]["content"]


def test_usage_is_summed_over_steps(toolbox):
    first = tool_turn(("list_tables", {}))
    first.prompt_tokens, first.completion_tokens, first.cost_usd = 100, 10, 0.25
    second = tool_turn(FINAL)
    second.prompt_tokens, second.completion_tokens, second.cost_usd = 150, 20, 0.5
    result = Agent(FakeLLM([first, second]), toolbox).run("q")

    assert result.prompt_tokens == 250
    assert result.completion_tokens == 30
    assert result.cost_usd == 0.75


def test_llm_failure_is_recorded_not_raised(toolbox):
    result = Agent(FakeLLM([]), toolbox).run("q")  # empty script -> RuntimeError in the LLM
    assert result.status == "llm_error"
    assert "exhausted" in result.error


def test_no_docs_ablation_hides_search_docs_tool(db, toolbox):
    names_with = [s["function"]["name"] for s in toolbox.schemas()]
    names_without = [s["function"]["name"] for s in ToolBox(db, docs=None).schemas()]
    assert "search_docs" in names_with
    assert "search_docs" not in names_without
    assert ToolBox(db, docs=None).call("search_docs", {"query": "x"})["ok"] is False


def test_result_is_json_serialisable(toolbox):
    result = Agent(FakeLLM([tool_turn(FINAL)]), toolbox).run("q")
    assert json.loads(json.dumps(result.to_dict()))["answer"]["sql"] == GOOD_SQL
