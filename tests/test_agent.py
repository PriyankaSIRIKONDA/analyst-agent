"""Agent loop tests, using FakeLLM so no network or API key is needed."""

import json

from analyst_agent.agent import (
    ANSWER_RULES,
    FORCE_FINAL_ANSWER,
    LAST_STEP_MESSAGE,
    LAST_STEP_REFUSAL,
    MAX_TRACE_ERROR_CHARS,
    Agent,
    AgentAnswer,
    build_system_prompt,
    summarise_output,
)
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


def test_last_step_forces_final_answer_and_rescues_the_answer(toolbox):
    script = [tool_turn(("list_tables", {})) for _ in range(7)] + [tool_turn(FINAL)]
    llm = FakeLLM(script)
    result = Agent(llm, toolbox, max_steps=8).run("q")

    assert result.status == "ok"
    assert result.steps == 8
    assert llm.tool_choices[:7] == [None] * 7
    assert llm.tool_choices[7] == FORCE_FINAL_ANSWER
    assert llm.calls[7][-1] == {"role": "user", "content": LAST_STEP_MESSAGE}


def test_other_tools_are_refused_on_the_last_step(toolbox):
    llm = FakeLLM([tool_turn(("list_tables", {})) for _ in range(8)])
    result = Agent(llm, toolbox, max_steps=8).run("q")

    assert result.status == "max_steps"
    assert result.tool_calls[-1]["result"] == {"ok": False, "error": LAST_STEP_REFUSAL}
    assert result.tool_calls[-2]["result"] == {"ok": True}


def test_v0_loop_never_forces_the_last_step(toolbox):
    llm = FakeLLM([tool_turn(("list_tables", {})) for _ in range(8)])
    result = Agent(llm, toolbox, max_steps=8, finish_on_last_step=False).run("q")

    assert result.status == "max_steps"
    assert llm.tool_choices == [None] * 8
    assert all(m.get("content") != LAST_STEP_MESSAGE for m in llm.calls[-1])
    assert result.tool_calls[-1]["result"] == {"ok": True}


def test_tool_results_are_summarised_in_the_trace(toolbox):
    llm = FakeLLM([
        tool_turn(("run_sql", {"sql": "SELECT * FROM itemz"})),
        tool_turn(("run_sql", {"sql": GOOD_SQL})),
        tool_turn(FINAL),
    ])
    result = Agent(llm, toolbox).run("q")
    failed, worked, final = (c["result"] for c in result.tool_calls)

    assert failed["ok"] is False and "no such table: itemz" in failed["error"]
    assert worked == {"ok": True, "row_count": 1, "truncated": False}
    assert final == {"ok": True}


def test_trace_errors_are_cut_to_a_fixed_length():
    summary = summarise_output("run_sql", {"ok": False, "error": "x" * 1000})
    assert len(summary["error"]) == MAX_TRACE_ERROR_CHARS
    docs_summary = summarise_output("search_docs", {"ok": True, "results": [1, 2]})
    assert docs_summary == {"ok": True, "hits": 2}


def test_answer_rules_are_in_the_prompt_by_default():
    for has_docs in (True, False):
        with_rules = build_system_prompt(has_docs)
        v0 = build_system_prompt(has_docs, answer_rules=False)
        assert with_rules == v0 + "\n" + ANSWER_RULES
        assert "only the columns the question asks for" in with_rules
        assert "LIMIT 1" in with_rules and "100" in with_rules
        assert v0.endswith("answer only what was asked.")  # the prompt of the v0 runs


def test_agent_sends_the_prompt_it_was_given(toolbox):
    llm_new, llm_v0 = FakeLLM([tool_turn(FINAL)]), FakeLLM([tool_turn(FINAL)])
    Agent(llm_new, toolbox).run("q")
    Agent(llm_v0, toolbox, answer_rules=False).run("q")

    assert ANSWER_RULES in llm_new.calls[0][0]["content"]
    assert ANSWER_RULES not in llm_v0.calls[0][0]["content"]
