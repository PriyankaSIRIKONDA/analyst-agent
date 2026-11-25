"""The agent loop: call the LLM, run the tools it asks for, feed results back, repeat.

The loop stops when the model calls ``final_answer`` with a valid structured answer,
when it has used ``max_steps`` LLM calls, or when SQL errors exhaust the repair budget.
On the last allowed call the model is offered only ``final_answer`` and told so, because in
the v0 BIRD runs it often spent every step re-checking a query instead of answering.
The system prompt also carries answer-shape rules, added after the error analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from pydantic import BaseModel, ValidationError

from analyst_agent.llm import LLMResponse
from analyst_agent.tools import ToolBox, function_schema

MAX_STEPS = 8
MAX_REPAIRS = 3


class AgentAnswer(BaseModel):
    """The structured final answer. Validated with pydantic."""

    answer: str
    sql: str
    tables_used: list[str]


FINAL_ANSWER_SCHEMA = function_schema(
    "final_answer",
    "Give the final answer to the user. Call this once, when you are done.",
    {
        "answer": {"type": "string", "description": "The answer in one or two sentences."},
        "sql": {"type": "string", "description": "The single SQL query that produced the answer."},
        "tables_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of the tables used by that query.",
        },
    },
    ["answer", "sql", "tables_used"],
)


@dataclass
class AgentResult:
    """Everything we record about one question."""

    question: str
    status: str = "no_answer"  # ok | max_steps | sql_error | llm_error | no_answer
    answer: AgentAnswer | None = None
    steps: int = 0  # number of LLM calls
    # [{"name": ..., "arguments": ..., "result": short summary of what the tool returned}]
    tool_calls: list[dict] = field(default_factory=list)
    repairs_used: int = 0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Plain dict, ready for json.dumps (used for the eval JSONL files)."""
        data = asdict(self)
        data["answer"] = self.answer.model_dump() if self.answer else None
        return data


LAST_STEP_MESSAGE = (
    "This is your last step. Call final_answer now with the best query you have already run. "
    "No other tool is available."
)
LAST_STEP_REFUSAL = "Only final_answer is available on the last step."
# OpenAI-style tool_choice; LiteLLM turns it into Gemini's mode ANY limited to final_answer.
FORCE_FINAL_ANSWER = {"type": "function", "function": {"name": "final_answer"}}

MAX_TRACE_ERROR_CHARS = 300


def summarise_output(name: str, output: dict) -> dict:
    """A short record of a tool result for the run files: enough for error analysis, not the data.

    Errors keep their message (cut to 300 characters); run_sql keeps the row count and whether
    the rows were cut at the limit; search_docs keeps the number of hits.
    """
    summary: dict = {"ok": bool(output.get("ok"))}
    if not summary["ok"]:
        summary["error"] = str(output.get("error", ""))[:MAX_TRACE_ERROR_CHARS]
    elif name == "run_sql":
        summary["row_count"] = len(output.get("rows") or [])
        summary["truncated"] = bool(output.get("truncated"))
    elif name == "search_docs":
        summary["hits"] = len(output.get("results") or [])
    return summary


# Added after the error analysis of the v0 BIRD runs (results/error_analysis.md): 23 of the 56
# questions every configuration got wrong had the right data in the wrong form (extra columns,
# or a list where one answer was asked for), and some ignored a formula given in the hint.
ANSWER_RULES = (
    "Answer shape rules:\n"
    "- Select only the columns the question asks for, in the order it asks for them. "
    "Do not add ids, names, counts or other columns as context.\n"
    "- If the question asks for one thing (which, who, the most, the highest, the lowest), "
    "return only that row, for example with ORDER BY ... LIMIT 1.\n"
    "- If the hint gives a formula or a definition, compute exactly that formula, "
    "including any multiplication by 100, even if a stored column looks similar."
)


def build_system_prompt(has_docs: bool, answer_rules: bool = True) -> str:
    """Instructions for the model. Mentions search_docs only when that tool exists.

    answer_rules=False gives the v0 prompt, used for the paired comparison with the old runs.
    """
    steps = ["1. Call list_tables, then describe_table for the tables you need."]
    if has_docs:
        steps.append("2. Call search_docs to check column meanings and business definitions.")
        steps.append("3. Write one SQLite query and test it with run_sql.")
    else:
        steps.append("2. Write one SQLite query and test it with run_sql.")
    steps.append(
        "If run_sql returns an error, read the message, fix the query and try again."
    )
    steps.append(
        "Finally call final_answer with a short answer, the exact SQL that produced it, "
        "and the tables it uses."
    )
    prompt = (
        "You are a careful data analyst. You answer business questions about a SQLite "
        "database by writing SQL.\n"
        + "\n".join(steps)
        + "\nRules: only read data; never guess values you have not seen in the results; "
        "answer only what was asked."
    )
    return prompt + "\n" + ANSWER_RULES if answer_rules else prompt


class Agent:
    """A tool-calling SQL agent."""

    def __init__(
        self,
        llm,
        toolbox: ToolBox,
        max_steps: int = MAX_STEPS,
        max_repairs: int = MAX_REPAIRS,
        finish_on_last_step: bool = True,
        answer_rules: bool = True,
    ):
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps
        self.max_repairs = max_repairs  # 0 turns the repair loop off (ablation)
        # On the last step, offer only final_answer and say so (False: the v0 behaviour).
        self.finish_on_last_step = finish_on_last_step
        # Add the answer-shape rules to the system prompt (False: the v0 prompt).
        self.answer_rules = answer_rules

    def run(self, question: str) -> AgentResult:
        """Answer one question and return the answer plus all recorded statistics."""
        start = time.perf_counter()
        result = AgentResult(question=question)
        tools = self.toolbox.schemas() + [FINAL_ANSWER_SCHEMA]
        system_prompt = build_system_prompt(self.toolbox.docs is not None, self.answer_rules)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        while result.steps < self.max_steps and result.status == "no_answer":
            last_step = self.finish_on_last_step and result.steps == self.max_steps - 1
            if last_step:
                messages.append({"role": "user", "content": LAST_STEP_MESSAGE})
            try:
                if last_step:  # every tool stays declared, but the call must be final_answer
                    response = self.llm.complete(messages, tools, tool_choice=FORCE_FINAL_ANSWER)
                else:
                    response = self.llm.complete(messages, tools)
            except Exception as exc:  # a failed API call should not crash a whole eval run
                result.status = "llm_error"
                result.error = f"{type(exc).__name__}: {exc}"
                break
            result.steps += 1
            self._add_usage(result, response)
            messages.append(response.assistant_message())

            if not response.tool_calls:
                # The model replied in plain text. Remind it to use final_answer.
                messages.append(
                    {"role": "user", "content": "Please call a tool. Use final_answer when done."}
                )
                continue

            for call in response.tool_calls:
                if call.name == "final_answer":
                    tool_output = self._handle_final_answer(call.arguments, result)
                elif last_step:  # the model asked for a tool it was not offered
                    tool_output = {"ok": False, "error": LAST_STEP_REFUSAL}
                else:
                    tool_output = self.toolbox.call(call.name, call.arguments)
                    self._handle_sql_error(call.name, tool_output, result)
                result.tool_calls.append({
                    "name": call.name,
                    "arguments": call.arguments,
                    "result": summarise_output(call.name, tool_output),
                })
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(tool_output, default=str),
                    }
                )
                if result.status != "no_answer":
                    break  # finished (ok) or stopped (sql_error): ignore remaining calls

        if result.status == "no_answer" and result.steps >= self.max_steps:
            result.status = "max_steps"
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _add_usage(result: AgentResult, response: LLMResponse) -> None:
        result.prompt_tokens += response.prompt_tokens
        result.completion_tokens += response.completion_tokens
        if response.cost_usd is not None:
            result.cost_usd = (result.cost_usd or 0.0) + response.cost_usd

    @staticmethod
    def _handle_final_answer(arguments: dict, result: AgentResult) -> dict:
        """Validate the structured answer. On failure, the error goes back to the model."""
        try:
            result.answer = AgentAnswer.model_validate(arguments)
        except ValidationError as exc:
            return {"ok": False, "error": f"Invalid final_answer: {exc.errors()}"}
        result.status = "ok"
        return {"ok": True}

    def _handle_sql_error(self, tool_name: str, output: dict, result: AgentResult) -> None:
        """Count SQL errors. Each one is a repair attempt until the budget is used up."""
        if tool_name != "run_sql" or output.get("ok"):
            return
        if result.repairs_used < self.max_repairs:
            result.repairs_used += 1
            output["hint"] = "The query failed. Fix it and call run_sql again."
        else:
            result.status = "sql_error"
            result.error = output.get("error")
