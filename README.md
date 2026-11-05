# Analyst Agent

A tool-calling agent that answers business questions in English by looking at a database, writing SQL,
running it read-only, repairing it when the database returns an error, and returning a structured answer.
The repository also contains an evaluation harness that measures how often the agent is right, with
confidence intervals, so the claim "it works" comes with a number and its uncertainty.

Status: version 0. The agent, the guardrails, the harness and the tests are in place. No real
evaluation has been run yet (see [Results](#results)).

## How it works

1. `list_tables` and `describe_table` show which tables and columns exist.
2. `search_docs` looks up short written docs about the tables (keyword search with BM25, no embeddings in v0).
3. `run_sql` runs one SELECT query against the database.
4. If the database returns an error, the error goes back to the model, which repairs the query
   (at most 3 repairs).
5. `final_answer` returns `{answer, sql, tables_used}`, validated with pydantic.

The loop stops after 8 model calls at most.

## Architecture

```
             question
                |
         +------v-------+   tool calls    +------------------------------+
         |  Agent loop  | --------------> | list_tables / describe_table |
         |  (agent.py)  | <-------------- | search_docs  (BM25 on docs)  |
         |              |    results      | run_sql (read-only, row limit|
         +------+-------+                 |          and 5 s timeout)    |
                |                         +--------------+---------------+
         +------v-------+                                |
         |   llm.py     |                           SQLite (read-only)
         |  LiteLLM or  |
         |  FakeLLM     |
         +--------------+

 Entry points:  FastAPI (api.py)  POST /ask, GET /health
 Measured by:   eval/  (gold questions -> execution accuracy + bootstrap CI)
```

Guardrails in `db.py` (an LLM writes the SQL, so the database layer defends itself):

1. only a single `SELECT` or `WITH` statement is accepted,
2. the connection is opened read-only (`mode=ro`),
3. a SQLite authorizer denies every operation except reads,
4. at most 200 rows are returned and a query is stopped after 5 seconds.

## Repository layout

```
src/analyst_agent/
    db.py       read-only connection, run_sql with guardrails
    tools.py    the four tools, their JSON schemas, BM25 docs search
    llm.py      LiteLLM wrapper and FakeLLM for tests
    agent.py    the loop, repair budget, structured final answer, per-question statistics
    api.py      FastAPI app
eval/
    make_demo_db.py   builds data/demo.db (e-commerce, fixed seed)
    demo_gold.jsonl   20 questions with gold SQL
    run_eval.py       runs the agent on a gold file, writes results/run_<timestamp>.jsonl
    score.py          execution match
    analyze.py        bootstrap CIs, paired comparison, results/results.md
data/
    demo.db, demo_docs.md
tests/              pytest (guardrails, scorer, agent loop with FakeLLM, bootstrap, API)
```

## Setup

Windows PowerShell shown; on macOS/Linux use `source .venv/bin/activate`.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
copy .env.example .env      # then put your key in .env
```

`.env` holds `MISTRAL_API_KEY` (or `GEMINI_API_KEY`) and `MODEL` (LiteLLM format, default
`mistral/mistral-small-latest`). The file is git-ignored.

## Run

```powershell
# 1. Build the demo database (already included, this regenerates it identically)
.venv\Scripts\python eval\make_demo_db.py

# 2. Run the tests (no API key needed)
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m ruff check .

# 3. Evaluate the agent (needs your API key in .env)
.venv\Scripts\python eval\run_eval.py                  # full agent
.venv\Scripts\python eval\run_eval.py --no-repair      # ablation: repair loop off
.venv\Scripts\python eval\run_eval.py --no-docs        # ablation: search_docs off

# 4. Summarise one run, or compare two runs on the same questions
.venv\Scripts\python eval\analyze.py results\run_<stamp>.jsonl
.venv\Scripts\python eval\analyze.py results\run_<full>.jsonl results\run_<ablation>.jsonl

# 5. Serve the API
.venv\Scripts\python -m uvicorn analyst_agent.api:app --reload
# then:  POST http://127.0.0.1:8000/ask   {"question": "How many customers live in Germany?"}
```

## Evaluation method

- Metric: execution accuracy. The gold SQL and the agent's SQL are both run; the answer is correct if the
  result sets match. Row order is ignored unless the gold SQL has `ORDER BY`; floats are rounded to 4 decimals;
  column names are ignored.
- Uncertainty: 1,000 bootstrap resamples of the questions give a 95% confidence interval (fixed seed).
- Paired comparison: two runs on the same questions are resampled together, so the CI on the difference
  accounts for question difficulty.
- Also reported: median and p95 latency, mean tokens and cost per question (when LiteLLM can price the model),
  mean tool calls and repairs.

## Results

Results will be filled in after the first real evaluation run. This section intentionally contains no numbers
until `results/results.md` has been produced from real runs.

## Limitations of v0

- Only a 20-question demo set on a small synthetic database; a public benchmark comes later.
- Keyword docs search only; no embeddings.
- No tracing, no MCP server, no Docker image yet.
