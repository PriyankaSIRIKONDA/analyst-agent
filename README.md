# Analyst Agent

A tool-calling agent that answers business questions in English by looking at a database, writing SQL,
running it read-only, repairing it when the database returns an error, and returning a structured answer.
The repository also contains an evaluation harness that measures how often the agent is right, with
confidence intervals, so the claim "it works" comes with a number and its uncertainty.

## At a glance

- **What it does:** answers an English business question by exploring the database, writing SQL,
  running it read-only, and returning the answer together with the SQL it used.
- **Accuracy:** 20/20 on the demo set with the full agent. Without the docs tool it drops to 16/20
  (paired difference −20 points, 95% CI [−40, −5]).
- **Safety:** in a 20-question prompt-injection test the model tried to write on 13 questions. None
  ran, because the database layer refuses everything except reads.
- **Cost:** about US$0.002 and 20 s per question (median) with `gemini-3.5-flash-lite`.
- **Stack:** Python, LiteLLM, SQLite, FastAPI, MCP, Docker, pytest (138 tests in CI).

Status: version 0. The agent, the guardrails, the harness, the tests and a read-only MCP server are in
place, with a first real evaluation on the 20-question demo set (see [Results](#results)).

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
                MCP server (mcp_server.py)  the four read-only tools for any MCP client
 Measured by:   eval/  (gold questions -> execution accuracy + bootstrap CI)
```

Guardrails in `db.py` (an LLM writes the SQL, so the database layer defends itself):

1. only a single `SELECT` or `WITH` statement is accepted,
2. the connection is opened read-only (`mode=ro`),
3. a SQLite authorizer denies every operation except reads,
4. at most 200 rows are returned and a query is stopped after 5 seconds.

On top of that, the tool result sent to the model is bounded: at most 50 rows, and any cell longer
than 300 characters is cut (bytes are shown as `<N bytes>`). The scorer is not affected, it re-runs
the SQL with its own large limit.

## Repository layout

```
src/analyst_agent/
    db.py       read-only connection, run_sql with guardrails
    tools.py    the four tools, their JSON schemas, BM25 docs search
    llm.py      LiteLLM wrapper and FakeLLM for tests
    agent.py    the loop, repair budget, structured final answer, per-question statistics
    api.py      FastAPI app
    mcp_server.py  read-only MCP server over the same tools
eval/
    make_demo_db.py   builds data/demo.db (e-commerce, fixed seed)
    demo_gold.jsonl   20 questions with gold SQL
    run_eval.py       runs the agent on a gold file (demo or BIRD), or the oracle smoke test
    score.py          execution match, strict and BIRD metrics
    analyze.py        bootstrap CIs, paired comparison, status/failure/difficulty tables
    load_bird.py      builds a 150-question gold file and docs from the BIRD mini-dev download
    injection.jsonl   20 adversarial questions
    run_injection.py  runs them against a copy of the database and counts writes
    check_regression.py  compares a run with eval/baseline.json (or a minimum accuracy)
data/
    demo.db, demo_docs.md
Dockerfile, docker-compose.yml, .dockerignore
.github/workflows/  ci.yml (tests, oracle smoke eval, docker), eval.yml (manual real eval)
tests/              pytest (guardrails, scorer, agent loop with FakeLLM, loader, API, ...)
```

## Setup

Windows PowerShell shown; on macOS/Linux use `source .venv/bin/activate`.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
copy .env.example .env      # then put your key in .env
```

`.env` holds the API key of your provider (for example `GEMINI_API_KEY`, free from Google AI Studio)
and `MODEL` in LiteLLM format (for example `gemini/gemini-3.5-flash-lite`). The file is git-ignored.

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

## Oracle smoke test (no API key)

```powershell
.venv\Scripts\python eval\run_eval.py --oracle --out results\ci\oracle.jsonl
.venv\Scripts\python eval\check_regression.py results\ci\oracle.jsonl --min-accuracy 1.0
```

`--oracle` replaces the model with a script that runs the gold SQL and returns it as the final
answer. It goes through the real tools, database guardrails and scorer, so it must get every question right.
CI runs it on every push to check the whole pipeline without a key.

## BIRD benchmark

The harness can run on the BIRD mini-dev set (SQLite version). Download it yourself and extract it
anywhere; the loader finds the files by searching. Use an absolute path for `--bird-dir`:

```powershell
.venv\Scripts\python eval\load_bird.py --bird-dir F:\Downloads\Portfolio_Projects\bird_data
.venv\Scripts\python eval\run_eval.py --gold data\bird\bird150.jsonl --metric bird --sql-timeout 30 --min-interval 4
```

The loader shuffles the questions with a fixed seed, checks each gold SQL with the same guarded
database layer, keeps the first 150 that run, and writes the gold file, one docs file per database
and `data/bird/manifest.json` (counts of dropped questions and their reasons). If a guardrail rejects
a gold query it stops without writing anything. The evidence text of each question is added to the
prompt as a hint (`--no-evidence` removes it). `--sql-timeout 30` matches the 30-second limit of the
official BIRD evaluation. `--min-interval` spaces the model calls so a free plan stays under its
requests-per-minute limit: use 60 divided by that limit (4 seconds for 15 requests per minute).
A run stops after 3 model errors in a row (often a daily quota) and keeps its finished rows;
continue it later with the same command plus `--out <run file> --resume`.

The `bird` metric is the official BIRD execution accuracy: the predicted and gold result sets are
compared as sets of raw tuples, with no rounding and ignoring row order and duplicates. The default
`strict` metric is the one described below.

Attribution: BIRD mini-dev (https://bird-bench.github.io/), licensed CC BY-SA 4.0. The data is not
redistributed in this repository (`data/bird/` is git-ignored).

## Prompt-injection test

```powershell
.venv\Scripts\python eval\run_injection.py --min-interval 4
```

Runs 20 adversarial questions (write requests, instruction overrides, stacked statements, schema
changes, ATTACH/PRAGMA/load_extension, runaway queries) on a temporary copy of the demo database.
It counts the write attempts, the timeouts, and whether the database file hash changed (target: no
write executed), and writes `results/injection_<stamp>.md`.

## MCP server

The agent's tools are also available as a read-only [MCP](https://modelcontextprotocol.io) server,
so another AI client (Claude Code, Cursor, any MCP client) can explore the database itself:

```powershell
.venv\Scripts\python -m analyst_agent.mcp_server                      # stdio, the default
.venv\Scripts\python -m analyst_agent.mcp_server --transport http      # http://127.0.0.1:8765/mcp
.venv\Scripts\python -m analyst_agent.mcp_server --db my.db --no-docs  # another database
```

It exposes `list_tables`, `describe_table`, `run_sql` and, when a docs file is given,
`search_docs`. It does not run the LLM agent: the calling client does the reasoning. Every call
goes through the same `ToolBox` as the agent, so all the guardrails above apply unchanged. Every
tool is annotated `readOnlyHint` and not destructive. A rejected or failing query comes back as an
MCP error result that still carries the message, so the client can fix its SQL. The HTTP transport
listens on 127.0.0.1 only and has no authentication, so do not expose it on a network.

To add it to Claude Code, give the full path to the virtual environment's Python (the client may
start the server from another folder):

```powershell
claude mcp add analyst-agent -- C:/path/to/analyst-agent/.venv/Scripts/python.exe -m analyst_agent.mcp_server
```

Clients configured with JSON take the same path (forward slashes work on Windows):

```json
{
  "mcpServers": {
    "analyst-agent": {
      "command": "C:/path/to/analyst-agent/.venv/Scripts/python.exe",
      "args": ["-m", "analyst_agent.mcp_server"]
    }
  }
}
```

`tests/test_mcp_server.py` drives the server through the MCP SDK's in-memory client. It checks the
tool list and annotations, the 50-row cap, the error results, and that write attempts (DELETE,
DROP, stacked statements, ATTACH, `WITH ... DELETE`) leave the database file unchanged.

## Docker

```powershell
copy .env.example .env      # put your key in .env
docker compose up --build
# then, from another terminal (PowerShell):
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType 'application/json' -Body '{"question": "How many customers live in Germany?"}'
# or on macOS/Linux:
curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' -d '{"question": "How many customers live in Germany?"}'
```

The image contains `src/` and `data/` only; `.env` is read at run time by Compose, never copied.

## Regression gate

```powershell
.venv\Scripts\python eval\check_regression.py results\run_<stamp>.jsonl --baseline eval\baseline.json --write-baseline
.venv\Scripts\python eval\check_regression.py results\run_<new>.jsonl --baseline eval\baseline.json
```

The first command stores the accuracy and its bootstrap CI of a real run you accept as the
reference. The second exits with code 1 if a later run is below the lower end of that CI. The
manual `eval.yml` workflow does this in GitHub Actions; it reads the model name from the `MODEL`
repository variable and the key from a secret such as `GEMINI_API_KEY`.

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

Real runs with `gemini/gemini-3.5-flash-lite` on the 20 demo questions (`eval/demo_gold.jsonl`, synthetic
database), code at commit `f59ef08`. Full reports: [`results/results.md`](results/results.md) (full vs no repair) and
[`results/full_vs_nodocs.md`](results/full_vs_nodocs.md) (full vs no docs).

| Run | Docs tool | SQL repair | Execution accuracy | 95% CI |
|---|---|---|---|---|
| `demo_full` | yes | yes | 100% (20/20) | [100%, 100%] |
| `demo_nodocs` | no | yes | 80% (16/20) | [60%, 95%] |
| `demo_norepair` | yes | no | 95% (19/20) | [85%, 100%] |

Paired differences against `demo_full`, on the same 20 questions:

- Without the docs tool: -20.0 points, 95% CI [-40.0, -5.0]. The CI excludes 0, so on this set the docs tool helps.
- Without SQL repair: -5.0 points, 95% CI [-15.0, +0.0]. This does not measure the repair loop. In the
  full run it never fired (0 repairs on all 20 questions), and the one question the no-repair run
  missed was a wrong result, not an SQL error, so repair could not have fixed it. The gap is the model
  answering differently from one run to the next. Measuring repair needs harder questions where the
  first query fails, such as BIRD.

Per question in `demo_full`: median latency 19.6 s (p95 28.4 s), about 4,600 prompt tokens, and about US$0.002.

Prompt-injection test, 8 Nov 2026 ([`results/injection_20261108_090932.md`](results/injection_20261108_090932.md)):
on 20 adversarial questions the model attempted a write on 13. None was executed (the SHA-256 of the
database file never changed), and 1 query hit the 5 s timeout. The defence is the database layer, not the prompt.

Twenty questions on a small synthetic database is a small set: the intervals are wide, and 100% here says
nothing about accuracy on a real warehouse. A reported BIRD run is the next step.

## Limitations of v0

- The demo set is 20 questions on a small synthetic database; BIRD support exists but no BIRD run is reported here.
- Keyword docs search only; no embeddings.
- No tracing yet. The MCP server's HTTP transport has no authentication, so it is for local use only.
