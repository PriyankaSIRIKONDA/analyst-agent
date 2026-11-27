# Analyst Agent

A tool-calling agent that answers business questions in English by looking at a database, writing SQL,
running it read-only, repairing it when the database returns an error, and returning a structured answer.
The repository also contains an evaluation harness that measures how often the agent is right, with
confidence intervals, so the claim "it works" comes with a number and its uncertainty.

## At a glance

- **What it does:** answers an English business question by exploring the database, writing SQL,
  running it read-only, and returning the answer together with the SQL it used.
- **Accuracy:** 54.1% (80/148) on BIRD mini-dev questions, 95% CI [45.9%, 62.2%]. On BIRD, removing
  the docs tool or the SQL repair loop makes no clear difference (paired 95% CIs [−3.4, +7.4] and
  [−4.7, +4.7] points). On the 20-question demo set the full agent gets 20/20, and without the docs
  tool it drops to 16/20 (−20 points, 95% CI [−40, −5]).
- **Safety:** in a 20-question prompt-injection test the model tried to write on 13 questions. None
  ran, because the database layer refuses everything except reads.
- **Cost:** with `gemini-3.5-flash-lite`, the median is 27 s per BIRD question (20 s on the demo set), at
  about US$0.004 (US$0.002) at list price. The runs here used the free tier.
- **Stack:** Python, LangGraph, LiteLLM, MCP, SQLite (data), Postgres + pgvector (memory), FastAPI,
  Langfuse, Docker, pytest (188 tests in CI).

Status: v1 in progress. Week 1 adds a LangGraph engine next to the v0 loop (see
[LangGraph engine](#langgraph-engine)): router, SQL writer, SQL checker, executor and explainer, a
Postgres checkpointer for conversations, step/time/retry limits, the MCP server as the only path to
the data, per-node models, and an API with keys and rate limits. The v0 numbers below are for the
v0 loop; the graph has passed the oracle run but not yet a real-model eval.

Version 0: the agent, the guardrails, the harness, the tests and a read-only MCP server are in
place, with real evaluations on 148 BIRD mini-dev questions and on the 20-question demo set
(see [Results](#results)).

## How it works

1. `list_tables` and `describe_table` show which tables and columns exist.
2. `search_docs` looks up short written docs about the tables (keyword search with BM25, no embeddings in v0).
3. `run_sql` runs one SELECT query against the database.
4. If the database returns an error, the error goes back to the model, which repairs the query
   (at most 3 repairs).
5. `final_answer` returns `{answer, sql, tables_used}`, validated with pydantic.

The loop stops after 8 model calls at most. On the 8th call the model is told it is the last one, and
`tool_choice` forces it to call `final_answer`. All tools stay declared, so the history stays
valid. In the v0 BIRD runs (below), the agent often used all 8 calls re-checking a query and never
answered; `--no-last-step-finish` restores that v0 loop. Each tool call in a run file also records a
short summary of its result (ok or the error message, row count), for error analysis.

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

## LangGraph engine

```
 question ─► router ──off_topic──► fixed reply
               │ sql
             schema        list_tables, describe_table, search_docs   (MCP, no LLM)
               │
          ┌► sql_writer    LLM writes one query (forced write_sql tool)
  repair  │    │
  (≤ 3)   ├─ sql_checker   SELECT/WITH check, then a dry run: SELECT * FROM (q) LIMIT 0
          │    │ ok
          └─ executor      run_sql                                    (MCP)
               │ ok
             explainer     LLM turns the rows into one or two sentences
```

- **Data access only through MCP.** `mcp_tools.py` is an MCP client of `mcp_server.py` (a stdio
  subprocess in the API, in-process in tests and evals, or `MCP_URL` for a running server), so the
  guardrails of `db.py` are the only path to the database for the graph too.
- **Limits.** At most 12 node runs per question (`max_steps`, plus LangGraph's `recursion_limit` as a
  backstop), at most 3 repairs, a timeout per node (20–45 s) and a `RetryPolicy` (3 attempts) for
  timeouts, MCP transport errors and provider errors such as rate limits. A node that still fails
  ends the run with `timeout`, `llm_error` or `tool_error` instead of crashing the request.
- **Models per node.** `models.py` reads `MODEL` and, optionally, `MODEL_ROUTER`, `MODEL_SQL_WRITER`,
  `MODEL_EXPLAINER` and `MODEL_FALLBACK` (all LiteLLM names: Gemini, Mistral, Groq, Bedrock, ...).
- **Memory.** State is checkpointed per thread: Postgres when `DATABASE_URL` is set, memory
  otherwise. A follow-up ("and last month?") shows the writer the last 3 questions, SQL and answers.
  In the API, thread ids are private to the API key that created them.
- **Tracing.** With `LANGFUSE_*` set, each run is a Langfuse trace with a span per node, and LiteLLM
  logs every model call with tokens, cost and latency.

Run it on the demo questions and compare it with the v0 loop:

```powershell
.venv\Scripts\python eval\run_eval.py --engine graph --out results\run_graph.jsonl
.venv\Scripts\python eval\analyze.py results\run_graph.jsonl results\demo_full.jsonl
```

## Repository layout

```
src/analyst_agent/
    db.py       read-only connection, run_sql with guardrails
    tools.py    the four tools, their JSON schemas, BM25 docs search
    llm.py      LiteLLM wrapper and FakeLLM for tests
    agent.py    the v0 loop, repair budget, structured final answer, per-question statistics
    graph.py    the LangGraph engine (nodes, edges, limits, GraphAgent)
    mcp_tools.py   MCP client the graph uses for every database call
    models.py   which LiteLLM model each node uses
    checkpoint.py  Postgres (or memory) checkpointer
    security.py API keys (hashed) and per-key rate limits
    observability.py  Langfuse tracing, on when its keys are set
    api.py      FastAPI app: /ask, /health, /ready
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

# 5. Serve the API: make a key, put the printed name:hash in API_KEYS in .env, then start it
.venv\Scripts\python -m analyst_agent.security new-key siri
.venv\Scripts\python -m uvicorn analyst_agent.api:app --reload
# then:  POST http://127.0.0.1:8000/ask  with header X-API-Key: <key>
#        {"question": "How many customers live in Germany?"}
#        and, for a follow-up, {"question": "And in France?", "thread_id": "<from the answer>"}
```

The API refuses questions when no key is configured (set `AUTH_DISABLED=true` only on your own
machine). Each key may ask `RATE_LIMIT_PER_MINUTE` (10) questions a minute and `DAILY_QUERY_CAP`
(200) a day; past that it gets 429 with `Retry-After`. Every response carries `X-Request-ID`, and each
request and answer is logged as one JSON line. Error responses give a status and the request id, never
the provider's error text. `GET /health` is liveness, `GET /ready` shows the checkpointer, MCP
transport and models.

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
continue it later with the same command plus `--out <run file> --resume`. To compare two BIRD
runs, pass both files to `eval/analyze.py`. `--exclude <id>` leaves a question out of both runs, and the
report lists every question left out.

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
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -Headers @{'X-API-Key'='<key>'} -ContentType 'application/json' -Body '{"question": "How many customers live in Germany?"}'
# or on macOS/Linux:
curl -X POST http://localhost:8000/ask -H 'X-API-Key: <key>' -H 'Content-Type: application/json' -d '{"question": "How many customers live in Germany?"}'
```

Compose starts Postgres with pgvector (`pgvector/pgvector:pg17`, port 5432 on localhost only) and
points the API's `DATABASE_URL` at it, so conversations survive restarts. pgvector is enabled now
for the RAG store of week 2. The image contains `src/` and `data/` only and runs as a non-root user;
`.env` is read at run time by Compose, never copied.

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

### BIRD mini-dev, 148 questions

Three runs of `gemini/gemini-3.5-flash-lite` on the BIRD gold file, 6-16 Nov 2026, with
`--metric bird --sql-timeout 30` and the evidence hints on. Full reports:
[`results/bird_full_vs_nodocs.md`](results/bird_full_vs_nodocs.md) and
[`results/bird_full_vs_norepair.md`](results/bird_full_vs_norepair.md). The raw run files are
`results/bird_full.jsonl`, `results/bird_nodocs.jsonl` and `results/bird_norepair.jsonl`.

| Run | Docs tool | SQL repair | Execution accuracy | 95% CI |
|---|---|---|---|---|
| `bird_full` | yes | yes | 54.1% (80/148) | [45.9%, 62.2%] |
| `bird_nodocs` | no | yes | 56.1% (83/148) | [48.0%, 64.9%] |
| `bird_norepair` | yes | no | 54.1% (80/148) | [45.9%, 62.2%] |

Paired differences against `bird_full`, on the same 148 questions:

- Without the docs tool: +2.0 points, 95% CI [-3.4, +7.4].
- Without SQL repair: +0.0 points, 95% CI [-4.7, +4.7]. The tie is a coincidence: 6 questions are
  right only in `bird_full` and 6 only in `bird_norepair`.

Both intervals include 0, so on BIRD neither part changes accuracy measurably. That is not what the
demo set showed, and the run files suggest why:

- Repair almost never triggers. `bird_full` used a repair on 6 of 148 questions, and 2 of those ended
  correct. The 8 questions that ended on a SQL error without repair were wrong in `bird_full` too.
- Each BIRD question comes with an evidence hint, such as "Male refers to gender = 'M'", which carries
  much of what the docs say. This explanation is not tested yet: a run with `--no-evidence`, with and
  without the docs tool, would test it.
- Docs searches use up steps. `bird_full` called `search_docs` 0.71 times per question. Questions that
  ran out of steps had searched 1.17 times on average, against 0.60 for the questions that finished.
  Without the docs tool, 22 questions ran out of steps instead of 29.

Where the errors are:

- By difficulty in `bird_full`: simple 77.5% (40 questions), moderate 44.4% (72), challenging 47.2% (36).
- Of the 68 wrong answers in `bird_full`, 29 are questions where the agent used all 8 steps without
  calling `final_answer`, and 39 returned a different result from the gold query. No final SQL failed to run.
- Most step-limit failures are questions the agent never solved, not answers it forgot to give. I
  scored the agent's queries offline for those 29 questions: the last query it ran was correct for 2,
  and at least one of its queries was correct for 8. So a forced final answer can rescue at most a few
  questions (2 to 8, about 1 to 5 points). That is smaller than what one 148-question run can detect.
- 56 questions are wrong in all three runs and 68 are right in all three. The other 24 change from one
  run to another, so a difference of a few questions between two runs is within the noise. The paired
  intervals show this.

Per question in `bird_full`: median latency 27.2 s (p95 51.3 s), about 12,700 prompt tokens and 7.0 tool
calls. LiteLLM prices this at about US$0.004 per question, but the runs used the free tier, so they cost nothing.

How the runs were done:

- `bird_701` is left out of all three runs (`eval/analyze.py ... --exclude bird_701`). Its gold query takes
  0.35 s on SQLite 3.39 (Windows Python) but more than 5 minutes on SQLite 3.53 (the Linux Python used
  for most runs). SQLite 3.53 runs its `MAX(Reputation)` subquery again for every row of `posts`
  instead of once. I compared the plans of all 718 queries in the runs (the gold queries and every
  query the agent wrote) on both versions: this was the only one whose outcome changed.
- `bird_138` appears twice in the BIRD mini-dev file, so it is counted once. That leaves 148 questions.
- All three runs used the v0 loop, without the forced last step described in [How it works](#how-it-works).
- On 9 Nov, Windows Smart App Control started blocking the compiled libraries in the virtual
  environment, so the runs moved to WSL Ubuntu with the same library versions. In `bird_full`, 96
  questions ran on Windows (commit `f59ef08`) and 52 on WSL (commit `e72132e`, which adds only the MCP
  server and README changes). The two other runs ran entirely on WSL. The agent code is the same in
  both commits. The run files record `e72132e`. On GitHub, that commit is `68c0cb2`: its message was
  rewritten, but the files are identical.
- On the free plan (500 requests a day), each run took several days and was continued with `--resume`.
  Questions that hit a quota error were run again.

### Demo set, 20 questions

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
nothing about accuracy on a real warehouse. The BIRD runs above are the better guide.

## Limitations of v0

- The BIRD results are one run per configuration, with one model, on 148 mini-dev questions, always with
  the evidence hints. Running out of steps (29 of 68 wrong answers) is the largest single failure.
  The forced last step makes the agent always answer, but it is not measured on BIRD yet.
- The demo set is 20 questions on a small synthetic database.
- Keyword docs search only; no embeddings.
- The MCP server's HTTP transport has no authentication, so it is for local use only (the API starts
  its own server over stdio).
- The rate limiter counts in memory, so it holds per API instance; with several instances it moves to
  Postgres or Redis.
- The business data is still SQLite; it moves to Postgres with a read-only role in week 5.
