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

Results will be filled in after the first real evaluation run. This section intentionally contains no numbers
until `results/results.md` has been produced from real runs.

## Limitations of v0

- The demo set is 20 questions on a small synthetic database; BIRD support exists but no BIRD run is reported here.
- Keyword docs search only; no embeddings.
- No tracing and no MCP server.
