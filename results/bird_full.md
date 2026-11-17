# Results

## Run configuration

- `bird_full`: model=gemini/gemini-3.5-flash-lite, use_docs=True, use_repair=True, use_evidence=True, metric=bird, max_steps=8, sql_timeout=30.0, gold=bird150.jsonl, git_commit=f59ef08 (96 rows) + e72132e (52 rows)

## Questions left out

- `bird_full`: left out with --exclude: bird_701
- `bird_full`: duplicate question counted once: bird_138

## Execution accuracy (95% bootstrap CI, 1000 resamples)

| Run | Model | Docs | Repair | N | Accuracy | 95% CI |
|---|---|---|---|---|---|---|
| bird_full | gemini/gemini-3.5-flash-lite | True | True | 148 | 54.1% | [45.9%, 62.2%] |

## Latency, tokens and cost per question

| Run | Median latency (ms) | p95 latency (ms) | Prompt tokens | Completion tokens | Cost (USD) | Tool calls | Repairs |
|---|---|---|---|---|---|---|---|
| bird_full | 27226 | 51284 | 12707 | 332 | 0.00415 | 7.0 | 0.04 |

## Status of each run

| Run | ok | max_steps | sql_error | llm_error | no_answer |
|---|---|---|---|---|---|
| bird_full | 119 | 29 | 0 | 0 | 0 |

## Failure reasons (wrong answers only)

| Run | no_sql | pred_sql_error | result_mismatch |
|---|---|---|---|
| bird_full | 29 | 0 | 39 |

## Accuracy by difficulty

| Run | Difficulty | N | Accuracy |
|---|---|---|---|
| bird_full | challenging | 36 | 47.2% |
| bird_full | moderate | 72 | 44.4% |
| bird_full | simple | 40 | 77.5% |
