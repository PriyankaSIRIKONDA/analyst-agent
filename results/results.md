# Results

## Run configuration

- `demo_full`: model=gemini/gemini-3.5-flash-lite, use_docs=True, use_repair=True, use_evidence=True, metric=strict, max_steps=8, sql_timeout=5.0, gold=demo_gold.jsonl, git_commit=f59ef08
- `demo_norepair`: model=gemini/gemini-3.5-flash-lite, use_docs=True, use_repair=False, use_evidence=True, metric=strict, max_steps=8, sql_timeout=5.0, gold=demo_gold.jsonl, git_commit=f59ef08

## Execution accuracy (95% bootstrap CI, 1000 resamples)

| Run | Model | Docs | Repair | N | Accuracy | 95% CI |
|---|---|---|---|---|---|---|
| demo_full | gemini/gemini-3.5-flash-lite | True | True | 20 | 100.0% | [100.0%, 100.0%] |
| demo_norepair | gemini/gemini-3.5-flash-lite | True | False | 20 | 95.0% | [85.0%, 100.0%] |

## Latency, tokens and cost per question

| Run | Median latency (ms) | p95 latency (ms) | Prompt tokens | Completion tokens | Cost (USD) | Tool calls | Repairs |
|---|---|---|---|---|---|---|---|
| demo_full | 19632 | 28417 | 4569 | 230 | 0.00195 | 5.6 | 0.00 |
| demo_norepair | 19846 | 24166 | 4384 | 220 | 0.00187 | 5.5 | 0.00 |

## Status of each run

| Run | ok | max_steps | sql_error | llm_error | no_answer |
|---|---|---|---|---|---|
| demo_full | 20 | 0 | 0 | 0 | 0 |
| demo_norepair | 20 | 0 | 0 | 0 | 0 |

## Failure reasons (wrong answers only)

| Run | no_sql | pred_sql_error | result_mismatch |
|---|---|---|---|
| demo_full | 0 | 0 | 0 |
| demo_norepair | 0 | 0 | 1 |

## Accuracy by difficulty

| Run | Difficulty | N | Accuracy |
|---|---|---|---|
| demo_full | easy | 6 | 100.0% |
| demo_full | hard | 7 | 100.0% |
| demo_full | medium | 7 | 100.0% |
| demo_norepair | easy | 6 | 100.0% |
| demo_norepair | hard | 7 | 100.0% |
| demo_norepair | medium | 7 | 85.7% |

## Paired comparison

`demo_norepair` minus `demo_full` on 20 shared questions (the same resampled questions are used for both runs):

- Accuracy difference: -5.0 points, 95% CI [-15.0, +0.0] (CI includes 0, no clear difference).
