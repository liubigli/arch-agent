# Benchmark capability and tool-routing analysis

Primary source: raw JSON records. The saved evaluation is used only for legacy groundedness and language checks.
The full-CSV and graph-only runs were produced on different dates and may use different code revisions; deltas are descriptive, not causal.
The no-CSV abstention policy is marked as proposed in the approved reference and is reported separately.

## Model summary

| Condition | Model | Preferred routing | Acceptable routing | Clean routing | Args | Answer accuracy | Partial credit | Operational success | Invalid tool attempts | Avg latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_csv | command-r | 68.3% | 83.3% | 83.3% | 10.0% | 42.0% | 48.5% | 55.0% | 0 | 4.673s |
| full_csv | gemma4:31b | 83.3% | 96.7% | 93.3% | 100.0% | 90.0% | 94.0% | 96.7% | 2 | 15.134s |
| full_csv | gpt-oss:20b | 75.0% | 90.0% | 90.0% | 100.0% | 86.0% | 88.1% | 90.0% | 0 | 3.223s |
| full_csv | llama3.1 | 60.0% | 78.3% | 78.3% | 95.0% | 60.0% | 66.7% | 76.7% | 0 | 1.396s |
| full_csv | qwen3.5 | 81.7% | 98.3% | 98.3% | 90.0% | 78.0% | 80.6% | 85.0% | 0 | 3.653s |
| graph_no_csv | command-r | 66.7% | 79.6% | 79.6% | 10.0% | 40.0% | 47.0% | 46.7% | 0 | 9.261s |
| graph_no_csv | gemma4:31b | 72.2% | 83.3% | 75.9% | 65.0% | 64.0% | 65.6% | 78.3% | 14 | 32.629s |
| graph_no_csv | gpt-oss:20b | 55.6% | 70.4% | 70.4% | 65.0% | 50.0% | 53.6% | 66.7% | 14 | 3.518s |
| graph_no_csv | llama3.1 | 46.3% | 61.1% | 61.1% | 70.0% | 48.0% | 53.8% | 55.0% | 0 | 1.220s |
| graph_no_csv | qwen3.5 | 66.7% | 79.6% | 75.9% | 60.0% | 40.0% | 44.5% | 60.0% | 24 | 4.996s |

## Interpretation

- **full_csv:** best acceptable routing: qwen3.5 (98.3%); best answer partial credit: gemma4:31b (94.0%); best operational success: gemma4:31b (96.7%).
- **graph_no_csv:** best acceptable routing: gemma4:31b (83.3%); best answer partial credit: gemma4:31b (65.6%); best operational success: gemma4:31b (78.3%).

## Metric definitions

- Preferred routing: at least one preferred reference tool was called.
- Acceptable routing: a preferred or explicitly acceptable tool was called.
- Clean routing: acceptable routing without first attempting a tool unavailable in that condition.
- Argument accuracy: required semantic classes/material filters match the structured reference.
- Answer accuracy: exact correctness on automatically scorable reference questions.
- Partial credit: mean structured-reference score across scorable questions.
- Operational success: answered, no runtime error, acceptable routing, valid required arguments, no unsupported object IDs, and correct no-CSV abstention when applicable.
- Legacy groundedness: secondary metric copied from the saved evaluation, not the primary judgment source.

## Files

See question_level_audit.csv for every question, tool call, argument verdict, score, and answer.
