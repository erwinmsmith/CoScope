# Test Suite

The tests exercise runtime contracts rather than legacy dataset construction.
Every security invariant should include a negative test.

- `test_scope_policy.py`: tenant/domain/visibility/branch boundaries.
- `test_grouping.py`: scope signatures, similarity thresholds, no chaining.
- `test_retrieval_runtime.py`: safe reuse, fallback, promotion, trace redaction.
- `test_benchmark_metrics.py`: official benchmark scoring, token ledger, threshold config.
- `test_benchmark_suite_and_modes.py`: all dataset loaders, MATH grading, live mode graphs.
- `test_code_benchmark.py`: strict code extraction, EvalPlus score mapping, and Docker isolation.
- `test_context_reasoning.py`: context pollution controls and reasoning semantics.
- `test_cache_and_execution.py`: cache isolation/invalidation and mock execution.

Run `pytest` from the repository root.
