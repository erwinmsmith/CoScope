# Benchmark Datasets

Download the additional benchmark files from pinned upstream revisions:

```bash
python -m coscope.scripts.download_benchmarks
```

The downloader verifies SHA-256 before atomically placing files under
gitignored `raw/`.

| Benchmark | Local data | Upstream version | Size | License |
|---|---|---|---:|---|
| AIME 2024 | `raw/aime/aime_2024_{I,II}.parquet` | MathArena pinned commits | 30 | CC BY-NC-SA 4.0 |
| AIME 2025 | `raw/aime/aime_2025.parquet` | MathArena `c94da77...` | 30 | CC BY-NC-SA 4.0 |
| MBPP-Plus | `raw/mbpp_plus/MbppPlus.jsonl.gz` | EvalPlus v0.2.0 | 378 | Apache-2.0 |

AIME I and II are kept distinguishable in example IDs. Scoring accepts only
integer answers in the official `000`–`999` range. MathArena data is
non-commercial ShareAlike material; do not redistribute it under incompatible
terms.

MBPP-Plus prompts enter the model context, but canonical solutions and
base/plus tests do not. Generated code is extracted without execution and
scored with the official EvalPlus v0.3.1 image pinned to digest
`sha256:26b118...f4740`. CoScope invokes the evaluator in a Docker container
with networking disabled, a read-only root filesystem, dropped capabilities,
resource limits, and only an ignored artifact directory mounted writable.
The official task metric is MBPP+ pass@1: both base and augmented tests must
pass.

Sources:

- [MathArena AIME 2024 I](https://huggingface.co/datasets/MathArena/aime_2024_I)
- [MathArena AIME 2024 II](https://huggingface.co/datasets/MathArena/aime_2024_II)
- [MathArena AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025)
- [EvalPlus MBPP+ release](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0)
- [EvalPlus evaluator](https://github.com/evalplus/evalplus)
