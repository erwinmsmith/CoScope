"""
Run A1/A3/A4/A5 over S1/S2/S3/S4 synthetic cases.
"""

from __future__ import annotations

import os

from evaluation import (
    evaluate_synthetic_suite,
    format_synthetic_case_tables,
    format_synthetic_table,
)


def main() -> None:
    os.environ.setdefault("COSCOPE_LOG_LEVEL", "WARNING")
    summaries = evaluate_synthetic_suite(k=3)

    print("# Synthetic Suite Summary")
    print(format_synthetic_table(summaries))

    print("\n# Per-Case Details")
    print(format_synthetic_case_tables(summaries))


if __name__ == "__main__":
    main()
