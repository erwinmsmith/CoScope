"""
Prompt isolation check (v2.2 §16.4 / §18.7).

Scans every `.py` file under `coscope/` and fails if any of them
imports a forbidden external prompt / template library. The whole point of
the v2.2 prompt-internalization rule is to make GoT / CoT / ToT prompts
entirely self-contained so ablation experiments stay reproducible across
dependency upgrades.

Usage
-----
    python -m coscope.scripts.check_prompt_isolation            # default src dir
    python -m coscope.scripts.check_prompt_isolation --src other/path

Exit code is 0 when clean, 1 when at least one violation is found.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


FORBIDDEN_IMPORT_PREFIXES = (
    "langchain.prompts",
    "langchain_core.prompts",
    "llama_index.prompts",
    "llama_index.core.prompts",
    "haystack.nodes.prompt",
    "guidance",
    "promptflow",
    "openai_function_call",
)


def _iter_python_files(root: Path) -> Iterable[Path]:
    for py in sorted(root.rglob("*.py")):
        # Skip our own check script to avoid self-match on string literals.
        if py.name == Path(__file__).name:
            continue
        yield py


def _check_file(py_file: Path) -> List[Tuple[int, str]]:
    """Return (lineno, module_name) for every forbidden import in this file."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return [(e.lineno or 0, f"[SyntaxError] {e.msg}")]

    violations: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    violations.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_forbidden(module):
                violations.append((node.lineno, module))
    return violations


def _is_forbidden(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


def check_no_external_prompt_import(src_dir: Path) -> List[str]:
    """Return a list of violation messages. Empty list == clean."""
    messages: List[str] = []
    for py_file in _iter_python_files(src_dir):
        for lineno, module in _check_file(py_file):
            messages.append(f"{py_file}:{lineno} imports {module!r}")
    return messages


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default="coscope",
        help="Root directory to scan (default: coscope/data)",
    )
    args = parser.parse_args(argv)

    src_dir = Path(args.src).resolve()
    if not src_dir.exists():
        print(f"[ERROR] src directory not found: {src_dir}", file=sys.stderr)
        return 2

    violations = check_no_external_prompt_import(src_dir)
    if not violations:
        print(
            f"[OK] {src_dir} has no forbidden prompt-library imports "
            f"({len(FORBIDDEN_IMPORT_PREFIXES)} prefixes checked)."
        )
        return 0

    print(f"[FAIL] Found {len(violations)} forbidden import(s) under {src_dir}:")
    for v in violations:
        print(f"  - {v}")
    print()
    print("Forbidden prefixes:")
    for p in FORBIDDEN_IMPORT_PREFIXES:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
