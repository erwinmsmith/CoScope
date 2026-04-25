"""Deprecated: authorization lives under :mod:`coscope.memory.crud.auth`.

This module is a thin re-export kept for backward compatibility. New code
should import from :mod:`coscope.memory.crud` directly.
"""

from memory.crud.auth import (
    AccessController,
    PolicyValidator,
    ValidationResult,
    check_false_merge,
    compute_fmr,
)

__all__ = [
    "AccessController",
    "PolicyValidator",
    "ValidationResult",
    "check_false_merge",
    "compute_fmr",
]
