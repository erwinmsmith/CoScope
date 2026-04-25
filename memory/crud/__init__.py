"""Memory CRUD Module.

High-level memory operations plus the ``auth`` sub-package that enforces
access control, policy validation and S4 false-merge checks. Authorization
and CRUD stay together so both can be wired to a permission database later.
"""

from memory.crud.manager import MemoryCRUD
from memory.crud.types import MemoryFilter, MemoryQuery
from memory.crud.auth import (
    AccessController,
    PolicyValidator,
    ValidationResult,
    check_false_merge,
    compute_fmr,
)

__all__ = [
    "MemoryCRUD",
    "MemoryFilter",
    "MemoryQuery",
    "AccessController",
    "PolicyValidator",
    "ValidationResult",
    "check_false_merge",
    "compute_fmr",
]
