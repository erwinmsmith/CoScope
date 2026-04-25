"""Memory CRUD: authorization and policy enforcement sub-module.

Houses access control, policy validation and S4 false-merge checks so that
all high-level memory gating happens alongside the CRUD layer. Designed to be
plugged into an external permission database later without touching the
retrieval pipeline.
"""

from memory.crud.auth.access_controller import AccessController
from memory.crud.auth.policy_validator import PolicyValidator, ValidationResult
from memory.crud.auth.s4_checker import check_false_merge, compute_fmr

__all__ = [
    "AccessController",
    "PolicyValidator",
    "ValidationResult",
    "check_false_merge",
    "compute_fmr",
]
