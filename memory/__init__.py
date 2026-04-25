"""
CoScope Memory Module.

Memory storage with multiple backend support and CRUD operations.
"""

# Backend stores
from memory.store import (
    InMemoryMemoryStore,
    MemoryManager,
    create_memory_store,
)

# CRUD operations
from memory.crud import MemoryCRUD, MemoryFilter, MemoryQuery

# Entry builders
from memory.builders import (
    PrivateBuilder,
    RestrictedBuilder,
    TaskSharedBuilder,
    WorkspaceBuilder,
)

__all__ = [
    # Stores
    "InMemoryMemoryStore",
    "MemoryManager",
    "create_memory_store",
    # CRUD
    "MemoryCRUD",
    "MemoryFilter",
    "MemoryQuery",
    # Builders
    "PrivateBuilder",
    "RestrictedBuilder",
    "TaskSharedBuilder",
    "WorkspaceBuilder",
]
