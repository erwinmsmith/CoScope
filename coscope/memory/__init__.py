"""
CoScope Memory Module.

Memory storage with multiple backend support and CRUD operations.
"""

# Backend stores
from coscope.memory.store import (
    InMemoryMemoryStore,
    MemoryManager,
    create_memory_store,
)

# CRUD operations
from coscope.memory.crud import MemoryCRUD, MemoryFilter, MemoryQuery

__all__ = [
    # Stores
    "InMemoryMemoryStore",
    "MemoryManager",
    "create_memory_store",
    # CRUD
    "MemoryCRUD",
    "MemoryFilter",
    "MemoryQuery",
]
