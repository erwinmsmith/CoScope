"""Scope permissions."""

from enum import Enum


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    APPEND = "append"
    SHARE = "share"
    DERIVE = "derive"
    MERGE = "merge"
    PROMOTE = "promote"
    EXPORT = "export"
    DELETE = "delete"
