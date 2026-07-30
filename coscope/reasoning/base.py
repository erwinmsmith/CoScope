"""Reasoning configuration independent from agent topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from coscope.reasoning.modes import ReasoningMode
from coscope.reasoning.node import ReasoningNode


@dataclass(frozen=True)
class ReasoningConfig:
    mode: ReasoningMode
    max_depth: int = 4
    branching_factor: int = 3
    pruning_policy: str = "score"


class ReasoningRuntime(Protocol):
    config: ReasoningConfig

    def nodes(self) -> list[ReasoningNode]: ...
