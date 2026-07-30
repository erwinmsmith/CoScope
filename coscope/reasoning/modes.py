"""Reasoning modes, independent from application agent topology."""

from enum import Enum


class ReasoningMode(str, Enum):
    COT = "cot"
    TOT = "tot"
    GOT = "got"
