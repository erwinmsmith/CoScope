"""Official-compatible answer metrics for supported benchmarks."""

from __future__ import annotations

import re
import string
from collections import Counter

GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[0-9.,]+)")
INVALID_ANSWER = "[invalid]"
FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER:\s*(.+)", flags=re.IGNORECASE)


def normalize_answer(text: str) -> str:
    """HotpotQA/SQuAD normalization: lower, de-punctuate, de-article, compact."""

    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in set(string.punctuation)
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def token_f1(prediction: str, answer: str) -> float:
    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(answer).split()
    if (
        normalized_special_mismatch(predicted, gold)
        or not predicted
        or not gold
    ):
        return float(predicted == gold)
    common = Counter(predicted) & Counter(gold)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def normalized_special_mismatch(predicted: list[str], gold: list[str]) -> bool:
    special = {"yes", "no", "noanswer"}
    predicted_text = " ".join(predicted)
    gold_text = " ".join(gold)
    return (
        predicted_text in special or gold_text in special
    ) and predicted_text != gold_text


def extract_gsm8k_answer(completion: str) -> str:
    """Match the numeric value immediately following GSM8K's #### marker."""

    match = GSM8K_ANSWER_RE.search(completion)
    if match is None:
        return INVALID_ANSWER
    return match.group(1).replace(",", "").strip()


def gsm8k_is_correct(prediction: str, reference: str) -> bool:
    gold = extract_gsm8k_answer(reference)
    if gold == INVALID_ANSWER:
        raise ValueError("GSM8K reference does not contain a valid #### answer")
    return extract_gsm8k_answer(prediction) == gold


def task_success_rate(successes: list[bool]) -> float:
    return sum(successes) / len(successes) if successes else 0.0


def extract_final_answer(completion: str) -> str:
    matches = FINAL_ANSWER_RE.findall(completion)
    return matches[-1].strip().rstrip(".") if matches else INVALID_ANSWER


def extract_aime_answer(completion: str) -> str:
    """Extract an AIME integer in the official 000--999 answer range."""
    candidate = extract_final_answer(completion)
    if candidate == INVALID_ANSWER:
        return INVALID_ANSWER
    boxed = re.fullmatch(r"\\boxed\{(?P<answer>[^{}]+)\}", candidate)
    if boxed:
        candidate = boxed.group("answer").strip()
    if not re.fullmatch(r"\d{1,3}", candidate):
        return INVALID_ANSWER
    value = int(candidate)
    return str(value) if 0 <= value <= 999 else INVALID_ANSWER


def aime_is_correct(prediction: str, reference: str) -> bool:
    parsed = extract_aime_answer(f"FINAL_ANSWER: {prediction}")
    return parsed != INVALID_ANSWER and parsed == str(int(reference))


def qa_answer_metrics(
    prediction: str,
    references: tuple[str, ...],
) -> dict[str, float]:
    if not references:
        raise ValueError("at least one QA reference is required")
    return {
        "em": max(exact_match(prediction, reference) for reference in references),
        "f1": max(token_f1(prediction, reference) for reference in references),
    }
