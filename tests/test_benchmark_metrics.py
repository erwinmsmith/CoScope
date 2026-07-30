import pytest

from coscope import CoScopeRuntime
from coscope.config import CoScopeSettings, ProviderConfigurationError
from coscope.core import UsageLedger
from coscope.evaluation.task_metrics import (
    INVALID_ANSWER,
    aime_is_correct,
    exact_match,
    extract_aime_answer,
    extract_gsm8k_answer,
    gsm8k_is_correct,
    token_f1,
)


def test_gsm8k_official_answer_extraction_requires_marker():
    assert extract_gsm8k_answer("work\n#### -1,234.5") == "-1234.5"
    assert extract_gsm8k_answer("the answer is 18") == INVALID_ANSWER
    assert gsm8k_is_correct("reasoning\n#### 18", "gold\n#### 18")


def test_aime_answer_extraction_accepts_leading_zeroes_but_rejects_expressions():
    assert extract_aime_answer(r"work\FINAL_ANSWER: \boxed{073}") == "73"
    assert extract_aime_answer("FINAL_ANSWER: 1000") == INVALID_ANSWER
    assert extract_aime_answer("FINAL_ANSWER: 70+3") == INVALID_ANSWER
    assert aime_is_correct("073", "73")


def test_hotpot_official_normalization_removes_articles_and_punctuation():
    assert exact_match("The Eiffel Tower!", "eiffel tower") == 1.0
    assert token_f1("Paris, France", "Paris") == pytest.approx(2 / 3)
    assert token_f1("yes", "no") == 0.0


def test_usage_ledger_aggregates_categories_and_models():
    ledger = UsageLedger()
    ledger.record(
        "llm",
        "deepseek-v4-flash",
        {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "reasoning_tokens": 5,
            "total_tokens": 18,
        },
    )
    ledger.record(
        "embedding",
        "embedding-3:1024",
        {"prompt_tokens": 4, "total_tokens": 4},
        metadata={"latency_seconds": "0.125"},
    )
    summary = ledger.summary()
    assert summary["all"]["calls"] == 2
    assert summary["all"]["total_tokens"] == 22
    assert summary["by_category"]["llm"]["reasoning_tokens"] == 5
    assert ledger.duration_seconds("embedding") == pytest.approx(0.125)


def test_threshold_environment_is_injected_into_runtime():
    settings = CoScopeSettings.from_env(
        None,
        environ={
            "COSCOPE_RUNTIME_MODE": "offline",
            "COSCOPE_MEDOID_THRESHOLD": "0.72",
            "COSCOPE_MIN_PAIRWISE_SIMILARITY": "0.68",
            "COSCOPE_MAX_GROUP_SIZE": "7",
            "COSCOPE_SHARED_CANDIDATE_K": "11",
        },
    )
    runtime = CoScopeRuntime.from_settings(settings)
    assert runtime.retrieval.grouper.medoid_threshold == 0.72
    assert runtime.retrieval.grouper.minimum_pairwise_similarity == 0.68
    assert runtime.retrieval.grouper.max_group_size == 7
    assert runtime.retrieval.shared_retriever.candidate_k == 11


def test_threshold_environment_rejects_out_of_range_values():
    with pytest.raises(ProviderConfigurationError, match="between 0 and 1"):
        CoScopeSettings.from_env(
            None,
            environ={
                "COSCOPE_RUNTIME_MODE": "offline",
                "COSCOPE_MEDOID_THRESHOLD": "1.01",
            },
        )
