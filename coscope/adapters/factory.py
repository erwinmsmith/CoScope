"""Provider adapter factories."""

from coscope.adapters.embedding import (
    DashScopeEmbedding,
    EmbeddingAdapter,
    FastEmbedEmbedding,
    ZhipuEmbedding,
)
from coscope.adapters.llm import DeepSeekLLM, LLMAdapter
from coscope.config import EmbeddingSettings, LLMSettings
from coscope.core.usage import UsageLedger


def build_llm(
    settings: LLMSettings, usage_ledger: UsageLedger | None = None
) -> LLMAdapter:
    if settings.provider == "deepseek":
        return DeepSeekLLM(settings, usage_ledger=usage_ledger)
    raise ValueError(f"unsupported LLM provider: {settings.provider}")


def build_embedding(
    settings: EmbeddingSettings, usage_ledger: UsageLedger | None = None
) -> EmbeddingAdapter:
    if settings.provider == "dashscope":
        return DashScopeEmbedding(settings, usage_ledger=usage_ledger)
    if settings.provider == "fastembed":
        return FastEmbedEmbedding(settings, usage_ledger=usage_ledger)
    if settings.provider == "zhipu":
        return ZhipuEmbedding(settings, usage_ledger=usage_ledger)
    raise ValueError(f"unsupported embedding provider: {settings.provider}")
