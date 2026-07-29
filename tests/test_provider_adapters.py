from types import SimpleNamespace

import pytest

from coscope.adapters.embedding import DashScopeEmbedding, ZhipuEmbedding
from coscope.adapters.llm import DeepSeekLLM
from coscope.config import (
    CoScopeSettings,
    EmbeddingSettings,
    LLMSettings,
    ProviderConfigurationError,
)
from coscope.core import UsageLedger


class _FakeCompletions:
    def __init__(self):
        self.arguments = None

    def create(self, **kwargs):
        self.arguments = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
            ),
        )


class _FakeLLMClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class _FakeEmbeddings:
    def __init__(self):
        self.arguments = None

    def create(self, **kwargs):
        self.arguments = kwargs
        dimension = kwargs["dimensions"]
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[2.0] * dimension),
                SimpleNamespace(index=0, embedding=[1.0] * dimension),
            ]
        )


class _FakeEmbeddingClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


def test_live_settings_select_deepseek_flash_and_text_embedding_v3():
    settings = CoScopeSettings.from_env(
        None,
        environ={
            "COSCOPE_RUNTIME_MODE": "live",
            "DEEPSEEK_API_KEY": "deepseek-test",
            "DASHSCOPE_API_KEY": "dashscope-test",
        },
    )
    assert settings.llm.model == "deepseek-v4-flash"
    assert settings.llm.max_tokens is None
    assert settings.embedding.model == "text-embedding-v3"
    assert settings.embedding.dimension == 1024


def test_live_settings_require_both_provider_keys():
    with pytest.raises(ProviderConfigurationError, match="DASHSCOPE_API_KEY"):
        CoScopeSettings.from_env(
            None,
            environ={
                "COSCOPE_RUNTIME_MODE": "live",
                "DEEPSEEK_API_KEY": "deepseek-test",
            },
        )


def test_deepseek_adapter_forwards_model_and_usage():
    client = _FakeLLMClient()
    ledger = UsageLedger()
    adapter = DeepSeekLLM(
        LLMSettings(api_key="test", model="deepseek-v4-flash"),
        client=client,
        usage_ledger=ledger,
    )
    output = adapter.invoke([{"role": "user", "content": "question"}])
    assert output.text == "answer"
    assert output.usage["total_tokens"] == 5
    assert client.chat.completions.arguments["model"] == "deepseek-v4-flash"
    assert "max_tokens" not in client.chat.completions.arguments
    assert ledger.summary()["by_category"]["llm"]["total_tokens"] == 5


def test_deepseek_adapter_forwards_an_explicit_output_cap():
    client = _FakeLLMClient()
    adapter = DeepSeekLLM(
        LLMSettings(api_key="test", max_tokens=1234),
        client=client,
    )

    adapter.invoke([{"role": "user", "content": "question"}])

    assert client.chat.completions.arguments["max_tokens"] == 1234


def test_empty_output_cap_environment_value_means_unset():
    settings = CoScopeSettings.from_env(
        None,
        environ={
            "COSCOPE_RUNTIME_MODE": "live",
            "DEEPSEEK_API_KEY": "deepseek-test",
            "DASHSCOPE_API_KEY": "dashscope-test",
            "COSCOPE_LLM_MAX_TOKENS": "",
        },
    )

    assert settings.llm.max_tokens is None


def test_dashscope_adapter_forwards_dimension_and_restores_input_order():
    client = _FakeEmbeddingClient()
    adapter = DashScopeEmbedding(
        EmbeddingSettings(
            api_key="test",
            model="text-embedding-v3",
            dimension=512,
        ),
        client=client,
    )
    vectors = adapter.embed_many(["first", "second"])
    assert client.embeddings.arguments == {
        "model": "text-embedding-v3",
        "input": ["first", "second"],
        "dimensions": 512,
    }
    assert vectors[0][0] == 1.0
    assert vectors[1][0] == 2.0
    assert all(len(vector) == 512 for vector in vectors)


def test_zhipu_embedding_3_configuration_uses_attached_api_shape():
    client = _FakeEmbeddingClient()
    settings = EmbeddingSettings(
        provider="zhipu",
        api_key="test",
        model="embedding-3",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        dimension=1024,
    )
    adapter = ZhipuEmbedding(settings, client=client)
    vectors = adapter.embed_many(["first", "second"])
    assert adapter.model_version == "embedding-3:1024"
    assert client.embeddings.arguments["model"] == "embedding-3"
    assert len(vectors[0]) == 1024


def test_zhipu_env_selects_provider_specific_defaults():
    settings = CoScopeSettings.from_env(
        None,
        environ={
            "COSCOPE_RUNTIME_MODE": "live",
            "DEEPSEEK_API_KEY": "deepseek-test",
            "COSCOPE_EMBEDDING_PROVIDER": "zhipu",
            "ZHIPU_API_KEY": "zhipu-test",
        },
    )
    assert settings.embedding.model == "embedding-3"
    assert settings.embedding.base_url == "https://open.bigmodel.cn/api/paas/v4"
