from types import SimpleNamespace

import pytest

from coscope.adapters.embedding import (
    DashScopeEmbedding,
    FastEmbedEmbedding,
    ZhipuEmbedding,
)
from coscope.adapters.factory import build_embedding
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


def test_live_settings_select_deepseek_flash_and_local_bge():
    settings = CoScopeSettings.from_env(
        None,
        environ={
            "COSCOPE_RUNTIME_MODE": "live",
            "DEEPSEEK_API_KEY": "deepseek-test",
        },
    )
    assert settings.llm.model == "deepseek-v4-flash"
    assert settings.llm.max_tokens is None
    assert settings.embedding.provider == "fastembed"
    assert settings.embedding.model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding.dimension == 384
    assert settings.embedding.threads == 2
    assert settings.embedding.batch_size == 32


def test_remote_embedding_settings_require_provider_key():
    with pytest.raises(ProviderConfigurationError, match="DASHSCOPE_API_KEY"):
        CoScopeSettings.from_env(
            None,
            environ={
                "COSCOPE_RUNTIME_MODE": "live",
                "DEEPSEEK_API_KEY": "deepseek-test",
                "COSCOPE_EMBEDDING_PROVIDER": "dashscope",
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
            provider="dashscope",
            api_key="test",
            model="text-embedding-v3",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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


class _FakeFastEmbedBackend:
    dimension = 384
    model_version = (
        "fastembed:BAAI/bge-small-en-v1.5:384:"
        "fastembed-test:sha256-0123456789abcdef"
    )
    model_sha256 = "0123456789abcdef" * 4

    def embed_many(self, texts):
        return (
            [
                tuple([float(index + 1)] * self.dimension)
                for index, _ in enumerate(texts)
            ],
            7,
            0.012,
            1,
            1,
        )


def test_fastembed_adapter_records_local_tokens_and_identity():
    ledger = UsageLedger()
    adapter = FastEmbedEmbedding(
        EmbeddingSettings(
            provider="fastembed",
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        ),
        backend=_FakeFastEmbedBackend(),
        usage_ledger=ledger,
    )

    vectors = adapter.embed_many(["first", "second"])

    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    assert adapter.model_version.endswith("sha256-0123456789abcdef")
    summary = ledger.summary()
    assert summary["by_category"]["embedding"]["calls"] == 1
    assert summary["by_category"]["embedding"]["prompt_tokens"] == 7
    assert summary["events"][0]["metadata"]["execution"] == "local_cpu"
    assert summary["events"][0]["metadata"]["result_cache_hits"] == "1"


def test_fastembed_factory_uses_injected_shared_backend(monkeypatch):
    from coscope.adapters.embedding import fastembed as fastembed_module

    backends = []

    def create_backend(settings):
        del settings
        backend = _FakeFastEmbedBackend()
        backends.append(backend)
        return backend

    fastembed_module._BACKENDS.clear()
    monkeypatch.setattr(fastembed_module, "_OnnxBackend", create_backend)
    settings = EmbeddingSettings(
        provider="fastembed",
        model="BAAI/bge-small-en-v1.5",
        dimension=384,
        cache_dir="test-cache",
    )

    first = build_embedding(settings)
    second = build_embedding(settings)

    assert isinstance(first, FastEmbedEmbedding)
    assert first.backend is second.backend
    assert len(backends) == 1
    fastembed_module._BACKENDS.clear()
