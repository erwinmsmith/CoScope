"""Environment-backed configuration for offline and live provider modes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ProviderConfigurationError(ValueError):
    """Raised when live provider configuration is missing or invalid."""


def _read(
    environment: Mapping[str, str],
    key: str,
    default: str,
    *,
    fallback_key: str | None = None,
) -> str:
    value = environment.get(key)
    if value is None and fallback_key is not None:
        value = environment.get(fallback_key)
    return value if value is not None else default


def _positive_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ProviderConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ProviderConfigurationError(f"{name} must be positive")
    return parsed


def _optional_positive_int(name: str, value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return _positive_int(name, value)


def _positive_float(name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ProviderConfigurationError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise ProviderConfigurationError(f"{name} must be positive")
    return parsed


def _nonnegative_float(name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ProviderConfigurationError(f"{name} must be a number") from exc
    if parsed < 0:
        raise ProviderConfigurationError(f"{name} cannot be negative")
    return parsed


def _embedding_key(environment: Mapping[str, str]) -> str:
    explicit = environment.get("COSCOPE_EMBEDDING_API_KEY")
    if explicit is not None:
        return explicit
    provider = _read(
        environment, "COSCOPE_EMBEDDING_PROVIDER", "dashscope"
    ).casefold()
    if provider == "zhipu":
        return environment.get("ZHIPU_API_KEY", "")
    return environment.get("DASHSCOPE_API_KEY", "")


def _unit_interval(name: str, value: str) -> float:
    parsed = _nonnegative_float(name, value)
    if parsed > 1:
        raise ProviderConfigurationError(f"{name} must be between 0 and 1")
    return parsed


def _boolean(name: str, value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ProviderConfigurationError(
        f"{name} must be true/false, yes/no, on/off, or 1/0"
    )


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "deepseek"
    api_key: str = ""
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2
    max_tokens: int | None = None
    timeout_seconds: float = 60.0

    def validate(self) -> None:
        if self.provider != "deepseek":
            raise ProviderConfigurationError(
                f"unsupported LLM provider: {self.provider}"
            )
        if not self.api_key:
            raise ProviderConfigurationError(
                "DEEPSEEK_API_KEY is required when COSCOPE_RUNTIME_MODE=live"
            )
        if not self.model:
            raise ProviderConfigurationError("COSCOPE_LLM_MODEL cannot be empty")


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str = "fastembed"
    api_key: str = ""
    model: str = "BAAI/bge-small-en-v1.5"
    base_url: str = ""
    dimension: int = 384
    timeout_seconds: float = 60.0
    cache_dir: str = "fastembed_cache"
    threads: int = 2
    batch_size: int = 32
    result_cache_size: int = 4096
    local_files_only: bool = False

    def validate(self) -> None:
        if self.provider not in {"dashscope", "fastembed", "zhipu"}:
            raise ProviderConfigurationError(
                f"unsupported embedding provider: {self.provider}"
            )
        if self.provider != "fastembed" and not self.api_key:
            key_name = (
                "DASHSCOPE_API_KEY"
                if self.provider == "dashscope"
                else "ZHIPU_API_KEY"
            )
            raise ProviderConfigurationError(
                f"{key_name} is required when COSCOPE_RUNTIME_MODE=live"
            )
        expected_model = {
            "dashscope": "text-embedding-v3",
            "fastembed": "BAAI/bge-small-en-v1.5",
            "zhipu": "embedding-3",
        }[self.provider]
        if self.model != expected_model:
            raise ProviderConfigurationError(
                f"{self.provider} embedding model must be {expected_model}"
            )
        if self.provider == "fastembed":
            if self.dimension != 384:
                raise ProviderConfigurationError(
                    "BAAI/bge-small-en-v1.5 dimension must be 384"
                )
            if not self.cache_dir.strip():
                raise ProviderConfigurationError(
                    "COSCOPE_EMBEDDING_CACHE_DIR cannot be empty"
                )
            if (
                self.threads <= 0
                or self.batch_size <= 0
                or self.result_cache_size <= 0
            ):
                raise ProviderConfigurationError(
                    "local embedding threads, batch size, and result cache "
                    "size must be positive"
                )
        if self.model == "text-embedding-v3" and self.dimension not in {
            512,
            768,
            1024,
        }:
            raise ProviderConfigurationError(
                "text-embedding-v3 dimension must be 512, 768, or 1024"
            )
        if self.model == "embedding-3" and self.dimension not in {
            256,
            512,
            1024,
            2048,
        }:
            raise ProviderConfigurationError(
                "embedding-3 dimension must be 256, 512, 1024, or 2048"
            )


@dataclass(frozen=True)
class RetrievalSettings:
    medoid_threshold: float = 0.90
    minimum_pairwise_similarity: float = 0.85
    max_group_size: int = 16
    shared_candidate_k: int = 50

    def validate(self) -> None:
        for name, value in (
            ("medoid_threshold", self.medoid_threshold),
            ("minimum_pairwise_similarity", self.minimum_pairwise_similarity),
        ):
            if not 0 <= value <= 1:
                raise ProviderConfigurationError(f"{name} must be between 0 and 1")
        if self.max_group_size <= 0 or self.shared_candidate_k <= 0:
            raise ProviderConfigurationError(
                "max_group_size and shared_candidate_k must be positive"
            )


@dataclass(frozen=True)
class CoScopeSettings:
    runtime_mode: str
    llm: LLMSettings
    embedding: EmbeddingSettings
    retrieval: RetrievalSettings

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = ".env",
        *,
        environ: Mapping[str, str] | None = None,
    ) -> CoScopeSettings:
        if env_file is not None:
            load_dotenv(Path(env_file), override=False)
        source = environ if environ is not None else os.environ
        runtime_mode = _read(source, "COSCOPE_RUNTIME_MODE", "offline").casefold()
        if runtime_mode not in {"offline", "live"}:
            raise ProviderConfigurationError(
                "COSCOPE_RUNTIME_MODE must be 'offline' or 'live'"
            )

        llm = LLMSettings(
            provider=_read(source, "COSCOPE_LLM_PROVIDER", "deepseek").casefold(),
            api_key=_read(
                source, "COSCOPE_LLM_API_KEY", "", fallback_key="DEEPSEEK_API_KEY"
            ),
            model=_read(source, "COSCOPE_LLM_MODEL", "deepseek-v4-flash"),
            base_url=_read(
                source, "COSCOPE_LLM_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/"),
            temperature=_nonnegative_float(
                "COSCOPE_LLM_TEMPERATURE",
                _read(source, "COSCOPE_LLM_TEMPERATURE", "0.2"),
            ),
            max_tokens=_optional_positive_int(
                "COSCOPE_LLM_MAX_TOKENS",
                source.get("COSCOPE_LLM_MAX_TOKENS"),
            ),
            timeout_seconds=_positive_float(
                "COSCOPE_LLM_TIMEOUT_SECONDS",
                _read(source, "COSCOPE_LLM_TIMEOUT_SECONDS", "60"),
            ),
        )
        embedding_provider = _read(
            source, "COSCOPE_EMBEDDING_PROVIDER", "fastembed"
        ).casefold()
        embedding_defaults = {
            "dashscope": (
                "text-embedding-v3",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "1024",
            ),
            "fastembed": ("BAAI/bge-small-en-v1.5", "", "384"),
            "zhipu": (
                "embedding-3",
                "https://open.bigmodel.cn/api/paas/v4",
                "1024",
            ),
        }
        embedding_model, embedding_base_url, embedding_dimension = (
            embedding_defaults.get(
                embedding_provider,
                ("", "", "384"),
            )
        )
        embedding = EmbeddingSettings(
            provider=embedding_provider,
            api_key=_embedding_key(source),
            model=_read(source, "COSCOPE_EMBEDDING_MODEL", embedding_model),
            base_url=_read(
                source,
                "COSCOPE_EMBEDDING_BASE_URL",
                embedding_base_url,
            ).rstrip("/"),
            dimension=_positive_int(
                "COSCOPE_EMBEDDING_DIMENSION",
                _read(
                    source,
                    "COSCOPE_EMBEDDING_DIMENSION",
                    embedding_dimension,
                ),
            ),
            timeout_seconds=_positive_float(
                "COSCOPE_EMBEDDING_TIMEOUT_SECONDS",
                _read(source, "COSCOPE_EMBEDDING_TIMEOUT_SECONDS", "60"),
            ),
            cache_dir=_read(
                source,
                "COSCOPE_EMBEDDING_CACHE_DIR",
                "fastembed_cache",
            ),
            threads=_positive_int(
                "COSCOPE_EMBEDDING_THREADS",
                _read(source, "COSCOPE_EMBEDDING_THREADS", "2"),
            ),
            batch_size=_positive_int(
                "COSCOPE_EMBEDDING_BATCH_SIZE",
                _read(source, "COSCOPE_EMBEDDING_BATCH_SIZE", "32"),
            ),
            result_cache_size=_positive_int(
                "COSCOPE_EMBEDDING_RESULT_CACHE_SIZE",
                _read(
                    source,
                    "COSCOPE_EMBEDDING_RESULT_CACHE_SIZE",
                    "4096",
                ),
            ),
            local_files_only=_boolean(
                "COSCOPE_EMBEDDING_LOCAL_FILES_ONLY",
                _read(
                    source,
                    "COSCOPE_EMBEDDING_LOCAL_FILES_ONLY",
                    "false",
                ),
            ),
        )
        retrieval = RetrievalSettings(
            medoid_threshold=_unit_interval(
                "COSCOPE_MEDOID_THRESHOLD",
                _read(source, "COSCOPE_MEDOID_THRESHOLD", "0.90"),
            ),
            minimum_pairwise_similarity=_unit_interval(
                "COSCOPE_MIN_PAIRWISE_SIMILARITY",
                _read(source, "COSCOPE_MIN_PAIRWISE_SIMILARITY", "0.85"),
            ),
            max_group_size=_positive_int(
                "COSCOPE_MAX_GROUP_SIZE",
                _read(source, "COSCOPE_MAX_GROUP_SIZE", "16"),
            ),
            shared_candidate_k=_positive_int(
                "COSCOPE_SHARED_CANDIDATE_K",
                _read(source, "COSCOPE_SHARED_CANDIDATE_K", "50"),
            ),
        )
        settings = cls(runtime_mode, llm, embedding, retrieval)
        retrieval.validate()
        if settings.live:
            llm.validate()
            embedding.validate()
        return settings

    @property
    def live(self) -> bool:
        return self.runtime_mode == "live"
