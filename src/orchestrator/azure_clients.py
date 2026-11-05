"""Shared Azure client factories for the orchestrator.

Every factory supports two authentication modes:
  * API key / connection string when the corresponding env var is set (legacy path)
  * Microsoft Entra ID via DefaultAzureCredential (env service principal, managed identity,
    Azure CLI login, ...) when no key is set
"""
import os
import logging
from functools import lru_cache
from typing import Callable, Optional

logger = logging.getLogger(__name__)

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
DEFAULT_CHAT_DEPLOYMENT = "gpt-6.1-sol"
DEFAULT_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 3072
DEFAULT_OPENAI_API_VERSION = "2025-04-01-preview"
MAX_EMBEDDING_DIMENSIONS = 16000  # pgvector hard limit for the `vector` type


def env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first non-empty environment variable among names."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


@lru_cache(maxsize=1)
def get_credential():
    from azure.identity import DefaultAzureCredential
    return DefaultAzureCredential()


def get_token_provider(scope: str = COGNITIVE_SERVICES_SCOPE) -> Callable[[], str]:
    from azure.identity import get_bearer_token_provider
    return get_bearer_token_provider(get_credential(), scope)


def normalize_openai_endpoint(endpoint: Optional[str]) -> Optional[str]:
    """Accept both the resource root and the '/openai/v1' style URL."""
    if not endpoint:
        return endpoint
    endpoint = endpoint.strip().rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if endpoint.lower().endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
    return endpoint


def get_embedding_dimensions() -> int:
    """Vector size stored in Postgres (EMBEDDING_DIMENSIONS, default 3072)."""
    raw = env_first("EMBEDDING_DIMENSIONS")
    if raw is None:
        return DEFAULT_EMBEDDING_DIMENSIONS
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"EMBEDDING_DIMENSIONS must be an integer, got {raw!r}") from e
    if not 1 <= value <= MAX_EMBEDDING_DIMENSIONS:
        raise ValueError(f"EMBEDDING_DIMENSIONS must be between 1 and {MAX_EMBEDDING_DIMENSIONS}, got {value}")
    return value


def get_chat_deployment_name() -> str:
    return env_first("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "AZURE_AI_AGENT_CHAT_MODEL_DEPLOYMENT_NAME",
                     default=DEFAULT_CHAT_DEPLOYMENT)


def get_embedding_deployment_name() -> str:
    return env_first("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME",
                     default=DEFAULT_EMBEDDING_DEPLOYMENT)


def _auth_kwargs(api_key: Optional[str]) -> dict:
    if api_key:
        return {"api_key": api_key}
    logger.info("No API key configured; using Microsoft Entra ID token authentication")
    return {"ad_token_provider": get_token_provider()}


def build_chat_completion_service(service_id: Optional[str] = None):
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    endpoint = normalize_openai_endpoint(env_first("AZURE_OPENAI_ENDPOINT"))
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not configured")
    kwargs = dict(
        endpoint=endpoint,
        deployment_name=get_chat_deployment_name(),
        api_version=env_first("AZURE_OPENAI_API_VERSION", default=DEFAULT_OPENAI_API_VERSION),
        **_auth_kwargs(env_first("AZURE_OPENAI_APIKEY")),
    )
    if service_id:
        kwargs["service_id"] = service_id
    return AzureChatCompletion(**kwargs)


def build_embedding_service():
    from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
    endpoint = normalize_openai_endpoint(env_first("AZURE_TEXT_EMBEDDING_ENDPOINT", "AZURE_OPENAI_ENDPOINT"))
    if not endpoint:
        raise ValueError("AZURE_TEXT_EMBEDDING_ENDPOINT / AZURE_OPENAI_ENDPOINT is not configured")
    return AzureTextEmbedding(
        service_id="azure-text-embedding",
        endpoint=endpoint,
        deployment_name=get_embedding_deployment_name(),
        api_version=env_first("AZURE_TEXT_EMBEDDING_API_VERSION", "AZURE_OPENAI_API_VERSION",
                              default=DEFAULT_OPENAI_API_VERSION),
        **_auth_kwargs(env_first("AZURE_TEXT_EMBEDDING_APIKEY", "AZURE_OPENAI_APIKEY")),
    )


def embedding_settings():
    """Ask the service for EMBEDDING_DIMENSIONS only when it is set explicitly
    (text-embedding-3-* supports shortening; ada-002 rejects the parameter)."""
    if env_first("EMBEDDING_DIMENSIONS") is None:
        return None
    from semantic_kernel.connectors.ai.open_ai import OpenAIEmbeddingPromptExecutionSettings
    return OpenAIEmbeddingPromptExecutionSettings(dimensions=get_embedding_dimensions())


def get_blob_service_client():
    """Connection string when set, otherwise AZURE_STORAGE_ACCOUNT_NAME + Entra ID."""
    from azure.storage.blob import BlobServiceClient
    connection_string = env_first("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)
    account_name = env_first("AZURE_STORAGE_ACCOUNT_NAME")
    if not account_name:
        raise ValueError("Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME")
    return BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=get_credential())


def format_exception(exc: BaseException) -> str:
    """Readable message with the cause chain (SK/OpenAI wrap the real error)."""
    parts, seen = [], set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen and len(parts) < 5:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return " <- ".join(parts)
