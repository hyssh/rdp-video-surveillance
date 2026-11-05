"""Shared Azure helpers for the orchestrator: Entra ID credential, Blob client, embedding size.

Authentication supports two modes:
  * API key / connection string when the corresponding env var is set
  * Microsoft Entra ID via DefaultAzureCredential (env service principal, managed identity,
    Azure CLI login, ...) when no key is set
Chat/embedding clients live in agents/llm.py.
"""
import os
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIMENSIONS = 3072
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


def storage_configured() -> bool:
    return bool(env_first("AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT_NAME"))


def get_blob_service_client(connection_string: Optional[str] = None):
    """Connection string (argument or env) when set, otherwise AZURE_STORAGE_ACCOUNT_NAME + Entra ID."""
    from azure.storage.blob import BlobServiceClient
    connection_string = connection_string or env_first("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)
    account_name = env_first("AZURE_STORAGE_ACCOUNT_NAME")
    if not account_name:
        raise ValueError("Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME")
    return BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=get_credential())


def format_exception(exc: BaseException) -> str:
    """Readable message with the cause chain, never empty."""
    parts, seen = [], set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen and len(parts) < 5:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return " <- ".join(parts)
