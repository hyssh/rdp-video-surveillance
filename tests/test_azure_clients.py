from unittest import mock

import pytest

import azure_clients as ac


def test_embedding_dimensions_default(clean_env):
    assert ac.get_embedding_dimensions() == 3072


@pytest.mark.parametrize("value,expected", [("1536", 1536), ("3072", 3072)])
def test_embedding_dimensions_env(clean_env, value, expected):
    clean_env.setenv("EMBEDDING_DIMENSIONS", value)
    assert ac.get_embedding_dimensions() == expected


@pytest.mark.parametrize("value", ["abc", "0", "-5", "99999"])
def test_embedding_dimensions_invalid(clean_env, value):
    clean_env.setenv("EMBEDDING_DIMENSIONS", value)
    with pytest.raises(ValueError):
        ac.get_embedding_dimensions()


def test_default_deployments(clean_env):
    assert ac.get_chat_deployment_name() == "gpt-6.1-sol"
    assert ac.get_embedding_deployment_name() == "text-embedding-3-large"


def test_agent_env_names_are_honoured(clean_env):
    clean_env.setenv("AZURE_AI_AGENT_CHAT_MODEL_DEPLOYMENT_NAME", "chat-x")
    clean_env.setenv("AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME", "emb-x")
    assert ac.get_chat_deployment_name() == "chat-x"
    assert ac.get_embedding_deployment_name() == "emb-x"


@pytest.mark.parametrize("raw,expected", [
    ("https://a.openai.azure.com/openai/v1", "https://a.openai.azure.com"),
    ("https://a.openai.azure.com/openai/v1/", "https://a.openai.azure.com"),
    ("https://a.openai.azure.com/", "https://a.openai.azure.com"),
    (None, None),
])
def test_normalize_endpoint(raw, expected):
    assert ac.normalize_openai_endpoint(raw) == expected


def test_chat_service_uses_api_key_when_set(clean_env):
    clean_env.setenv("AZURE_OPENAI_APIKEY", "k")
    with mock.patch("semantic_kernel.connectors.ai.open_ai.AzureChatCompletion") as cls, \
            mock.patch.object(ac, "get_token_provider") as tp:
        ac.build_chat_completion_service()
    kwargs = cls.call_args.kwargs
    assert kwargs["api_key"] == "k"
    assert "ad_token_provider" not in kwargs
    tp.assert_not_called()


def test_chat_service_uses_entra_without_key(clean_env):
    with mock.patch("semantic_kernel.connectors.ai.open_ai.AzureChatCompletion") as cls, \
            mock.patch.object(ac, "get_token_provider", return_value=lambda: "tok"):
        ac.build_chat_completion_service()
    kwargs = cls.call_args.kwargs
    assert "api_key" not in kwargs
    assert callable(kwargs["ad_token_provider"])
    assert kwargs["deployment_name"] == "gpt-6.1-sol"


def test_embedding_service_entra_without_key(clean_env):
    with mock.patch("semantic_kernel.connectors.ai.open_ai.AzureTextEmbedding") as cls, \
            mock.patch.object(ac, "get_token_provider", return_value=lambda: "tok"):
        ac.build_embedding_service()
    kwargs = cls.call_args.kwargs
    assert "api_key" not in kwargs and "ad_token_provider" in kwargs
    assert kwargs["deployment_name"] == "text-embedding-3-large"


def test_embedding_settings_only_when_explicit(clean_env):
    assert ac.embedding_settings() is None
    clean_env.setenv("EMBEDDING_DIMENSIONS", "1024")
    assert ac.embedding_settings().dimensions == 1024


def test_blob_client_connection_string_wins(clean_env):
    clean_env.setenv("AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")
    with mock.patch("azure.storage.blob.BlobServiceClient") as cls:
        ac.get_blob_service_client()
    cls.from_connection_string.assert_called_once()


def test_blob_client_entra_with_account_name(clean_env):
    clean_env.setenv("AZURE_STORAGE_ACCOUNT_NAME", "acct")
    with mock.patch("azure.storage.blob.BlobServiceClient") as cls, \
            mock.patch.object(ac, "get_credential", return_value="cred"):
        ac.get_blob_service_client()
    cls.assert_called_once_with(account_url="https://acct.blob.core.windows.net", credential="cred")


def test_blob_client_requires_some_config(clean_env):
    with pytest.raises(ValueError):
        ac.get_blob_service_client()


def test_format_exception_includes_cause_and_is_never_empty():
    try:
        try:
            raise KeyError("inner")
        except KeyError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as e:
        text = ac.format_exception(e)
    assert "outer" in text and "inner" in text
    assert ac.format_exception(ValueError()).strip()
