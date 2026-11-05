import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

import azure_clients as ac
from agents import llm


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


def test_blob_client_connection_string_wins(clean_env):
    clean_env.setenv("AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")
    clean_env.setenv("AZURE_STORAGE_ACCOUNT_NAME", "acct")
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
    assert not ac.storage_configured()


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


def test_llm_default_models(clean_env):
    assert llm.chat_model() == "gpt-6.1-sol"
    assert llm.embedding_model() == "text-embedding-3-large"


def test_llm_agent_env_names_are_honoured(clean_env):
    clean_env.setenv("AZURE_AI_AGENT_CHAT_MODEL_DEPLOYMENT_NAME", "chat-x")
    clean_env.setenv("AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME", "emb-x")
    assert llm.chat_model() == "chat-x"
    assert llm.embedding_model() == "emb-x"


def test_llm_uses_api_key_when_set(clean_env):
    clean_env.setenv("AZURE_OPENAI_API_KEY", "k")
    with mock.patch.object(llm, "OpenAIChatCompletionClient") as cls, \
            mock.patch("azure.identity.DefaultAzureCredential") as cred:
        llm.make_chat_client()
    assert cls.call_args.kwargs["api_key"] == "k"
    assert "credential" not in cls.call_args.kwargs
    cred.assert_not_called()


def test_llm_uses_entra_without_key(clean_env):
    with mock.patch.object(llm, "OpenAIChatCompletionClient") as cls, \
            mock.patch("azure.identity.DefaultAzureCredential", return_value="cred"):
        llm.make_chat_client()
    kwargs = cls.call_args.kwargs
    assert kwargs["credential"] == "cred" and "api_key" not in kwargs
    assert kwargs["model"] == "gpt-6.1-sol"


class _Emb:
    def __init__(self, dims):
        self.dims, self.options = dims, None

    async def get_embeddings(self, values, options=None):
        self.options = options
        return [SimpleNamespace(vector=[0.0] * self.dims) for _ in values]


def test_embed_texts_requests_configured_dimensions(clean_env):
    clean_env.setenv("EMBEDDING_DIMENSIONS", "3072")
    client = _Emb(3072)
    vectors = asyncio.run(llm.embed_texts(client, ["a", "b"]))
    assert len(vectors) == 2 and len(vectors[0]) == 3072
    assert client.options == {"dimensions": 3072}


def test_embed_texts_no_dimensions_option_by_default(clean_env):
    client = _Emb(3072)
    asyncio.run(llm.embed_texts(client, ["a"]))
    assert client.options is None


def test_embed_texts_rejects_wrong_dimension(clean_env):
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        asyncio.run(llm.embed_texts(_Emb(1536), ["a"]))
