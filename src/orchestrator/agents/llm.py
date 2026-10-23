"""Microsoft Agent Framework (MAF) helpers shared by every agent and by the embedding calls.

Environment variables (the first one that is set wins):
  endpoint   AZURE_OPENAI_ENDPOINT
  api key    AZURE_OPENAI_API_KEY | AZURE_OPENAI_APIKEY   (unset -> Entra ID via DefaultAzureCredential)
  api ver    AZURE_OPENAI_API_VERSION
  chat       AZURE_OPENAI_CHAT_COMPLETION_MODEL | AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
  embedding  AZURE_OPENAI_EMBEDDING_MODEL | AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
             | AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME   (default text-embedding-ada-002)
  embedding endpoint / key fall back to AZURE_TEXT_EMBEDDING_ENDPOINT / AZURE_TEXT_EMBEDDING_APIKEY,
  then to the chat endpoint / key.
"""
import os
from typing import Any, Dict, List, Optional

from agent_framework import Agent, AgentSession, Content, Message
from agent_framework.openai import OpenAIChatCompletionClient, OpenAIEmbeddingClient

DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_OPTIONS = {"temperature": 0.1, "top_p": 0.1}


def _first_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def chat_model() -> Optional[str]:
    return _first_env("AZURE_OPENAI_CHAT_COMPLETION_MODEL", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")


def embedding_model() -> str:
    return _first_env(
        "AZURE_OPENAI_EMBEDDING_MODEL",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
        "AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME",
        default=DEFAULT_EMBEDDING_MODEL,
    )


def chat_configured() -> bool:
    return bool(_first_env("AZURE_OPENAI_ENDPOINT") and chat_model())


def embedding_endpoint() -> Optional[str]:
    return _first_env("AZURE_TEXT_EMBEDDING_ENDPOINT", "AZURE_OPENAI_ENDPOINT")


def embedding_key() -> Optional[str]:
    return _first_env("AZURE_TEXT_EMBEDDING_APIKEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_APIKEY")


def embedding_configured() -> bool:
    return bool(embedding_endpoint() and (embedding_key() or _first_env("AZURE_OPENAI_USE_ENTRA")))


def _auth(api_key: Optional[str]) -> Dict[str, Any]:
    if api_key:
        return {"api_key": api_key}
    from azure.identity import DefaultAzureCredential

    return {"credential": DefaultAzureCredential()}


def make_chat_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=chat_model(),
        azure_endpoint=_first_env("AZURE_OPENAI_ENDPOINT"),
        api_version=_first_env("AZURE_OPENAI_API_VERSION", default=DEFAULT_API_VERSION),
        **_auth(_first_env("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_APIKEY")),
    )


def make_agent(name: str, instructions: str) -> Agent:
    return make_chat_client().as_agent(name=name, instructions=instructions, default_options=dict(DEFAULT_OPTIONS))


def make_embedding_client() -> OpenAIEmbeddingClient:
    return OpenAIEmbeddingClient(
        model=embedding_model(),
        azure_endpoint=embedding_endpoint(),
        api_version=_first_env("AZURE_TEXT_EMBEDDING_API_VERSION", "AZURE_OPENAI_API_VERSION", default=DEFAULT_API_VERSION),
        **_auth(embedding_key()),
    )


async def embed_texts(client: OpenAIEmbeddingClient, texts: List[str]) -> List[List[float]]:
    result = await client.get_embeddings(list(texts))
    return [list(item.vector) for item in result]


def image_message(base64_image: str, text: str, mime_type: str = "image/jpeg") -> Message:
    return Message(
        role="user",
        contents=[Content.from_uri(f"data:{mime_type};base64,{base64_image}"), Content.from_text(text)],
    )


def response_text(response: Any) -> str:
    return getattr(response, "text", None) or str(response)


def trim_session(session: AgentSession, keep: int) -> bool:
    """Keep only the last `keep` stored messages (image messages are large). Returns True if trimmed."""
    for state in session.state.values():
        messages = state.get("messages") if isinstance(state, dict) else None
        if isinstance(messages, list) and len(messages) > keep:
            state["messages"] = messages[-keep:]
            return True
    return False


def missing_llm_config(chat: bool = True, embedding: bool = True) -> List[str]:
    """Human-readable names of the env vars still needed (empty list = configured)."""
    missing: List[str] = []
    if chat:
        if not _first_env("AZURE_OPENAI_ENDPOINT"):
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not chat_model():
            missing.append("AZURE_OPENAI_CHAT_COMPLETION_MODEL (or AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)")
    if embedding and not embedding_endpoint():
        missing.append("AZURE_OPENAI_ENDPOINT (or AZURE_TEXT_EMBEDDING_ENDPOINT)")
    return missing
