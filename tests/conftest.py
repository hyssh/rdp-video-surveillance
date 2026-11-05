import os
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORCH = ROOT / "src" / "orchestrator"
sys.path.insert(0, str(ORCH))

# postgresql/__init__ connects at import time, so the pool is mocked for the whole session.
mock.patch("psycopg2.pool.ThreadedConnectionPool").start()

# Fake, non-secret environment; the cloud is always mocked in this suite.
FAKE_ENV = {
    "AZURE_STORAGE_ACCOUNT_NAME": "fakeacct",
    "OMNIPARSER_HOST": "127.0.0.1",
    "OMNIPARSER_PORT": "8081",
    "OMNIPARSER_ENABLE": "true",
    "CHAT_COMPLETION_ENABLE": "true",
    "AZURE_OPENAI_API_KEY": "fake-key",
    "AZURE_OPENAI_CHAT_COMPLETION_MODEL": "gpt-6.1-sol",
    "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com",
    "PGHOST": "x", "PGUSER": "x", "PGPASSWORD": "x", "PGDATABASE": "x", "PGPORT": "5432",
}
for _k, _v in FAKE_ENV.items():
    os.environ.setdefault(_k, _v)

AUTH_ENV_VARS = [
    "AZURE_OPENAI_APIKEY", "AZURE_OPENAI_API_KEY", "AZURE_TEXT_EMBEDDING_APIKEY", "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_OPENAI_CHAT_COMPLETION_MODEL", "AZURE_OPENAI_EMBEDDING_MODEL",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "AZURE_AI_AGENT_CHAT_MODEL_DEPLOYMENT_NAME",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME",
    "AZURE_TEXT_EMBEDDING_ENDPOINT", "EMBEDDING_DIMENSIONS", "AZURE_STORAGE_ACCOUNT_NAME",
]


@pytest.fixture()
def clean_env(monkeypatch):
    for name in AUTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake.openai.azure.com")
    return monkeypatch


class FakeBlob:
    def __init__(self, store, container, name):
        self.store, self.key = store, (container, name)
        self.url = f"https://fakeacct.blob.core.windows.net/{container}/{name}"

    def upload_blob(self, data, overwrite=False, content_settings=None):
        self.store[self.key] = data

    def download_blob(self):
        m = mock.Mock()
        m.readall.return_value = self.store[self.key]
        return m


class FakeContainer:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def exists(self):
        return True

    def create_container(self):
        pass

    def get_blob_client(self, blob):
        return FakeBlob(self.store, self.name, blob)


class FakeBSC:
    def __init__(self):
        self.store = {}

    def get_container_client(self, name):
        return FakeContainer(self.store, name)


@pytest.fixture(scope="session")
def app_module():
    import app
    return app


@pytest.fixture()
def fake_blob(app_module):
    bsc = FakeBSC()
    with mock.patch.object(app_module, "get_blob_service_client", return_value=bsc):
        yield bsc


@pytest.fixture()
def sample_video(tmp_path):
    """Synthetic 7s 10fps video generated with OpenCV: frames at 0s, 3s, 6s."""
    import cv2
    import numpy as np

    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(70):
        writer.write(np.full((240, 320, 3), (i * 3) % 255, dtype=np.uint8))
    writer.release()
    assert path.stat().st_size > 0
    return path
