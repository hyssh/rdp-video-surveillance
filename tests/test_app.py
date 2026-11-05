import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi.testclient import TestClient

JOB_ID = "11111111-1111-1111-1111-111111111111"
DESCRIPTION = '{"risk_score": 10, "risk_level": "low", "action_type": "browsing", "description": "desktop"}'


def test_root_endpoint(app_module):
    assert TestClient(app_module.orchestrator).get("/").json()["message"]


def test_omniparser_base_url(app_module, monkeypatch):
    monkeypatch.setenv("OMNIPARSER_HOST", "omniparser")
    monkeypatch.setenv("OMNIPARSER_PORT", "9000")
    assert app_module.omniparser_base_url() == "http://omniparser:9000"


def test_split_video_extracts_frames(app_module, fake_blob, sample_video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    urls = app_module.split_video(str(sample_video), "", output_dir="frames")
    assert len(urls) == 3  # 0s, 3s, 6s
    assert urls[0].endswith("sample/sample_0000000000.jpg")
    assert next(iter(fake_blob.store.values()))[:2] == b"\xff\xd8"


def test_split_video_raises_on_storage_error(app_module, sample_video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(app_module, "get_blob_service_client", side_effect=ValueError("no storage")):
        with pytest.raises(ValueError, match="no storage"):
            app_module.split_video(str(sample_video), "")


def test_split_video_raises_on_missing_file(app_module, fake_blob, tmp_path):
    with pytest.raises(FileNotFoundError):
        app_module.split_video(str(tmp_path / "missing.mp4"), "")


def test_upload_without_storage_config_is_400(app_module, monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_NAME", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    r = TestClient(app_module.orchestrator).post("/v2/ingest-video/", files={"file": ("a.mp4", b"x", "video/mp4")})
    assert r.status_code == 400, r.text
    assert "AZURE_STORAGE" in r.text


class FakeDB:
    def __init__(self):
        self.frames, self.review, self.embedded_dims = [], None, set()

    def insert_video(self, v):
        return 1

    def insert_frames(self, vid, urls):
        for i, u in enumerate(urls):
            self.frames.append({"id": i + 1, "container_name": "frames", "blob_name": u.split("/frames/")[1],
                                "url": u, "frame_timestamp": i * 3, "image_description": None})
        return {"status": "success", "count": len(urls)}

    def get_frames(self, vid):
        return [dict(f) for f in self.frames]

    def update_embeddings(self, updates):
        for fid, desc, vec in updates:
            self.embedded_dims.add(len(vec))
            for f in self.frames:
                if f["id"] == fid:
                    f["image_description"] = desc
        return len(updates)

    def update_video(self, vid, description=None, **k):
        self.review = description
        return True


class FakeEmbeddings:
    def __init__(self, dims):
        self.dims = dims

    async def get_embeddings(self, values, options=None):
        return [SimpleNamespace(vector=[0.0] * self.dims) for _ in values]


def _ingest(app_module, sample_video, dims=3072, query="?wait=true"):
    db = FakeDB()
    seen = {}

    async def fake_agent_session(session_id=None):
        return object(), SimpleNamespace(state={})

    async def fake_chat(agent, base64_image, omniparser_response, session=None, custom_prompt=None):
        return DESCRIPTION

    async def fake_omni(b64):
        return {"som_image_base64": b64, "parsed_content_list": [
            {"type": "text", "interactivity": False, "content": "cmd.exe", "bbox": [0, 0, 0.5, 0.5]}]}

    async def fake_sec(agent, image_analysis_results, custom_prompt=None, **kw):
        seen["sec_input"] = image_analysis_results
        return SimpleNamespace(content='{"risk_score": 12, "risk_level": "low"}')

    patches = [
        mock.patch.object(app_module, "create_job", return_value=JOB_ID),
        mock.patch.object(app_module, "update_job", return_value=True),
        mock.patch.object(app_module, "insert_video", db.insert_video),
        mock.patch.object(app_module, "insert_frames", db.insert_frames),
        mock.patch.object(app_module, "get_frames_by_video_id", db.get_frames),
        mock.patch.object(app_module, "update_frames_with_embeddings", db.update_embeddings),
        mock.patch.object(app_module, "update_frames_analysis", return_value=True),
        mock.patch.object(app_module, "update_video_metrics", return_value=True),
        mock.patch.object(app_module, "update_video_with_security_review", db.update_video),
        mock.patch.object(app_module, "get_chat_completion_agent_session", fake_agent_session),
        mock.patch.object(app_module, "chat_completion_agent_run", fake_chat),
        mock.patch.object(app_module, "get_or_create_chat_completion_agent", mock.AsyncMock(return_value=object())),
        mock.patch.object(app_module, "security_reviewer_agent_run", fake_sec),
        mock.patch.object(app_module, "_embedding_service", return_value=FakeEmbeddings(dims)),
        mock.patch.object(app_module, "call_omniparser", fake_omni),
    ]
    for p in patches:
        p.start()
    try:
        client = TestClient(app_module.orchestrator)
        with open(sample_video, "rb") as fh:
            resp = client.post(f"/v2/ingest-video/{query}", files={"file": ("sample.mp4", fh, "video/mp4")})
    finally:
        for p in patches:
            p.stop()
    return resp, db, seen


def test_ingest_video_v2_end_to_end_mocked(app_module, fake_blob, sample_video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resp, db, seen = _ingest(app_module, sample_video)
    assert resp.status_code == 200, resp.text
    assert resp.json()["video_id"] == 1
    assert len(db.frames) == 3 and db.embedded_dims == {3072}
    assert db.review.startswith('{"risk_score"')
    assert len(seen["sec_input"]) == 3


def test_ingest_async_returns_202_with_job(app_module, fake_blob, sample_video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resp, _, _ = _ingest(app_module, sample_video, query="")
    assert resp.status_code == 202, resp.text
    assert resp.json()["job_id"] == JOB_ID


def test_ingest_rejects_wrong_embedding_dimension(app_module, fake_blob, sample_video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resp, db, _ = _ingest(app_module, sample_video, dims=1536)
    assert resp.status_code >= 400
    assert "1536" in resp.text and "EMBEDDING_DIMENSIONS" in resp.text
    assert not db.embedded_dims


def test_ingest_without_frames_returns_400(app_module, fake_blob, sample_video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(app_module, "split_video", return_value=[]):
        resp, _, _ = _ingest(app_module, sample_video)
    assert resp.status_code == 400, resp.text
    assert "No frames could be extracted" in resp.text


def test_ingest_split_error_is_reported_not_swallowed(app_module, fake_blob, sample_video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(app_module, "split_video", side_effect=ValueError("Invalid frame rate: 0")):
        resp, _, _ = _ingest(app_module, sample_video)
    assert resp.status_code >= 400
    assert "Invalid frame rate" in resp.text


def test_agent_error_detail_is_never_empty():
    from agents.chat_completion import chat_completion_agent as m
    from fastapi import HTTPException

    class Boom:
        async def run(self, *a, **k):
            raise RuntimeError("model exploded")

    with pytest.raises(HTTPException) as e:
        asyncio.run(m.chat(Boom(), "aGk=", []))
    assert e.value.status_code == 500
    assert "model exploded" in e.value.detail
