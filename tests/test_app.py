import asyncio
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


def test_root_endpoint(app_module):
    assert TestClient(app_module.orchestrator).get("/").json()["message"]


def test_omniparser_base_url(app_module, monkeypatch):
    monkeypatch.setenv("OMNIPARSER_HOST", "omniparser")
    monkeypatch.setenv("OMNIPARSER_PORT", "9000")
    assert app_module.omniparser_base_url() == "http://omniparser:9000"


def test_split_video_extracts_frames(app_module, fake_blob, sample_video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    urls = app_module.split_video(str(sample_video), output_dir="frames")
    assert len(urls) == 3  # 0s, 3s, 6s
    assert urls[0].endswith("sample/sample_0000000000.jpg")
    assert next(iter(fake_blob.store.values()))[:2] == b"\xff\xd8"


def test_split_video_raises_on_storage_error(app_module, sample_video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(app_module, "get_blob_service_client", side_effect=ValueError("no storage")):
        with pytest.raises(ValueError, match="no storage"):
            app_module.split_video(str(sample_video))


def test_split_video_raises_on_missing_file(app_module, fake_blob, tmp_path):
    with pytest.raises(FileNotFoundError):
        app_module.split_video(str(tmp_path / "missing.mp4"))


def test_upload_without_storage_config_is_500(app_module, monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_NAME", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    f = SimpleNamespace(filename="a.mp4", content_type="video/mp4")
    with pytest.raises(HTTPException) as e:
        asyncio.run(app_module.upload_to_blob_storage(f, "videos"))
    assert e.value.status_code == 500
    assert "AZURE_STORAGE" in e.value.detail


class FakeDB:
    def __init__(self):
        self.frames, self.video_review = [], None

    def insert_video(self, v):
        return 1

    def insert_frames(self, vid, urls):
        for i, u in enumerate(urls):
            self.frames.append({"id": i + 1, "container_name": "frames", "blob_name": u.split("/frames/")[1],
                                "url": u, "frame_timestamp": i * 3, "image_description": None})
        return {"status": "success", "count": len(urls)}

    def get_frames(self, vid):
        return [dict(f) for f in self.frames]

    def update_frame(self, fid, desc, vec):
        for f in self.frames:
            if f["id"] == fid:
                f["vec_len"] = len(vec)
                f["image_description"] = desc
        return True

    def update_video(self, vid, description=None, **k):
        self.video_review = description
        return True


def _ingest(app_module, sample_video, embedding_dims=3072):
    db, seen = FakeDB(), {}

    async def fake_chat(agent, base64_image, omniparser_response, thread, user_input=None):
        return "Fake analysis"

    async def fake_omni(b64):
        return {"som_image_base64": b64, "parsed_content_list": [
            {"type": "text", "interactivity": False, "content": "cmd.exe", "bbox": [0, 0, 1, 1]}]}

    async def fake_sec(agent, image_analysis_results, custom_prompt=None, thread=None):
        seen["sec_input"] = image_analysis_results
        return SimpleNamespace(content="Security review: mock")

    emb = mock.Mock()
    emb.ai_model_id = "text-embedding-3-large"
    emb.generate_embeddings = mock.AsyncMock(return_value=[np.zeros(embedding_dims)])
    patches = [
        mock.patch.object(app_module, "insert_video", db.insert_video),
        mock.patch.object(app_module, "insert_frames", db.insert_frames),
        mock.patch.object(app_module, "get_frames_by_video_id", db.get_frames),
        mock.patch.object(app_module, "update_frame_with_embedding", db.update_frame),
        mock.patch.object(app_module, "update_video_with_security_review", db.update_video),
        mock.patch.object(app_module, "chat_completion_agent_run", fake_chat),
        mock.patch.object(app_module, "build_embedding_service", return_value=emb),
        mock.patch.object(app_module, "security_reviewer_agent_run", fake_sec),
        mock.patch.object(app_module, "call_omniparser", fake_omni),
    ]
    for p in patches:
        p.start()
    try:
        client = TestClient(app_module.orchestrator)
        with open(sample_video, "rb") as fh:
            resp = client.post("/v2/ingest-video/", files={"file": ("sample.mp4", fh, "video/mp4")})
    finally:
        for p in patches:
            p.stop()
    return resp, db, seen


def test_ingest_video_v2_end_to_end_mocked(app_module, fake_blob, sample_video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resp, db, seen = _ingest(app_module, sample_video)
    assert resp.status_code == 200, resp.text
    assert resp.json()["video_id"] == 1
    assert len(db.frames) == 3 and all(f["vec_len"] == 3072 for f in db.frames)
    assert db.video_review.startswith("Security review")
    assert len(seen["sec_input"]) == 3


def test_ingest_rejects_wrong_embedding_dimension(app_module, fake_blob, sample_video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resp, db, _ = _ingest(app_module, sample_video, embedding_dims=1536)
    assert resp.status_code == 500
    assert "1536" in resp.text and "EMBEDDING_DIMENSIONS" in resp.text
    assert not any("vec_len" in f for f in db.frames)


def test_ingest_without_frames_returns_400_not_500(app_module, fake_blob, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(app_module, "split_video", return_value=[]), \
            mock.patch.object(app_module, "store_video_frame_metadata", mock.AsyncMock(return_value={"video_id": 1})), \
            mock.patch.object(app_module, "update_image_review_and_textvector_v2", mock.AsyncMock(side_effect=lambda r: r)), \
            mock.patch.object(app_module, "get_frames_by_video_id", return_value=[]), \
            mock.patch.object(app_module, "get_or_create_chat_completion_agent", mock.AsyncMock(return_value=object())):
        r = TestClient(app_module.orchestrator).post(
            "/v2/ingest-video/", files={"file": ("a.mp4", b"x", "video/mp4")})
    assert r.status_code == 400, r.text


def test_agent_error_detail_is_never_empty():
    from agents.chat_completion import chat_completion_agent as m
    from semantic_kernel.agents import ChatCompletionAgent

    class Boom(ChatCompletionAgent):
        async def get_response(self, *a, **k):
            raise RuntimeError("model exploded")

    agent = Boom.model_construct()
    with pytest.raises(HTTPException) as e:
        asyncio.run(m.chat(agent, "aGk=", []))
    assert "model exploded" in e.value.detail
