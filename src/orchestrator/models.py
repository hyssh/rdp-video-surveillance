from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str = Field(description="Machine-readable error code, e.g. missing_config, video_not_found")
    stage: Optional[str] = Field(None, description="Pipeline/request stage where the error occurred")
    job_id: Optional[str] = None
    detail: str


class IngestAccepted(BaseModel):
    job_id: str
    status: str = "queued"
    status_url: str


class IngestResult(BaseModel):
    """Returned by POST /v2/ingest-video/?wait=true"""
    status: str
    video_id: int
    frames: Dict[str, Any]
    frames_failed: int
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: str = Field(description="queued | extracting | analyzing | summarizing | done | failed")
    stage: Optional[str] = None
    frames_done: int = 0
    frames_total: int = 0
    frames_failed: int = 0
    video_id: Optional[int] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list, description="Non-fatal issues, e.g. frames that failed and were skipped")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class VideoListItem(BaseModel):
    id: int
    filename: Optional[str] = None
    security_level: Optional[str] = None
    created_at: Optional[datetime] = None


class VideoList(BaseModel):
    items: List[VideoListItem]
    total: int
    limit: int
    offset: int


class VideoDetail(BaseModel):
    id: int
    filename: Optional[str] = None
    summary: Optional[str] = Field(None, description="Security-review summary (videos.video_description)")
    security_level: Optional[str] = None
    frame_count: int
    frames_described: int
    created_at: Optional[datetime] = None


class FrameItem(BaseModel):
    id: int
    frame_timestamp: Optional[int] = Field(None, description="Seconds from the start of the video")
    image_description: Optional[str] = None
    url: Optional[str] = None


class FrameList(BaseModel):
    items: List[FrameItem]
    total: int
    limit: int
    offset: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class AskMatch(BaseModel):
    frame_id: int
    frame_timestamp: Optional[int] = None
    image_description: Optional[str] = None
    similarity: float
    url: Optional[str] = None


class AskResponse(BaseModel):
    video_id: int
    question: str
    answer: Optional[str] = Field(None, description="LLM answer citing frame ids/timestamps; null if unavailable")
    matches: List[AskMatch]
    note: Optional[str] = Field(None, description="Set when the answer is missing or partial")
