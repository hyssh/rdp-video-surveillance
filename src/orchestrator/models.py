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
    risk_score: Optional[int] = Field(None, description="Overall risk 0-100")
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
    risk_score: Optional[int] = Field(None, description="Overall risk 0-100")
    duration_s: Optional[int] = Field(None, description="Video duration in seconds")
    frame_count: int
    frames_described: int
    created_at: Optional[datetime] = None


class FrameItem(BaseModel):
    id: int
    frame_timestamp: Optional[int] = Field(None, description="Seconds from the start of the video")
    image_description: Optional[str] = None
    url: Optional[str] = None
    risk_score: Optional[int] = Field(None, description="0-100")
    risk_level: Optional[str] = Field(None, description="low | medium | high | critical")
    action_type: Optional[str] = Field(None, description="e.g. login, file_transfer, admin_console, browsing, terminal, config_change, idle, other")


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


class RiskDistribution(BaseModel):
    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0


class ActionTypeCount(BaseModel):
    action_type: str
    count: int


class RecentVideo(BaseModel):
    id: int
    filename: Optional[str] = None
    security_level: Optional[str] = None
    risk_score: Optional[int] = None
    created_at: Optional[datetime] = None


class Stats(BaseModel):
    videos_total: int
    frames_total: int
    frames_described: int
    high_risk_videos: int = Field(description="Videos with security_level high or critical")
    risk_distribution: RiskDistribution = Field(description="Videos by security_level")
    action_types: List[ActionTypeCount]
    recent_videos: List[RecentVideo]


class TimelinePoint(BaseModel):
    frame_id: int
    t: Optional[int] = Field(None, description="Seconds from video start")
    risk_score: Optional[int] = None
    risk_level: Optional[str] = None
    action_type: Optional[str] = None


class TimelineEvent(BaseModel):
    frame_id: int
    t: Optional[int] = None
    risk_level: Optional[str] = None
    action_type: Optional[str] = None
    description: Optional[str] = None


class Timeline(BaseModel):
    video_id: int
    filename: Optional[str] = None
    duration_s: Optional[int] = None
    interval_s: int = Field(description="Approximate seconds between sampled frames")
    summary: Optional[str] = None
    security_level: Optional[str] = None
    risk_score: Optional[int] = None
    points: List[TimelinePoint] = Field(description="One per frame, ordered by t")
    events: List[TimelineEvent] = Field(description="Frames with risk_score >= 50, ordered by t")


class FrameDetail(BaseModel):
    id: int
    video_id: Optional[int] = None
    frame_timestamp: Optional[int] = None
    image_description: Optional[str] = None
    risk_score: Optional[int] = None
    risk_level: Optional[str] = None
    action_type: Optional[str] = None
    image_url: str = Field(description="Relative URL of the image bytes, /frames/{id}/image")
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None


class UiElement(BaseModel):
    type: Optional[str] = None
    content: Optional[str] = None
    interactivity: Optional[bool] = None
    bbox: Optional[List[float]] = Field(None, description="[x1, y1, x2, y2] as 0-1 ratios of the frame size")


class FrameElements(BaseModel):
    frame_id: int
    width: Optional[int] = None
    height: Optional[int] = None
    elements: List[UiElement]
