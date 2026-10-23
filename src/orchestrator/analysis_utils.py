"""Best-effort parsing of model/OmniParser output. Nothing here raises on bad input."""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

RISK_LEVELS = ("low", "medium", "high", "critical")

# Ordered: first matching keyword group wins when the model gave no usable action_type
_ACTION_KEYWORDS = (
    ("file_transfer", ("file transfer", "upload", "download", "copying file", "copied file", "scp", "ftp", "export", "archive", "zip")),
    ("login", ("login", "log in", "sign in", "signed in", "password prompt", "credentials", "authenticat")),
    ("terminal", ("terminal", "powershell", "command prompt", "cmd.exe", "bash", "shell", "console window", "script")),
    ("admin_console", ("admin console", "administrator", "admin panel", "management console", "ssms", "database console", "server manager")),
    ("config_change", ("config", "settings", "registry", "group policy", "firewall", "clearing log", "event log")),
    ("browsing", ("browser", "chrome", "edge", "firefox", "web page", "website", "browsing")),
    ("idle", ("idle", "empty desktop", "no activity", "desktop only", "screen saver", "lock screen")),
)


def extract_json_object(text: Any) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from text that may be fenced (```json) or wrapped in prose."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    candidates: List[str] = []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))
    candidates += [text, text[text.find("{"):text.rfind("}") + 1]]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def normalize_risk_level(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip().lower() in RISK_LEVELS:
        return value.strip().lower()
    return None


def normalize_risk_score(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(100, score))


def level_for_score(score: Optional[int]) -> Optional[str]:
    if score is None:
        return None
    if score >= 75:
        return "critical"
    if score >= 50:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


def _regex_fallback(text: str) -> Tuple[Optional[int], Optional[str]]:
    # For truncated / invalid JSON: pick the first top-level-looking pairs
    score_match = re.search(r'"risk_score"\s*:\s*"?(\d{1,3})', text)
    level_match = re.search(r'"risk_level"\s*:\s*"(low|medium|high|critical)"', text, re.IGNORECASE)
    return (
        normalize_risk_score(score_match.group(1)) if score_match else None,
        level_match.group(1).lower() if level_match else None,
    )


def parse_risk(text: Any) -> Tuple[Optional[int], Optional[str]]:
    """(risk_score, risk_level) from reviewer/frame output; either may be None."""
    data = extract_json_object(text)
    if data is not None:
        return normalize_risk_score(data.get("risk_score")), normalize_risk_level(data.get("risk_level"))
    return _regex_fallback(text) if isinstance(text, str) else (None, None)


def derive_action_type(text: str) -> str:
    lowered = text.lower()
    for action, keywords in _ACTION_KEYWORDS:
        if any(k in lowered for k in keywords):
            return action
    return "other"


def parse_frame_analysis(text: Any) -> Dict[str, Any]:
    """risk_score / risk_level / action_type for one analyzed frame. Falls back to heuristics; never raises."""
    text = text if isinstance(text, str) else str(text)
    data = extract_json_object(text) or {}
    score, level = parse_risk(text)
    if level is None:
        level = level_for_score(score)
    action = data.get("action_type")
    if isinstance(action, str):
        action = re.sub(r"[^a-z0-9]+", "_", action.strip().lower()).strip("_")[:50]
    else:
        action = None
    if not action:
        action = derive_action_type(text)
    return {"risk_score": score, "risk_level": level, "action_type": action}


def elements_from_omniparser(parsed_content_list: Any) -> List[Dict[str, Any]]:
    """UI elements as {type, content, interactivity, bbox} with bbox as 0-1 ratios [x1, y1, x2, y2]."""
    elements = []
    for item in parsed_content_list or []:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        try:
            bbox = [round(float(v), 5) for v in bbox][:4] if bbox is not None else None
        except (TypeError, ValueError):
            bbox = None
        if bbox is not None and len(bbox) != 4:
            bbox = None
        if bbox is None:
            continue
        elements.append({
            "type": item.get("type"),
            "content": item.get("content"),
            "interactivity": item.get("interactivity"),
            "bbox": bbox,
        })
    return elements


def image_size(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    """(width, height) of encoded image bytes, or (None, None)."""
    try:
        import cv2
        import numpy as np
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None, None
        height, width = image.shape[:2]
        return int(width), int(height)
    except Exception:
        return None, None


def summary_text(video_description: Any) -> Optional[str]:
    """Prose summary from the reviewer output (JSON with a 'summary' key) or the raw text if it isn't JSON."""
    if not video_description:
        return video_description
    obj = extract_json_object(video_description)
    if isinstance(obj, dict) and isinstance(obj.get("summary"), str) and obj["summary"].strip():
        return obj["summary"].strip()
    return video_description
