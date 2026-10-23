#!/usr/bin/env python3
"""Seed the database with 3 synthetic RDP sessions for the demo UI (no Azure video analysis needed).

    python scripts/seed_demo_data.py            # render frames + insert missing sessions (idempotent by filename)
    python scripts/seed_demo_data.py --reset    # delete the demo videos and recreate them
    python scripts/seed_demo_data.py --render-only   # only (re)generate the PNGs, no database access

Frames are stored with container_name='demo-local' and blob_name='<session_slug>/frame_XX.png';
the PNGs live in src/orchestrator/static/demo_assets/. Database settings (PG*) come from the repo-root .env.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ORCH_DIR = os.path.join(REPO, "src", "orchestrator")
ASSETS_DIR = os.path.join(ORCH_DIR, "static", "demo_assets")
CONTAINER = "demo-local"
INTERVAL_S = 3
EMBED_DIM = 1536

sys.path.insert(0, HERE)
from demo_render import H, W, render_frame, save_png  # noqa: E402
from demo_sessions import DATE, SESSIONS  # noqa: E402


def risk_level(score: int) -> str:
    return "critical" if score >= 75 else "high" if score >= 50 else "medium" if score >= 25 else "low"


def hhmmss(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def render_session(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Render all frames of a session to PNG and return per-frame records (with UI elements)."""
    start = datetime(2025, 10, 14, *session["start"])
    out = []
    for i, spec in enumerate(session["frames"]):
        t = i * INTERVAL_S
        clock = (start + timedelta(seconds=t)).strftime("%I:%M %p").lstrip("0")
        img, elements = render_frame(spec, clock, DATE)
        name = f"frame_{i:02d}.png"
        size = save_png(img, os.path.join(ASSETS_DIR, session["slug"], name))
        out.append({"index": i, "t": t, "blob_name": f"{session['slug']}/{name}", "bytes": size, "elements": elements,
                    "description": spec["desc"], "risk_score": spec["risk"], "risk_level": risk_level(spec["risk"]), "action_type": spec["action"]})
    return out


def pseudo_embedding(text: str) -> List[float]:
    """Deterministic hashed bag-of-words vector (unit length). Placeholder, not semantically comparable to Azure embeddings."""
    vec = [0.0] * EMBED_DIM
    for token in re.findall(r"[a-z0-9_.$-]+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0 if (h >> 100) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Embedder:
    """Azure OpenAI embeddings (same service as app.py) when configured, otherwise pseudo-embeddings."""

    def __init__(self, force_pseudo: bool) -> None:
        self.azure = False
        self.api_key = os.getenv("AZURE_TEXT_EMBEDDING_APIKEY")
        self.endpoint = os.getenv("AZURE_TEXT_EMBEDDING_ENDPOINT")
        self.deployment = os.getenv("AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME", "text-embedding-ada-002")
        if force_pseudo:
            return
        if not (self.api_key and self.endpoint):
            print("Embeddings: AZURE_TEXT_EMBEDDING_* not set -> using deterministic PSEUDO-embeddings "
                  "(the /ask similarity ranking will not be meaningful; set the Azure variables and rerun with --reset).")
            return
        try:
            import semantic_kernel  # noqa: F401
            self.azure = True
        except ImportError:
            print("Embeddings: semantic-kernel is not installed -> using PSEUDO-embeddings (pip install -r src/orchestrator/requirements.txt).")

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
        service = AzureTextEmbedding(service_id="azure-text-embedding", api_key=self.api_key, endpoint=self.endpoint, deployment_name=self.deployment)
        vectors: List[List[float]] = []
        for i in range(0, len(texts), 16):
            vectors += [v.tolist() for v in await service.generate_embeddings(texts[i:i + 16])]
        return vectors

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.azure:
            try:
                vectors = asyncio.run(self._embed(texts))
                if all(len(v) == EMBED_DIM for v in vectors):
                    return vectors
                print(f"Embeddings: unexpected dimension {len(vectors[0])} (column is {EMBED_DIM}) -> using PSEUDO-embeddings.")
            except Exception as exc:  # noqa: BLE001 - any Azure/network failure falls back
                print(f"Embeddings: Azure call failed ({exc.__class__.__name__}: {exc}) -> using PSEUDO-embeddings.")
            self.azure = False
        return [pseudo_embedding(t) for t in texts]


def connect_db():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
    sys.path.insert(0, ORCH_DIR)
    try:
        import postgresql  # noqa: F401 - importing runs init_connection_pool() and initialize_schemas()
        from postgresql.connection import DatabaseManager
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Could not connect to PostgreSQL ({exc.__class__.__name__}: {exc}). Check PGHOST/PGUSER/PGPASSWORD/PGDATABASE in .env.")
    return DatabaseManager


def ensure_columns(DatabaseManager) -> None:
    """Add the demo-UI columns if the orchestrator has not created them yet (idempotent, same DDL as the contract)."""
    statements = [
        "ALTER TABLE frames ADD COLUMN IF NOT EXISTS risk_score INTEGER",
        "ALTER TABLE frames ADD COLUMN IF NOT EXISTS risk_level VARCHAR(20)",
        "ALTER TABLE frames ADD COLUMN IF NOT EXISTS action_type VARCHAR(50)",
        "ALTER TABLE frames ADD COLUMN IF NOT EXISTS ui_elements JSONB",
        "ALTER TABLE frames ADD COLUMN IF NOT EXISTS frame_width INTEGER",
        "ALTER TABLE frames ADD COLUMN IF NOT EXISTS frame_height INTEGER",
        "ALTER TABLE videos ADD COLUMN IF NOT EXISTS duration_s INTEGER",
        "ALTER TABLE videos ADD COLUMN IF NOT EXISTS risk_score INTEGER",
    ]
    with DatabaseManager() as cur:
        for sql in statements:
            cur.execute(sql)


def review_document(session: Dict[str, Any], frames: List[Dict[str, Any]], frame_ids: List[int]) -> str:
    observations = []
    for index, description, context, evidence, confidence in session["key_frames"]:
        f = frames[index]
        observations.append({
            "frame_id": frame_ids[index], "frame_timestamp": hhmmss(f["t"]), "description": description, "context": context,
            "evidence": evidence, "confidence": confidence, "risk_score": f["risk_score"], "risk_level": f["risk_level"],
        })
    return json.dumps({
        "summary": session["summary"], "manual_review": session["manual_review"], "manual_review_reason": session["manual_review_reason"],
        "security_risk_score": session["security_risk_score"], "risk_score": session["risk_score"], "risk_level": session["security_level"],
        "analysis_results": {"key_observations": observations},
    })


def seed_session(DatabaseManager, session: Dict[str, Any], frames: List[Dict[str, Any]], embedder: Embedder, reset: bool) -> str:
    from psycopg2.extras import Json

    with DatabaseManager() as cur:
        cur.execute("SELECT id FROM videos WHERE filename = %s AND container_name = %s", (session["filename"], CONTAINER))
        existing = [row[0] for row in cur.fetchall()]
        if existing and not reset:
            return f"exists (video id {existing[0]}), skipped; use --reset to recreate"
        if existing:
            cur.execute("DELETE FROM videos WHERE id = ANY(%s)", (existing,))

    vectors = embedder.embed([f["description"] for f in frames])
    created = datetime.now() - timedelta(hours=session["age_hours"])
    with DatabaseManager() as cur:
        cur.execute(
            """
            INSERT INTO videos (filename, blob_name, container_name, size_bytes, content_type, url, security_level, risk_score, duration_s, created_at)
            VALUES (%s, %s, %s, NULL, 'video/mp4', NULL, %s, %s, %s, %s) RETURNING id
            """,
            (session["filename"], f"{session['slug']}/{session['filename']}", CONTAINER, session["security_level"], session["risk_score"],
             len(frames) * INTERVAL_S, created),
        )
        video_id = cur.fetchone()[0]
        frame_ids = []
        for f, vec in zip(frames, vectors):
            cur.execute(
                """
                INSERT INTO frames (video_id, blob_name, container_name, url, frame_timestamp, image_description, embedding,
                                    risk_score, risk_level, action_type, ui_elements, frame_width, frame_height, created_at)
                VALUES (%s, %s, %s, NULL, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (video_id, f["blob_name"], CONTAINER, f["t"], f["description"], "[" + ",".join(repr(x) for x in vec) + "]",
                 f["risk_score"], f["risk_level"], f["action_type"], Json(f["elements"]), W, H, created + timedelta(seconds=f["t"])),
            )
            frame_ids.append(cur.fetchone()[0])
        cur.execute("UPDATE videos SET video_description = %s WHERE id = %s", (review_document(session, frames, frame_ids), video_id))
    return f"created video id {video_id} with {len(frames)} frames"


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed 3 synthetic RDP sessions for the demo UI.")
    parser.add_argument("--reset", action="store_true", help="delete the demo videos (container demo-local) and recreate them")
    parser.add_argument("--render-only", action="store_true", help="only generate the frame PNGs; do not touch the database")
    parser.add_argument("--pseudo-embeddings", action="store_true", help="skip Azure and use deterministic pseudo-embeddings")
    args = parser.parse_args()

    rendered: Dict[str, List[Dict[str, Any]]] = {}
    for session in SESSIONS:
        rendered[session["slug"]] = render_session(session)
        total_kb = sum(f["bytes"] for f in rendered[session["slug"]]) / 1024
        print(f"Rendered {len(rendered[session['slug']])} frames for {session['filename']} ({total_kb:.0f} KB) -> {os.path.relpath(os.path.join(ASSETS_DIR, session['slug']), REPO)}")
    if args.render_only:
        return 0

    DatabaseManager = connect_db()
    ensure_columns(DatabaseManager)
    embedder = Embedder(args.pseudo_embeddings)
    for session in SESSIONS:
        try:
            result = seed_session(DatabaseManager, session, rendered[session["slug"]], embedder, args.reset)
        except Exception as exc:  # noqa: BLE001
            print(f"{session['filename']}: FAILED ({exc.__class__.__name__}: {exc})")
            return 1
        print(f"{session['filename']}: {result}")
    print("Done. Open http://localhost:8082/ui")
    return 0


if __name__ == "__main__":
    sys.exit(main())
