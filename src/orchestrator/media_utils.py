"""Source-video access and ffmpeg clip extraction (blocking helpers; call via asyncio.to_thread)."""
import os
import shutil
import subprocess

CACHE_DIR = os.getenv("CLIP_CACHE_DIR", os.path.join(os.getcwd(), "temp", "clip_cache"))
FFMPEG_TIMEOUT_SECONDS = float(os.getenv("CLIP_FFMPEG_TIMEOUT_SECONDS", "120"))


def ffmpeg_path():
    return shutil.which("ffmpeg")


def source_cache_path(video_id: int, blob_name: str) -> str:
    return os.path.join(CACHE_DIR, f"src_{video_id}{os.path.splitext(blob_name)[1] or '.mp4'}")


def clip_cache_path(video_id: int, start: float, duration: float) -> str:
    return os.path.join(CACHE_DIR, f"clip_{video_id}_{int(start * 1000)}_{int(duration * 1000)}.mp4")


def download_blob_to_file_sync(connection_string: str, container_name: str, blob_name: str, destination: str) -> None:
    """Stream a blob to disk (never held in memory); the file appears at destination only when complete."""
    from azure.storage.blob import BlobServiceClient

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    partial = f"{destination}.{os.getpid()}.part"
    try:
        blob_client = BlobServiceClient.from_connection_string(connection_string) \
            .get_container_client(container_name).get_blob_client(blob_name)
        with open(partial, "wb") as out:
            blob_client.download_blob().readinto(out)
        os.replace(partial, destination)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def make_clip_sync(source: str, destination: str, start: float, duration: float) -> None:
    """Cut [start, start+duration] to browser-playable H.264/AAC MP4 with faststart. Raises RuntimeError on failure."""
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    partial = f"{destination}.{os.getpid()}.part.mp4"
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", source, "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart", partial,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS)
        if result.returncode != 0 or not os.path.exists(partial) or os.path.getsize(partial) == 0:
            raise RuntimeError(f"ffmpeg failed: {(result.stderr or '').strip()[-300:]}")
        os.replace(partial, destination)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg timed out")
    finally:
        if os.path.exists(partial):
            os.remove(partial)
