import asyncio
import gc
import os
import shutil
import cv2
import base64
import logging
import uuid
import requests
from typing import Dict, Any, Optional
import json
import re
from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings
from postgresql import (
    insert_video, insert_frames, update_frames_with_embeddings, get_frames_by_video_id,
    update_video_with_security_review, list_videos, get_video_detail, list_frames,
    search_similar_frames, create_job, update_job, get_job, fail_interrupted_jobs,
)
# from agents.image_analyzer.image_analyzer_agent import get_image_analyzer_agent, image_analyzer_agent_run
from agents.security_reviewer.security_reviewer_agent import get_or_create_chat_completion_agent, security_reviewer_agent_run
from agents.chat_completion.chat_completion_agent import (
    get_chat_completion_agent_thread, chat_completion_agent_run, get_qa_agent, qa_agent_run
)
from semantic_kernel.connectors.ai.open_ai.settings.azure_open_ai_settings import AzureOpenAISettings
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
from retry_utils import retry_async
from errors import ApiError, api_error_response, api_errors, classify_exception, require_env, validation_error_response
from models import (
    AskRequest, AskResponse, ErrorResponse, FrameList, IngestAccepted, IngestResult,
    JobStatus, VideoDetail, VideoList,
)


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

load_dotenv()

@asynccontextmanager
async def lifespan(_app):
    # Jobs run in-process, so any job still non-terminal at startup was interrupted by a restart
    try:
        interrupted = await asyncio.to_thread(fail_interrupted_jobs)
        if interrupted:
            logging.warning(f"Marked {interrupted} interrupted job(s) as failed")
    except Exception as e:
        logging.warning(f"Could not clean up interrupted jobs: {e}")
    yield


orchestrator = FastAPI(lifespan=lifespan)
orchestrator.add_exception_handler(ApiError, api_error_response)
orchestrator.add_exception_handler(RequestValidationError, validation_error_response)

ERROR_RESPONSES = {
    code: {"model": ErrorResponse}
    for code in (400, 404, 422, 502, 503)
}

AzureOpenAISettings(
    embedding_deployment_name=os.getenv("AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME", "text-embedding-ada-002"),
    text_deployment_name=os.getenv("AZURE_AI_AGENT_CHAT_MODEL_DEPLOYMENT_NAME", "gpt-4.1"),
    chat_deployment_name=os.getenv("AZURE_AI_AGENT_CHAT_MODEL_DEPLOYMENT_NAME", "gpt-4.1"),
    api_key=os.getenv("AZURE_OPENAI_APIKEY"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_TEXT_EMBEDDING_API_VERSION", "2024-12-01-preview"),
)

# Bound concurrent OmniParser calls, and set explicit HTTP timeouts (seconds)
OMNIPARSER_SEMAPHORE = asyncio.Semaphore(max(1, int(os.getenv("MAX_CONCURRENT_OMNIPARSER", "2"))))
OMNIPARSER_TIMEOUT = (
    float(os.getenv("OMNIPARSER_CONNECT_TIMEOUT", "5")),
    float(os.getenv("OMNIPARSER_READ_TIMEOUT", "120")),
)
HEALTHCHECK_TIMEOUT = (
    float(os.getenv("HEALTHCHECK_CONNECT_TIMEOUT", "3")),
    float(os.getenv("HEALTHCHECK_READ_TIMEOUT", "5")),
)
EMBEDDING_BATCH_SIZE = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "16")))
# Ingestion jobs running the pipeline at once; extra jobs wait in status "queued"
JOB_SEMAPHORE = asyncio.Semaphore(max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "2"))))


def omniparser_base_url() -> str:
    return f"http://{os.getenv('OMNIPARSER_HOST')}:{os.getenv('OMNIPARSER_PORT')}"


@orchestrator.get("/")
def read_root():
    return {"message": "Orchestrator API is running", "ui": "/ui", "docs": "/docs"}


@orchestrator.get("/ui", include_in_schema=False)
async def auditor_ui():
    """Serve the self-contained auditor web UI"""
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html"),
                        media_type="text/html")

@orchestrator.get("/healthcheck")
async def healthcheck():
    """
    Check that the Omniparser API is reachable using its lightweight /health endpoint
    """
    try:
        response = await asyncio.to_thread(requests.get, f"{omniparser_base_url()}/health", timeout=HEALTHCHECK_TIMEOUT)
    except requests.RequestException as e:
        raise ApiError(503, "omniparser_unavailable", f"Omniparser API unreachable: {str(e)}", "healthcheck")
    if response.status_code == 200:        
        return {"message":"Orchestrator API and Omniparser API are running successfully"}
    else:
        raise ApiError(502, "upstream_omniparser_error", f"Omniparser API returned HTTP {response.status_code}: {response.text}", "healthcheck")


def _upload_blob_sync(connection_string: str, container_name: str, blob_name: str, file_content: Any, content_type: str) -> str:
    """Blocking Azure upload. file_content may be bytes or a binary file object."""
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        file_content,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type)
    )
    return blob_client.url


def _upload_file_sync(connection_string: str, container_name: str, blob_name: str, path: str, content_type: str) -> str:
    with open(path, "rb") as stream:
        return _upload_blob_sync(connection_string, container_name, blob_name, stream, content_type)


async def upload_to_blob_storage(file: UploadFile, container_name: str, blob_name: Optional[str] = None):
    """
    Upload a file to Azure Blob Storage

    Args:
        file: The file to upload
        container_name: The container to upload to
        blob_name: Optional custom blob name, defaults to the original filename if not provided

    Returns:
        dict: Information about the uploaded blob
    """
    require_env(["AZURE_STORAGE_CONNECTION_STRING"], "uploading_video")
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    with api_errors("uploading_video", "storage"):
        if not blob_name:
            blob_name = file.filename or f"{uuid.uuid4()}.jpg"
        file_content = await file.read()
        content_type = file.content_type or "application/octet-stream"
        blob_url = await asyncio.to_thread(
            _upload_blob_sync, connection_string, container_name, blob_name, file_content, content_type
        )
        return {
            "filename": file.filename,
            "blob_name": blob_name,
            "container_name": container_name,
            "size_bytes": len(file_content),
            "content_type": content_type,
            "url": blob_url
        }


async def upload_file_endpoint(file: UploadFile = File(...), container_name: str = "videos"):
    result = await upload_to_blob_storage(file, container_name)
    return result


async def store_video_frame_metadata(metadata: Dict[str, Any], job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Store video and frame metadata in Azure PostgreSQL database
    """
    stage = "storing_metadata"
    logging.info(f"Storing video and frame metadata: {metadata}")
    video_data = metadata.get("video", {})
    if not video_data:
        raise ApiError(400, "invalid_input", "No video metadata provided", stage, job_id)

    video_id = await asyncio.to_thread(insert_video, video_data)
    if not video_id:
        raise ApiError(503, "database_error", "Failed to insert video metadata", stage, job_id)

    frames_data = metadata.get("frames", {}).get("details", [])
    frames_result = await asyncio.to_thread(insert_frames, video_id, frames_data)
    if frames_result.get("status") == "error":
        raise ApiError(503, "database_error", f"Failed to insert frame metadata: {frames_result.get('message')}", stage, job_id)
    logging.info(f"Inserted frames for video ID {video_id}")
    return {
        "status": "success",
        "video_id": video_id,
        "frames": frames_result
    }


def _post_omniparser(base64_image: str) -> Dict[str, Any]:
    response = requests.post(
        f"{omniparser_base_url()}/parse/",
        json={"base64_image": base64_image},
        timeout=OMNIPARSER_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


async def _post_omniparser_limited(base64_image: str) -> Dict[str, Any]:
    # Semaphore is taken per attempt so backoff sleeps don't hold an OmniParser slot
    async with OMNIPARSER_SEMAPHORE:
        return await asyncio.to_thread(_post_omniparser, base64_image)


async def call_omniparser(base64_image: str) -> Dict[str, Any]:
    """
    Call the Omniparser API with a base64 encoded image (bounded concurrency, timeouts, retry on 429/5xx).
    Raises requests exceptions on failure; callers classify them.
    """
    json_response = await retry_async(lambda: _post_omniparser_limited(base64_image), description="Omniparser call")
    for item in json_response['parsed_content_list']:
        if 'bbox' in item:
            del item['bbox']
    return json_response


def _download_blob_sync(blob_service_client: BlobServiceClient, container_name: str, blob_name: str) -> bytes:
    blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)
    return blob_client.download_blob().readall()


async def _job_update(job_id: Optional[str], **fields):
    """Best-effort job progress update; never fails the pipeline."""
    if not job_id:
        return
    try:
        await asyncio.to_thread(update_job, job_id, **fields)
    except Exception as e:
        logging.warning(f"Could not update job {job_id} ({fields}): {e}")


def _embedding_service() -> AzureTextEmbedding:
    return AzureTextEmbedding(
        service_id="azure-text-embedding",
        api_key=os.getenv("AZURE_TEXT_EMBEDDING_APIKEY"),
        endpoint=os.getenv("AZURE_TEXT_EMBEDDING_ENDPOINT"),
        deployment_name=os.getenv("AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME")
    )


async def update_image_review_and_textvector_v2(preprocess_result: Dict[str, Any], job_id: Optional[str] = None):
    """
    Process images with Azure AI and update embeddings in the database.
    Using ChatCompletion Agent for image analysis.

    A frame that fails (OmniParser/GPT/embedding after retries) is logged and counted in
    frames_failed; the remaining frames still run. Frames that already have an
    image_description are skipped, so re-running a video is idempotent.
    """
    stage = "analyzing_frames"
    video_id = preprocess_result['video_id']
    logging.info(f"Updating image review and text vector for video ID: {video_id}")

    if os.getenv("CHAT_COMPLETION_ENABLE", "true").lower() != "true":
        raise ApiError(400, "feature_disabled", "ChatCompletion agent is disabled (CHAT_COMPLETION_ENABLE is not 'true')", stage, job_id)

    with api_errors(stage, "database", job_id):
        review_images = await asyncio.to_thread(get_frames_by_video_id, video_id)
    total = len(review_images)
    logging.info(f"Found {total} frames for video ID {video_id}")
    if total == 0:
        raise ApiError(404, "frames_not_found", f"No frames found for video ID {video_id}", stage, job_id)
    await _job_update(job_id, frames_total=total)

    with api_errors(stage, "storage", job_id):
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING", ""))
    azuretxtembedding = _embedding_service()
    with api_errors(stage, "openai", job_id):
        chat_completion_agent, chat_completion_agent_thread = await get_chat_completion_agent_thread(
            thread_id=f"image_analyzer_video_{video_id}_{uuid.uuid4().hex[:8]}"
        )
    logging.info(f"Using image analyzer agent: {chat_completion_agent}")

    def run_agent(image_b64, items):
        return retry_async(
            lambda: chat_completion_agent_run(
                agent=chat_completion_agent,
                base64_image=image_b64,
                omniparser_response=items,
                thread=chat_completion_agent_thread
            ),
            description="GPT image analysis",
        )

    # done = frames stored (or already described); failed = frames given up on
    stats = {"done": 0, "failed": 0, "last_error": None}
    pending = []  # (frame_id, description) awaiting embedding

    async def report():
        await _job_update(job_id, frames_done=stats["done"], frames_failed=stats["failed"])

    async def fail_frames(count: int, exc: BaseException, source: str, label: str):
        err = classify_exception(exc, stage, source, job_id)
        stats["failed"] += count
        stats["last_error"] = err.summary()
        logging.error(f"{label} failed ({count} frame(s)): {err.summary()}")
        await report()

    async def flush_pending():
        if not pending:
            return
        batch = list(pending)
        pending.clear()
        try:
            descriptions = [description for _, description in batch]
            embeddings = await retry_async(
                lambda: azuretxtembedding.generate_embeddings(descriptions),
                description="Embedding generation",
            )
        except Exception as e:
            await fail_frames(len(batch), e, "openai", "Embedding generation")
            return
        try:
            updates = [(frame_id, description, embeddings[i].tolist()) for i, (frame_id, description) in enumerate(batch)]
            updated = await asyncio.to_thread(update_frames_with_embeddings, updates)
        except Exception as e:
            await fail_frames(len(batch), e, "database", "Frame update")
            return
        stats["done"] += updated
        if updated != len(batch):
            await fail_frames(len(batch) - updated, RuntimeError(f"Bulk update touched {updated} of {len(batch)} frames"), "database", "Frame update")
            return
        logging.info(f"Updated {updated} frames with description and embedding")
        await report()

    for image in review_images:
        if image.get("image_description"):
            logging.info(f"Frame {image['id']} already has a description, skipping")
            stats["done"] += 1
            continue

        logging.info(f"Processing image {image['id']} with URL: {image.get('url', 'N/A')} and blob name: {image.get('blob_name', 'N/A')}")
        source = "storage"
        try:
            if len(chat_completion_agent_thread._chat_history.messages) > 33:
                logging.info("Trimming chat history to the last 33 messages")
                chat_completion_agent_thread._chat_history.messages = chat_completion_agent_thread._chat_history.messages[-33:]

            # read the image from Azure Blob Storage
            blob_data = await asyncio.to_thread(
                _download_blob_sync, blob_service_client, image["container_name"], image["blob_name"]
            )
            base64_image = base64.b64encode(blob_data).decode('utf-8')

            if os.getenv("OMNIPARSER_ENABLE", "true").lower() == "true":
                source = "omniparser"
                omniparser_response = await call_omniparser(base64_image)
                som_image_base64 = omniparser_response.pop('som_image_base64', None)
                item_list = []
                for item in omniparser_response['parsed_content_list']:
                    if 'bbox' in item:
                        del item['bbox']
                    item_list.append(f"type:{item['type']} interactivity:{item['interactivity']} content:{item['content']}")
                source = "openai"
                analysis = await run_agent(som_image_base64, set(item_list))
            else:
                source = "openai"
                analysis = await run_agent(base64_image, set())
            logging.info(f"Image analysis result: {analysis}")

            pending.append((image["id"], str(analysis)))
            if len(pending) >= EMBEDDING_BATCH_SIZE:
                await flush_pending()
        except Exception as e:
            await fail_frames(1, e, source, f"Frame {image['id']}")

    await flush_pending()
    await report()

    if stats["done"] == 0 and stats["failed"] > 0:
        raise ApiError(502, "all_frames_failed", f"All {stats['failed']} frames failed; last error: {stats['last_error']}", stage, job_id)
    if stats["failed"]:
        logging.warning(f"Video {video_id}: {stats['failed']} of {total} frames failed and were skipped")

    logging.info(f"Completed updating image review by ChatCompletion and text vector for video ID: {video_id}")
    return {**preprocess_result, "frames_failed": stats["failed"]}


async def update_video_review_and_text(preprocess_result: Dict[str, Any], job_id: Optional[str] = None):
    """
    Get image_descriptions from table, frames in database
    Use the image_descriptions as the input for agent which will generate video description with security review indicator
    """
    stage = "generating_summary"
    video_id = preprocess_result['video_id']
    logging.info(f"Updating video review and text for video ID: {video_id}")

    custom_prompt = """
                    Review the Remote Desktop session recording description
                    """

    if os.getenv("CHAT_COMPLETION_ENABLE", "true").lower() != "true":
        raise ApiError(400, "feature_disabled", "ChatCompletion agent is disabled (CHAT_COMPLETION_ENABLE is not 'true')", stage, job_id)

    with api_errors(stage, "openai", job_id):
        security_reviewer_agent = await get_or_create_chat_completion_agent()
    if not security_reviewer_agent:
        raise ApiError(502, "upstream_openai_error", "Security reviewer agent failed to initialize", stage, job_id)

    with api_errors(stage, "database", job_id):
        review_images = await asyncio.to_thread(get_frames_by_video_id, video_id)

    image_analysis_results = [
        {
            "id": img.get("id"),
            "frame_timestamp": img.get("frame_timestamp", 0),
            "image_description": img.get("image_description", "")
        }
        for img in review_images if img.get("image_description")
    ]
    if not image_analysis_results:
        raise ApiError(400, "no_descriptions", f"No image descriptions found for video ID {video_id}", stage, job_id)

    with api_errors(stage, "openai", job_id):
        security_review_result = await retry_async(
            lambda: security_reviewer_agent_run(
                agent=security_reviewer_agent,
                image_analysis_results=image_analysis_results,
                custom_prompt=custom_prompt,
            ),
            description="GPT security review",
        )
    if not security_review_result:
        raise ApiError(502, "upstream_openai_error", "Security reviewer returned no result", stage, job_id)
    logging.info(f"Security review result: {security_review_result}")

    # Extract the content from the result object based on expected return type
    if hasattr(security_review_result, 'content') and hasattr(security_review_result.content, 'content'):
        review_content = str(security_review_result.content.content)
    elif hasattr(security_review_result, 'content'):
        review_content = str(security_review_result.content)
    else:
        review_content = str(security_review_result)

    with api_errors(stage, "database", job_id):
        saved = await asyncio.to_thread(
            update_video_with_security_review, video_id, description=review_content,
            security_level=_parse_risk_level(review_content)
        )
    if not saved:
        raise ApiError(503, "database_error", f"Failed to save security review for video {video_id}", stage, job_id)
    logging.info(f"Successfully updated video {video_id} with security review")
    return preprocess_result


_RISK_LEVELS = {"low", "medium", "high", "critical"}


def _parse_risk_level(review_content: str) -> Optional[str]:
    """Extract risk_level from the reviewer output (JSON, possibly fenced/wrapped in text); None if absent."""
    text = review_content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates += [text, text[text.find("{"):text.rfind("}") + 1]]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        level = data.get("risk_level") if isinstance(data, dict) else None
        if isinstance(level, str) and level.strip().lower() in _RISK_LEVELS:
            return level.strip().lower()
    return None


def _required_ingest_env() -> list:
    names = ["AZURE_STORAGE_CONNECTION_STRING"]
    if os.getenv("CHAT_COMPLETION_ENABLE", "true").lower() == "true":
        names += [
            "AZURE_OPENAI_APIKEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
            "AZURE_TEXT_EMBEDDING_APIKEY", "AZURE_TEXT_EMBEDDING_ENDPOINT", "AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME",
        ]
    if os.getenv("OMNIPARSER_ENABLE", "true").lower() == "true":
        names += ["OMNIPARSER_HOST", "OMNIPARSER_PORT"]
    return names


def _remove_quietly(path: Optional[str]):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as e:
            logging.warning(f"Could not remove {path}: {e}")


async def run_ingest_pipeline(job_id: str, temp_filename: str, filename: str, content_type: Optional[str],
                              container_name: str, frames_container: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline for an already-saved upload, reporting progress on the job.
    Returns the result dict on success; on failure marks the job failed and raises ApiError.
    """
    stage = "queued"
    try:
        async with JOB_SEMAPHORE:
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

            stage = "uploading_video"
            await _job_update(job_id, status="extracting", stage=stage)
            # Unique blob name per job so re-uploads never overwrite; videos.filename keeps the original name
            blob_name = f"{job_id}/{os.path.basename(filename)}"
            with api_errors(stage, "storage", job_id):
                size_bytes = os.path.getsize(temp_filename)
                blob_url = await asyncio.to_thread(
                    _upload_file_sync, connection_string, container_name, blob_name, temp_filename,
                    content_type or "application/octet-stream"
                )
            video_blob = {
                "filename": filename,
                "blob_name": blob_name,
                "container_name": container_name,
                "size_bytes": size_bytes,
                "content_type": content_type or "application/octet-stream",
                "url": blob_url
            }

            stage = "extracting_frames"
            await _job_update(job_id, stage=stage)
            with api_errors(stage, "storage", job_id):
                frame_paths = await asyncio.to_thread(
                    split_video,
                    video_path=temp_filename,
                    blob_conn_str=connection_string,
                    output_dir=frames_container,
                )
            if not frame_paths:
                raise ApiError(400, "no_frames_extracted", "No frames could be extracted; the file may not be a readable video", stage, job_id)
            logging.info(f"Uploaded {len(frame_paths)} frames")
            _remove_quietly(temp_filename)

            stage = "storing_metadata"
            await _job_update(job_id, stage=stage, frames_total=len(frame_paths))
            stored = await store_video_frame_metadata({
                "video": video_blob,
                "frames": {"count": len(frame_paths), "container": frames_container, "details": frame_paths}
            }, job_id)
            await _job_update(job_id, video_id=stored["video_id"])

            stage = "analyzing_frames"
            await _job_update(job_id, status="analyzing", stage=stage)
            analyzed = await update_image_review_and_textvector_v2(stored, job_id)

            stage = "generating_summary"
            await _job_update(job_id, status="summarizing", stage=stage)
            result = await update_video_review_and_text(analyzed, job_id)

            await _job_update(job_id, status="done", stage="done")
            return {**result, "job_id": job_id}
    except Exception as e:
        err = classify_exception(e, stage, None, job_id)
        logging.error(f"Job {job_id} failed: {err.summary()}")
        await _job_update(job_id, status="failed", error=err.summary())
        raise err
    finally:
        _remove_quietly(temp_filename)


async def _run_ingest_in_background(*args, **kwargs):
    try:
        await run_ingest_pipeline(*args, **kwargs)
    except ApiError:
        pass  # already logged and recorded on the job


@orchestrator.post("/v2/ingest-video/", status_code=202, response_model=IngestAccepted, responses={
    200: {"model": IngestResult, "description": "Returned instead of 202 when ?wait=true: pipeline ran synchronously"},
    **ERROR_RESPONSES,
})
async def ingest_video_v2(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    container_name: str = Form("videos"),
    frames_container: str = Form("frames"),
    wait: bool = Query(False, description="Run the whole pipeline before responding (legacy synchronous behavior)"),
):
    """
    Save the upload, create a job and run the pipeline in the background (202 with job_id/status_url).
    Poll GET /jobs/{job_id} for progress. With ?wait=true the pipeline runs inline and the result is returned.
    """
    require_env(_required_ingest_env(), "validating_request")
    if not file.filename:
        raise ApiError(400, "invalid_input", "Uploaded file has no filename", "validating_request")

    temp_filename = None
    job_id = None
    try:
        with api_errors("saving_upload"):
            temp_dir = os.path.join(os.getcwd(), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_filename = os.path.join(temp_dir, f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
            await file.seek(0)
            await asyncio.to_thread(_save_upload_sync, file.file, temp_filename)
        with api_errors("creating_job", "database"):
            job_id = await asyncio.to_thread(create_job)
    except Exception:
        _remove_quietly(temp_filename)
        raise

    pipeline_args = (job_id, temp_filename, file.filename, file.content_type, container_name, frames_container)
    if wait:
        return JSONResponse(status_code=200, content=await run_ingest_pipeline(*pipeline_args))
    background_tasks.add_task(_run_ingest_in_background, *pipeline_args)
    return IngestAccepted(job_id=job_id, status="queued", status_url=f"/jobs/{job_id}")


@orchestrator.get("/jobs/{job_id}", response_model=JobStatus, responses=ERROR_RESPONSES)
async def get_job_status(job_id: str):
    """Progress of an ingestion job."""
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise ApiError(400, "invalid_job_id", f"'{job_id}' is not a valid job id", "get_job")
    with api_errors("get_job", "database", job_id):
        job = await asyncio.to_thread(get_job, job_id)
    if not job:
        raise ApiError(404, "job_not_found", f"Job {job_id} not found", "get_job", job_id)
    job["job_id"] = job.pop("id")
    job["warnings"] = []
    if job.get("frames_failed"):
        job["warnings"].append(
            f"{job['frames_failed']} of {job['frames_total']} frames failed analysis and were skipped"
        )
    return job


@orchestrator.get("/videos", response_model=VideoList, responses=ERROR_RESPONSES)
async def list_videos_endpoint(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    """Paginated list of videos, newest first."""
    with api_errors("list_videos", "database"):
        rows, total = await asyncio.to_thread(list_videos, limit, offset)
    return {"items": rows, "total": total, "limit": limit, "offset": offset}


async def _get_video_or_404(video_id: int, stage: str) -> Dict[str, Any]:
    with api_errors(stage, "database"):
        video = await asyncio.to_thread(get_video_detail, video_id)
    if not video:
        raise ApiError(404, "video_not_found", f"Video {video_id} not found", stage)
    return video


@orchestrator.get("/videos/{video_id}", response_model=VideoDetail, responses=ERROR_RESPONSES)
async def get_video_endpoint(video_id: int):
    """Video summary (security review), security level and frame counts."""
    video = await _get_video_or_404(video_id, "get_video")
    video["summary"] = video.pop("video_description")
    return video


@orchestrator.get("/videos/{video_id}/frames", response_model=FrameList, responses=ERROR_RESPONSES)
async def list_video_frames(video_id: int, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    """Paginated frames of a video ordered by timestamp."""
    await _get_video_or_404(video_id, "list_frames")
    with api_errors("list_frames", "database"):
        rows, total = await asyncio.to_thread(list_frames, video_id, limit, offset)
    return {"items": rows, "total": total, "limit": limit, "offset": offset}


@orchestrator.post("/videos/{video_id}/ask", response_model=AskResponse, responses=ERROR_RESPONSES)
async def ask_video(video_id: int, body: AskRequest):
    """
    Semantic search over a video's frame descriptions. Returns the top_k matching frames and an
    LLM answer citing them; if answer generation is unavailable, matches are returned with a note.
    """
    stage = "ask"
    await _get_video_or_404(video_id, stage)
    require_env(["AZURE_TEXT_EMBEDDING_APIKEY", "AZURE_TEXT_EMBEDDING_ENDPOINT", "AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME"], stage)

    with api_errors("embedding_question", "openai"):
        embedding_service = _embedding_service()
        embeddings = await retry_async(
            lambda: embedding_service.generate_embeddings([body.question]),
            description="Question embedding",
        )
        query_vector = embeddings[0].tolist()

    with api_errors("searching_frames", "database"):
        rows = await asyncio.to_thread(search_similar_frames, query_vector, None, body.top_k, video_id, True)

    matches = [
        {
            "frame_id": r["id"],
            "frame_timestamp": r.get("frame_timestamp"),
            "image_description": r.get("image_description"),
            "similarity": float(r["similarity"]),
            "url": r.get("url"),
        }
        for r in rows
    ]

    answer, note = None, None
    if not rows:
        note = "No analyzed frames with embeddings were found for this video."
    elif os.getenv("CHAT_COMPLETION_ENABLE", "true").lower() != "true" or not all(
        os.getenv(n) for n in ("AZURE_OPENAI_APIKEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    ):
        note = "Answer generation is not enabled/configured; returning matches only."
    else:
        try:
            qa_agent = await get_qa_agent()
            frames_for_llm = [{"id": r["id"], "frame_timestamp": r.get("frame_timestamp"), "image_description": r.get("image_description")} for r in rows]
            answer = await retry_async(
                lambda: qa_agent_run(qa_agent, body.question, frames_for_llm),
                description="GPT question answering",
            )
        except Exception as e:
            err = classify_exception(e, "answering_question", "openai")
            logging.warning(f"Answer generation failed: {err.summary()}")
            note = f"Answer generation failed ({err.error}); returning matches only."

    return {"video_id": video_id, "question": body.question, "answer": answer, "matches": matches, "note": note}


def _save_upload_sync(source, destination_path: str):
    with open(destination_path, "wb") as temp_file:
        shutil.copyfileobj(source, temp_file)


def split_video(video_path: str, blob_conn_str: str, output_dir: str=r"preprocess", interval=3):
    """
    Split a video into frames and upload directly to Azure Blob Storage.
    
    Args:
        video_path (str): Path to the input video file.
        blob_conn_str (str): Azure Blob Storage connection string.
        output_dir (str): Container name for the output frames.
        interval (int): Interval in seconds to split the video.
        
    Returns:
        list: List of paths to the saved frames.
    """
    frame_blob_paths = []
    local_temp_dir = None
    
    try:
        # Check if the video file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found.")

        # Create local temp directory for frames if needed
        # Unique per call so concurrent jobs don't delete each other's frames
        local_temp_dir = os.path.join(os.getcwd(), "temp_frames", uuid.uuid4().hex)
        os.makedirs(local_temp_dir, exist_ok=True)
        
        # Get video file name without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Initialize Azure Blob Storage client
        blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
        container_client = blob_service_client.get_container_client(output_dir)
        
        # Create container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Invalid frame rate: {fps}")
            
        frame_interval = int(fps * interval)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame every 'frame_interval' frames
            if frame_count % frame_interval == 0:
                # Calculate the time in seconds
                time_in_seconds = frame_count / fps
                
                # Create a filename with the time in seconds
                frame_basename = f"{video_name}_{int(time_in_seconds):0>10}.jpg"
                local_frame_path = os.path.join(local_temp_dir, frame_basename)
                
                # Save frame locally first
                cv2.imwrite(local_frame_path, frame)
                
                # Create blob name with logical hierarchy
                blob_name = f"{video_name}/{frame_basename}"
                
                # Upload to Azure Blob Storage
                blob_client = container_client.get_blob_client(blob_name)
                
                with open(local_frame_path, "rb") as frame_file:
                    # Upload with optimized settings for images
                    blob_client.upload_blob(
                        frame_file.read(),
                        overwrite=True,
                        content_settings=ContentSettings(
                            content_type="image/jpeg",
                            cache_control="max-age=86400"
                        )
                    )
                    frame_blob_paths.append(blob_client.url)

                
                # Explicitly release memory
                del frame
                gc.collect()

            frame_count += 1

        # Release the video capture object
        cap.release()
        shutil.rmtree(local_temp_dir, ignore_errors=True)

        # Return the list of saved frame paths
        return frame_blob_paths

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        # Return any frames that were successfully processed
        return frame_blob_paths
        
    finally:
        if local_temp_dir:
            shutil.rmtree(local_temp_dir, ignore_errors=True)
        # Ensure resources are released
        # if 'cap' in locals() and cap is not None:
        #     cap.release()
        gc.collect()


if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Only start the server if initialization was successful
    # Start the FastAPI server
    print("Starting Orchestrator API...")
    
    # Run the async initialization
    loop = asyncio.get_event_loop()
    init_success = loop.run_until_complete(healthcheck())

    # Start the FastAPI server after initialization
    if init_success:
        uvicorn.run(orchestrator, port=8082, log_level="info")
    else:
        print("Initialization failed, server not started")