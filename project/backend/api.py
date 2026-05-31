"""
FastAPI Backend for Manga Recap Video Pipeline
"""
import os
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging
import json
import shutil
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import config
from pipeline import MangaPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Manga Recap Video Pipeline",
    description="Transform manga chapters into video recaps",
    version="1.0.0"
)


def _normalize_status_payload(job_id: str, payload: dict) -> dict:
    """Normalize status payloads from Supabase/filesystem into one shape."""
    normalized = {
        "job_id": job_id,
        "status": payload.get("status", "pending"),
        "progress": payload.get("progress", 0.0),
    }

    for field in ("message", "pdf_filename", "total_pages", "total_panels", "error_message"):
        value = payload.get(field)
        if value is not None:
            normalized[field] = value

    if "error_message" not in normalized and payload.get("error"):
        normalized["error_message"] = payload["error"]

    return normalized


def _is_playable_video(path: Path) -> bool:
    """Return True when ffprobe can read the MP4 container metadata."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except FileNotFoundError:
        logger.warning("ffprobe not found; falling back to size-only video validation")
        return path.exists() and path.stat().st_size > 0
    except Exception as e:
        logger.warning(f"Video validation failed for {path}: {e}")
        return False


def _infer_filesystem_status(job_id: str) -> dict:
    """Infer job status from workspace artifacts."""
    job_workspace = WORKSPACE_DIR / job_id
    upload_pdf = UPLOAD_DIR / job_id / "manga.pdf"

    if not upload_pdf.exists() and not job_workspace.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    def has_files(path: Path) -> bool:
        return path.exists() and any(path.iterdir())

    videos_dir = job_workspace / "videos"
    audio_dir = job_workspace / "audio"
    panels_dir = job_workspace / "panels"
    pages_dir = job_workspace / "pages"

    video_parts = [videos_dir / f"part_{part}.mp4" for part in range(1, 5)]

    if all(_is_playable_video(path) for path in video_parts):
        status, progress = "completed", 1.0
    elif has_files(videos_dir):
        status, progress = "processing", 0.85
    elif has_files(audio_dir):
        status, progress = "processing", 0.75
    elif (job_workspace / "story_analysis.json").exists():
        status, progress = "processing", 0.5
    elif has_files(panels_dir):
        status, progress = "processing", 0.25
    elif has_files(pages_dir):
        status, progress = "processing", 0.15
    elif job_workspace.exists():
        status, progress = "processing", 0.05
    else:
        status, progress = "pending", 0.0

    return _normalize_status_payload(job_id, {"status": status, "progress": progress})

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase client
if config.SUPABASE_ENABLED:
    try:
        from supabase import create_client, Client
        supabase: Client = create_client(
            config.SUPABASE_URL,
            config.SUPABASE_SERVICE_ROLE_KEY
        )
    except Exception as e:
        logger.warning(f"Supabase not configured: {e}")
        supabase = None
else:
    logger.info(
        "Supabase disabled: missing a distinct service-role key; using local filesystem status only"
    )
    supabase = None

# Workspace
WORKSPACE_DIR = Path(config.WORKSPACE_DIR)
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Upload directory
UPLOAD_DIR = WORKSPACE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""


class ProcessRequest(BaseModel):
    llm_provider: str = "google"
    background_music_url: Optional[str] = None


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/jobs/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a manga PDF file."""
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Generate job ID
    import uuid
    job_id = str(uuid.uuid4())

    # Save file
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / "manga.pdf"

    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Validate PDF
    from modules import PDFProcessor
    processor = PDFProcessor()
    if not processor.validate_pdf(pdf_path):
        shutil.rmtree(job_dir)
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file")

    # Create job record in Supabase
    if supabase:
        try:
            supabase.table("jobs").insert({
                "id": job_id,
                "status": "pending",
                "pdf_filename": file.filename,
                "pdf_path": str(pdf_path)
            }).execute()
        except Exception as e:
            logger.error(f"Failed to create job record: {e}")

    logger.info(f"Uploaded PDF: {file.filename} -> job {job_id}")

    return {"job_id": job_id}


@app.post("/api/jobs/{job_id}/process")
async def process_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    request: ProcessRequest = ProcessRequest()
):
    """Start processing a job."""
    pdf_path = UPLOAD_DIR / job_id / "manga.pdf"

    # Get job info
    job = None
    if supabase:
        try:
            result = supabase.table("jobs").select("*").eq("id", job_id).maybe_single().execute()
            job = result.data if result else None
        except Exception as e:
            logger.warning(f"Could not fetch job from Supabase: {e}")

    if job is None:
        # Fallback: check filesystem (covers Supabase insert failures too)
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        logger.info(f"Job {job_id} not in DB, using filesystem fallback")
    else:
        if job["status"] not in ("pending", "failed"):
            raise HTTPException(status_code=400, detail="Job already processed or processing")
        pdf_path = Path(job["pdf_path"])

    # Update status to processing
    if supabase and job:
        try:
            supabase.table("jobs").update({
                "status": "processing",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", job_id).execute()
        except Exception as e:
            logger.warning(f"Could not update job status: {e}")

    # Start background processing
    background_tasks.add_task(
        run_pipeline,
        job_id,
        pdf_path,
        request.llm_provider
    )

    return {"status": "processing", "job_id": job_id}




def run_pipeline(job_id: str, pdf_path: Path, llm_provider: str):
    """Run the pipeline in background."""
    logger.info(f"Starting pipeline for job {job_id}")

    try:
        pipeline = MangaPipeline(llm_provider=llm_provider)
        results = pipeline.process(pdf_path, job_id)

        # Update job record
        if supabase:
            supabase.table("jobs").update({
                "status": "completed",
                "total_pages": results["phases"]["phase_1"]["pages"],
                "total_panels": results["phases"]["phase_1"]["panels"],
                "completed_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", job_id).execute()

            # Create video parts records
            for video in results["phases"]["phase_4"]["videos"]:
                # Get corresponding story part
                phase_2 = results["phases"]["phase_2"]
                audio_info = results["phases"]["phase_3"]["audio_files"][video["part"] - 1]

                supabase.table("video_parts").insert({
                    "job_id": job_id,
                    "part_number": video["part"],
                    "script": "Script placeholder",  # Would come from story_analysis.json
                    "selected_panels": [],  # Would come from story_analysis.json
                    "audio_path": audio_info["path"],
                    "audio_duration_ms": audio_info["duration_ms"],
                    "video_path": video["path"],
                    "status": "completed"
                }).execute()

        logger.info(f"Pipeline completed for job {job_id}")

    except Exception as e:
        logger.error(f"Pipeline failed for job {job_id}: {e}")

        # Update Supabase to "failed" so the frontend stops polling
        if supabase:
            try:
                supabase.table("jobs").update({
                    "status": "failed",
                    "error_message": str(e),
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", job_id).execute()
            except Exception as db_err:
                logger.error(f"Could not update job status in DB: {db_err}")

        # Filesystem fallback: write status.json so the status endpoint
        # can return "failed" even when Supabase is unavailable.
        try:
            job_workspace = WORKSPACE_DIR / job_id
            job_workspace.mkdir(parents=True, exist_ok=True)
            status_file = job_workspace / "status.json"
            import json as _json
            with open(status_file, "w") as _f:
                _json.dump(
                    {"job_id": job_id, "status": "failed", "progress": 0.0,
                     "error": str(e)},
                    _f
                )
        except Exception as write_err:
            logger.warning(f"Could not write status.json fallback: {write_err}")


@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get job processing status."""
    # Try Supabase first
    if supabase:
        try:
            result = supabase.table("jobs").select("*").eq("id", job_id).maybe_single().execute()
            job = result.data if result else None
            if job:
                progress = 0.0
                if job["status"] == "processing":
                    progress = 0.5
                elif job["status"] == "completed":
                    progress = 1.0
                return _normalize_status_payload(job_id, {
                    **job,
                    "progress": progress,
                })
        except Exception as e:
            logger.warning(f"Could not fetch status from Supabase: {e}")

    # Check for status file written by pipeline
    job_workspace = WORKSPACE_DIR / job_id
    status_file = job_workspace / "status.json"
    if status_file.exists():
        try:
            with open(status_file, encoding="utf-8") as f:
                file_status = _normalize_status_payload(job_id, json.load(f))
            inferred_status = _infer_filesystem_status(job_id)

            if inferred_status["progress"] > file_status["progress"]:
                merged = {**file_status, **inferred_status}
                if "message" in file_status:
                    merged["message"] = file_status["message"]
                if "error_message" in file_status:
                    merged["error_message"] = file_status["error_message"]
                return merged

            return file_status
        except Exception:
            pass

    return _infer_filesystem_status(job_id)



@app.get("/api/jobs/{job_id}/videos/{part_number}")
async def get_video(job_id: str, part_number: int, request: Request):
    """Stream a generated video with HTTP Range support for browser playback."""
    import os

    video_path = WORKSPACE_DIR / job_id / "videos" / f"part_{part_number}.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    if not _is_playable_video(video_path):
        raise HTTPException(status_code=409, detail="Video file is not ready or is incomplete")

    file_size = os.path.getsize(video_path)
    range_header = request.headers.get("range")

    def iter_file(path, start, end, chunk=1024 * 256):
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    if range_header:
        # Parse "bytes=start-end"
        range_val = range_header.strip().replace("bytes=", "")
        parts = range_val.split("-")
        start = int(parts[0])
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        content_length = end - start + 1

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Disposition": f'inline; filename="manga_recap_part_{part_number}.mp4"',
        }
        return StreamingResponse(
            iter_file(video_path, start, end),
            status_code=206,
            media_type="video/mp4",
            headers=headers,
        )

    # Full file response with Accept-Ranges header
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Disposition": f'inline; filename="manga_recap_part_{part_number}.mp4"',
    }
    return StreamingResponse(
        iter_file(video_path, 0, file_size - 1),
        status_code=200,
        media_type="video/mp4",
        headers=headers,
    )


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get full job details."""
    if supabase:
        job_result = supabase.table("jobs").select("*").eq("id", job_id).single().execute()
        job = job_result.data

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get video parts
        parts_result = supabase.table("video_parts").select("*").eq("job_id", job_id).execute()
        video_parts = parts_result.data or []

        return {
            "job": job,
            "video_parts": video_parts
        }

    raise HTTPException(status_code=503, detail="Database not configured")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
