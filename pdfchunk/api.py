"""FastAPI application: REST API + static web UI.

Run:  uvicorn pdfchunk.api:app --reload
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .config import settings
from .exporters import to_jsonl, to_markdown
from .models import Job, JobStatus, ProcessedDocument, ProcessingOptions
from .pipeline import process_pdf
from .store import Store

log = logging.getLogger("pdfchunk")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_CONCURRENT_JOBS = 2

app = FastAPI(title="PDF Intelligent Chunking & Summarization", version="1.0.0")
store = Store(settings.data_dir)
_job_slots = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_tasks: set[asyncio.Task] = set()


async def _run_job(job_id: str, filename: str, options: ProcessingOptions) -> None:
    async with _job_slots:
        store.update_job(job_id, status=JobStatus.running, stage="extracting")

        def progress(stage: str, frac: float, msg: str) -> None:
            store.update_job(job_id, stage=stage, progress=round(frac, 4), message=msg)

        try:
            doc = await process_pdf(store.upload_path(job_id), options, filename=filename,
                                    doc_id=job_id, progress=progress)
            store.save_result(doc)
            store.update_job(job_id, status=JobStatus.completed, stage="completed", progress=1.0,
                             message=f"{doc.stats.chunk_count} chunks")
        except Exception as e:  # surface any failure to the client
            log.exception("job %s failed", job_id)
            store.update_job(job_id, status=JobStatus.failed, stage="failed", error=str(e))


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "gemini_configured": settings.gemini_enabled,
        "model": settings.gemini_model,
        "embedding_model": settings.gemini_embedding_model,
    }


@app.post("/api/documents", response_model=Job, status_code=202)
async def upload(file: UploadFile = File(...), options: str | None = Form(None)) -> Job:
    try:
        opts = ProcessingOptions.model_validate_json(options) if options else ProcessingOptions()
    except ValidationError as e:
        raise HTTPException(422, f"Invalid options: {e.errors()}") from e

    if (opts.summarization.enabled or opts.chunking.strategy == "semantic") and not settings.gemini_enabled:
        raise HTTPException(400, "GEMINI_API_KEY is not configured on the server: disable summarization "
                                 "and use the 'structural' or 'fixed' strategy, or set the key.")
    if opts.chunking.overlap_tokens >= opts.chunking.max_tokens:
        raise HTTPException(422, "overlap_tokens must be smaller than max_tokens")

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Uploaded file is not a PDF")

    job_id = uuid.uuid4().hex[:12]
    filename = Path(file.filename or "document.pdf").name
    store.upload_path(job_id).write_bytes(data)
    job = store.create_job(job_id, filename)
    task = asyncio.create_task(_run_job(job_id, filename, opts))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return job


@app.get("/api/jobs", response_model=list[Job])
def list_jobs() -> list[Job]:
    return store.list_jobs()


@app.get("/api/jobs/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/documents/{doc_id}", response_model=ProcessedDocument)
def get_document(doc_id: str) -> ProcessedDocument:
    doc = store.get_result(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found (still processing or failed?)")
    return doc


@app.get("/api/documents/{doc_id}/export")
def export(doc_id: str, format: str = "markdown"):
    doc = store.get_result(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    stem = Path(doc.metadata.filename).stem
    if format in ("md", "markdown"):
        return PlainTextResponse(to_markdown(doc), media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="{stem}.summary.md"'})
    if format == "jsonl":
        return PlainTextResponse(to_jsonl(doc), media_type="application/x-ndjson",
                                 headers={"Content-Disposition": f'attachment; filename="{stem}.chunks.jsonl"'})
    if format == "json":
        return PlainTextResponse(doc.model_dump_json(indent=2), media_type="application/json",
                                 headers={"Content-Disposition": f'attachment; filename="{stem}.json"'})
    raise HTTPException(400, "format must be one of: markdown, json, jsonl")


@app.get("/api/documents/{doc_id}/pdf")
def get_pdf(doc_id: str) -> FileResponse:
    p = store.upload_path(doc_id)
    if not p.exists():
        raise HTTPException(404, "PDF not found")
    return FileResponse(p, media_type="application/pdf")


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str) -> None:
    job = store.get_job(doc_id)
    if job and job.status in (JobStatus.queued, JobStatus.running):
        raise HTTPException(409, "Document is still processing")
    if not store.delete(doc_id):
        raise HTTPException(404, "Document not found")


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
