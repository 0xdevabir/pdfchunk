"""File-backed storage for uploads, jobs and processed results."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import Job, JobStatus, ProcessedDocument


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.uploads = root / "uploads"
        self.results = root / "results"
        self.jobs_dir = root / "jobs"
        for d in (self.uploads, self.results, self.jobs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        for f in self.jobs_dir.glob("*.json"):
            try:
                job = Job.model_validate_json(f.read_text())
            except Exception:
                continue
            if job.status in (JobStatus.queued, JobStatus.running):
                job.status, job.error = JobStatus.failed, "Interrupted by server restart"
                self._write_job(job)
            self._jobs[job.id] = job

    # -------------------------------------------------------------- jobs

    def _write_job(self, job: Job) -> None:
        tmp = self.jobs_dir / f"{job.id}.json.tmp"
        tmp.write_text(job.model_dump_json(indent=2))
        tmp.replace(self.jobs_dir / f"{job.id}.json")

    def create_job(self, job_id: str, filename: str) -> Job:
        ts = now_iso()
        job = Job(id=job_id, filename=filename, created_at=ts, updated_at=ts)
        with self._lock:
            self._jobs[job_id] = job
            self._write_job(job)
        return job

    def update_job(self, job_id: str, **fields) -> Job:
        with self._lock:
            job = self._jobs[job_id].model_copy(update={**fields, "updated_at": now_iso()})
            self._jobs[job_id] = job
            # progress ticks are frequent; only persist state changes to disk
            if set(fields) - {"progress", "message"}:
                self._write_job(job)
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    # -------------------------------------------------------------- files

    def upload_path(self, job_id: str) -> Path:
        return self.uploads / f"{job_id}.pdf"

    def save_result(self, doc: ProcessedDocument) -> None:
        tmp = self.results / f"{doc.id}.json.tmp"
        tmp.write_text(doc.model_dump_json(indent=2))
        tmp.replace(self.results / f"{doc.id}.json")

    def get_result(self, doc_id: str) -> ProcessedDocument | None:
        p = self.results / f"{doc_id}.json"
        if not p.exists():
            return None
        return ProcessedDocument.model_validate(json.loads(p.read_text()))

    def delete(self, doc_id: str) -> bool:
        found = False
        for p in (self.results / f"{doc_id}.json", self.jobs_dir / f"{doc_id}.json", self.upload_path(doc_id)):
            if p.exists():
                p.unlink()
                found = True
        with self._lock:
            found = self._jobs.pop(doc_id, None) is not None or found
        return found
