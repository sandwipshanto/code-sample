from __future__ import annotations

import base64
import binascii
import sys
from pathlib import Path
from typing import List, Optional, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from app.adapters.model.model_client import ModelClient
from app.domain.enums import JobStatus
from app.domain.models import ExtractionJob, ReviewPackage
from app.observability.logging import JsonLogger
from app.pipeline.orchestrator import execute_job
from app.store.artifact_store import ArtifactStore
from app.store.job_store import JobStore

router = APIRouter()


class CreateJobRequest(BaseModel):
    source_file_id: str
    mode: str


class RunJobRequest(BaseModel):
    """Optional body for the run endpoint.

    ``pdf_path`` lets local callers point at a file on disk instead of
    requiring Drive download. ``pdf_base64`` lets remote callers upload
    a PDF directly to the API. ``drive_access_token`` lets callers pass
    an OAuth token so the API can download the file from Drive directly
    (used when no server-side Drive client is configured).
    """
    pdf_path: Optional[str] = None
    pdf_base64: Optional[str] = None
    drive_access_token: Optional[str] = None
    filename: Optional[str] = None


class ArtifactListResponse(BaseModel):
    artifacts: List[str]


def get_job_store(request: Request) -> JobStore:
    return cast(JobStore, request.app.state.job_store)


def get_artifact_store(request: Request) -> ArtifactStore:
    return cast(ArtifactStore, request.app.state.artifact_store)


def get_model_client(request: Request) -> Optional[ModelClient]:
    return getattr(request.app.state, "model_client", None)


def get_ocr_service(request: Request):
    return getattr(request.app.state, "ocr_service", None)


def get_drive_client(request: Request):
    return getattr(request.app.state, "drive_client", None)


@router.get("/jobs", response_model=List[ExtractionJob])
def list_jobs(
    store: JobStore = Depends(get_job_store),
) -> List[ExtractionJob]:
    return store.list_all()


@router.post("/jobs", response_model=ExtractionJob, status_code=201)
def create_job(
    payload: CreateJobRequest,
    store: JobStore = Depends(get_job_store),
) -> ExtractionJob:
    return store.create(source_file_id=payload.source_file_id, mode=payload.mode)


@router.get("/jobs/{job_id}", response_model=ExtractionJob)
def get_job(job_id: str, store: JobStore = Depends(get_job_store)) -> ExtractionJob:
    try:
        return store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.post("/jobs/{job_id}/run", response_model=ExtractionJob)
def run_job(
    request: Request,
    job_id: str,
    background_tasks: BackgroundTasks,
    payload: Optional[RunJobRequest] = None,
    store: JobStore = Depends(get_job_store),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
    model_client: Optional[ModelClient] = Depends(get_model_client),
    ocr_service=Depends(get_ocr_service),
    drive_client=Depends(get_drive_client),
) -> ExtractionJob:
    """Kick off the extraction pipeline for a job.

    The heavy work runs as a FastAPI ``BackgroundTask`` so the response
    returns immediately with the updated job status (``running``).
    """
    try:
        job = store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if job.status != JobStatus.QUEUED:
        raise HTTPException(
            status_code=409,
            detail=f"Job is already {job.status.value}, cannot start again",
        )

    if model_client is None:
        raise HTTPException(
            status_code=503,
            detail="No model client is configured — set GOOGLE_CLOUD_PROJECT or provide a model adapter",
        )

    pdf_path = _resolve_pdf_path(request=request, job_id=job_id, payload=payload)
    drive_access_token = payload.drive_access_token if payload else None
    if pdf_path is None and drive_client is None and not drive_access_token:
        raise HTTPException(
            status_code=400,
            detail="Provide pdf_path, pdf_base64, or drive_access_token when no drive client is configured",
        )

    logger = JsonLogger(stream=sys.stderr)

    background_tasks.add_task(
        execute_job,
        job_id=job_id,
        job_store=store,
        model_client=model_client,
        artifact_store=artifact_store,
        ocr_service=ocr_service,
        logger=logger,
        pdf_path=pdf_path,
        drive_client=drive_client,
        drive_access_token=drive_access_token,
        skip_ocr_threshold=request.app.state.settings.skip_ocr_threshold,
    )

    from app.domain.enums import JobStage

    updated = store.update_status(job_id, JobStatus.RUNNING, JobStage.INGEST)
    return updated


@router.post("/jobs/{job_id}/review-package", response_model=ExtractionJob)
def set_review_package(
    job_id: str,
    payload: ReviewPackage,
    store: JobStore = Depends(get_job_store),
) -> ExtractionJob:
    try:
        return store.set_review_package(job_id=job_id, review_package=payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.get("/jobs/{job_id}/artifacts", response_model=ArtifactListResponse)
def list_job_artifacts(
    job_id: str,
    store: JobStore = Depends(get_job_store),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
) -> ArtifactListResponse:
    _require_job(job_id, store)
    return ArtifactListResponse(artifacts=artifact_store.list_artifacts(job_id))


@router.get("/jobs/{job_id}/artifacts/{name}")
def get_job_artifact(
    job_id: str,
    name: str,
    store: JobStore = Depends(get_job_store),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
) -> Response:
    _require_job(job_id, store)
    if name not in artifact_store.list_artifacts(job_id):
        raise HTTPException(status_code=404, detail="Artifact not found")

    content = artifact_store.read_text(job_id, name)
    media_type = "application/json" if name.endswith(".json") else "text/plain"
    return Response(content=content, media_type=media_type)


def _resolve_pdf_path(request: Request, job_id: str, payload: Optional[RunJobRequest]) -> Optional[Path]:
    if payload is None:
        return None

    if payload.pdf_path and payload.pdf_base64:
        raise HTTPException(
            status_code=400,
            detail="Provide either pdf_path or pdf_base64, not both",
        )

    if payload.pdf_path:
        return Path(payload.pdf_path)

    if payload.pdf_base64:
        return _write_uploaded_pdf(
            root_dir=Path(request.app.state.settings.local_artifact_root),
            job_id=job_id,
            encoded_pdf=payload.pdf_base64,
            filename=payload.filename,
        )

    return None


def _write_uploaded_pdf(
    root_dir: Path,
    job_id: str,
    encoded_pdf: str,
    filename: Optional[str],
) -> Path:
    try:
        pdf_bytes = base64.b64decode(encoded_pdf, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid pdf_base64 payload") from exc

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="pdf_base64 payload was empty")

    safe_name = Path(filename or "uploaded.pdf").name or "uploaded.pdf"
    if Path(safe_name).suffix.lower() != ".pdf":
        safe_name = f"{Path(safe_name).stem or 'uploaded'}.pdf"

    upload_dir = root_dir / job_id / "inputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / safe_name
    upload_path.write_bytes(pdf_bytes)
    return upload_path


def _require_job(job_id: str, store: JobStore) -> ExtractionJob:
    try:
        return store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
