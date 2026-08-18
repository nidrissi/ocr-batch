"""Everything that talks to the Mistral API.

No function here constructs a client: the caller passes one in, so the whole
module can be driven by a fake in tests.
"""

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import RemoteError, UploadError
from .state import TERMINAL_STATUSES

log = logging.getLogger(__name__)


# The slice of the Mistral SDK this module uses, as a protocol: nothing here
# constructs a client, and tests drive the whole module with a fake.


class FilesAPI(Protocol):
    def upload(self, *, file: Any, purpose: Any, visibility: Any, expiry: Any) -> Any: ...

    def get_signed_url(self, *, file_id: str, expiry: int) -> Any: ...

    def download(self, *, file_id: str) -> Any: ...

    def delete(self, *, file_id: str) -> Any: ...


class JobsAPI(Protocol):
    def create(
        self,
        *,
        requests: Any,
        model: Any,
        endpoint: Any,
        timeout_hours: Any,
        metadata: Any,
    ) -> Any: ...

    def get(self, *, job_id: str) -> Any: ...

    def cancel(self, *, job_id: str) -> Any: ...


class BatchAPI(Protocol):
    # Read-only members: a mutable protocol attribute would be invariant and
    # reject both the real SDK class and the test fake.
    @property
    def jobs(self) -> JobsAPI: ...


class MistralClient(Protocol):
    @property
    def files(self) -> FilesAPI: ...

    @property
    def batch(self) -> BatchAPI: ...


# The API caps signed-URL lifetime at 168h (7 days).
MAX_URL_EXPIRY_HOURS = 168

POLL_INITIAL_SECONDS = 5.0
POLL_MAX_SECONDS = 300.0
POLL_BACKOFF = 1.5

DOWNLOAD_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class OcrOptions:
    """Per-request OCR settings, all exposed as CLI flags."""

    include_blocks: bool = True
    confidence_granularity: str | None = "block"
    table_format: str = "markdown"
    extract_header: bool = False
    extract_footer: bool = False
    include_image_base64: bool = False


@dataclass(frozen=True, slots=True)
class Upload:
    custom_id: str
    file_id: str
    signed_url: str


@dataclass(frozen=True, slots=True)
class Progress:
    total: int = 0
    completed: int = 0
    succeeded: int = 0
    failed: int = 0

    def __str__(self) -> str:
        return f"{self.succeeded}/{self.total} done, {self.failed} failed"


# ----------------------------------------------------------------------
# Uploads
# ----------------------------------------------------------------------


def upload_document(
    client: MistralClient,
    *,
    custom_id: str,
    path: Path,
    url_expiry_hours: int,
    upload_expiry_hours: int,
) -> Upload:
    """Upload one PDF and mint a signed URL for it.

    `upload_expiry_hours` is a server-side backstop: if we never manage to
    delete this file, the API drops it anyway.
    """
    try:
        uploaded = client.files.upload(
            file={"file_name": path.name, "content": path.read_bytes()},
            purpose="ocr",
            visibility="user",
            expiry=upload_expiry_hours,
        )
    except Exception as exc:
        raise UploadError(f"upload of {path} failed: {exc}") from exc

    try:
        signed = client.files.get_signed_url(
            file_id=uploaded.id,
            expiry=min(url_expiry_hours, MAX_URL_EXPIRY_HOURS),
        )
    except Exception as exc:
        # The file exists remotely even though we can't use it -- hand the id
        # back so the caller records it and cleanup can still delete it.
        raise UploadError(f"signed URL for {path} failed: {exc}", file_id=uploaded.id) from exc

    return Upload(custom_id=custom_id, file_id=uploaded.id, signed_url=signed.url)


def upload_documents(
    client: MistralClient,
    items: Sequence[tuple[str, Path]],
    *,
    workers: int,
    url_expiry_hours: int,
    upload_expiry_hours: int,
    on_upload: Callable[[Upload], None],
    on_orphan: Callable[[str], None],
) -> list[Upload]:
    """Upload documents concurrently, reporting each success as it lands.

    `on_upload` and `on_orphan` are called on the calling thread, so the caller
    can persist state without locking.
    """
    uploads: list[Upload] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                upload_document,
                client,
                custom_id=custom_id,
                path=path,
                url_expiry_hours=url_expiry_hours,
                upload_expiry_hours=upload_expiry_hours,
            ): custom_id
            for custom_id, path in items
        }

        try:
            for done, future in enumerate(as_completed(futures), 1):
                try:
                    upload = future.result()
                except UploadError as exc:
                    if exc.file_id is not None:
                        on_orphan(exc.file_id)

                    raise

                uploads.append(upload)
                on_upload(upload)

                log.info("uploaded %d/%d", done, len(futures))
        except BaseException:
            for future in futures:
                future.cancel()

            raise

    return uploads


# ----------------------------------------------------------------------
# Batch jobs
# ----------------------------------------------------------------------


def build_request(custom_id: str, document_url: str, options: OcrOptions) -> dict[str, Any]:
    """Build one `/v1/ocr` batch row.

    Mistral Annotations are deliberately unused: no `document_annotation_format`
    and no `bbox_annotation_format`.
    """
    body: dict[str, Any] = {
        "document": {"type": "document_url", "document_url": document_url},
        "include_blocks": options.include_blocks,
        "include_image_base64": options.include_image_base64,
        "extract_header": options.extract_header,
        "extract_footer": options.extract_footer,
        "table_format": options.table_format,
    }

    if options.confidence_granularity is not None:
        body["confidence_scores_granularity"] = options.confidence_granularity

    return {"custom_id": custom_id, "body": body}


def submit_jobs(
    client: MistralClient,
    requests: Sequence[dict[str, Any]],
    *,
    model: str,
    timeout_hours: int,
    batch_size: int,
    metadata: dict[str, str],
    on_job: Callable[[str, list[str]], None],
) -> list[str]:
    """Create one batch job per chunk of requests, reporting each id immediately.

    Chunking keeps a large corpus under any per-job request cap, and `on_job`
    exists so a job id reaches disk before the next one is created.
    """
    job_ids: list[str] = []

    for start in range(0, len(requests), max(1, batch_size)):
        chunk = list(requests[start : start + max(1, batch_size)])

        try:
            job = client.batch.jobs.create(
                requests=chunk,
                model=model,
                endpoint="/v1/ocr",
                timeout_hours=timeout_hours,
                metadata=metadata,
            )
        except Exception as exc:
            raise RemoteError(f"could not create batch job: {exc}") from exc

        custom_ids = [str(request["custom_id"]) for request in chunk]

        job_ids.append(job.id)
        on_job(job.id, custom_ids)

        log.info("batch job %s: %d requests", job.id, len(chunk))

    return job_ids


def get_job(client: MistralClient, job_id: str) -> Any:
    try:
        return client.batch.jobs.get(job_id=job_id)
    except Exception as exc:
        raise RemoteError(f"could not read batch job {job_id}: {exc}") from exc


def refresh_jobs(client: MistralClient, job_ids: Sequence[str]) -> dict[str, Any]:
    return {job_id: get_job(client, job_id) for job_id in job_ids}


def aggregate(jobs: dict[str, Any]) -> Progress:
    return Progress(
        total=sum(getattr(job, "total_requests", 0) or 0 for job in jobs.values()),
        completed=sum(getattr(job, "completed_requests", 0) or 0 for job in jobs.values()),
        succeeded=sum(getattr(job, "succeeded_requests", 0) or 0 for job in jobs.values()),
        failed=sum(getattr(job, "failed_requests", 0) or 0 for job in jobs.values()),
    )


def wait_for_jobs(
    client: MistralClient,
    job_ids: Sequence[str],
    *,
    on_update: Callable[[dict[str, Any], Progress], None],
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll until every job reaches a terminal status.

    Backs off from 5s to 5min and only reports when something actually changed,
    so a 24h job costs a few dozen requests and a few dozen log lines.
    """
    delay = POLL_INITIAL_SECONDS
    previous: tuple[Any, ...] | None = None

    while True:
        jobs = refresh_jobs(client, job_ids)
        progress = aggregate(jobs)
        snapshot = (progress, tuple(sorted(job.status for job in jobs.values())))

        if snapshot != previous:
            on_update(jobs, progress)
            previous = snapshot
            delay = POLL_INITIAL_SECONDS

        if all(job.status in TERMINAL_STATUSES for job in jobs.values()):
            return jobs

        sleep(delay)
        delay = min(delay * POLL_BACKOFF, POLL_MAX_SECONDS)


def cancel_jobs(client: MistralClient, job_ids: Sequence[str]) -> None:
    for job_id in job_ids:
        try:
            client.batch.jobs.cancel(job_id=job_id)
        except Exception as exc:
            log.warning("could not cancel job %s: %s", job_id, exc)


# ----------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------


def download_file(client: MistralClient, file_id: str, destination: Path) -> Path:
    """Stream a remote file to disk; the payload is never held in memory."""
    try:
        response = client.files.download(file_id=file_id)
    except Exception as exc:
        raise RemoteError(f"could not download {file_id}: {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with destination.open("wb") as out:
            for chunk in response.iter_bytes(DOWNLOAD_CHUNK):
                out.write(chunk)
    finally:
        response.close()

    return destination


def delete_files(client: MistralClient, file_ids: Sequence[str]) -> dict[str, str]:
    """Delete remote files; returns `{file_id: error}` for the ones that failed."""
    failures: dict[str, str] = {}

    for file_id in file_ids:
        try:
            client.files.delete(file_id=file_id)
        except Exception as exc:
            failures[file_id] = str(exc)

    return failures
