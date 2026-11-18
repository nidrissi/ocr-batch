"""Validate cached Batch rows and assemble documents from their page requests."""

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from openai import OpenAI

from .errors import RemoteError
from .openai_backend import download_file
from .paths import output_paths
from .state import JobState, RunState, write_atomic_outputs
from .text import join_pages, page_chunk

log = logging.getLogger(__name__)


def _page_payload(row: dict[str, Any]) -> tuple[dict[str, Any], str, bool]:
    if (error := row.get("error")) is not None:
        raise ValueError(f"Batch request error: {error}")
    response = row.get("response")
    if not isinstance(response, dict):
        raise ValueError("missing or malformed Batch response")
    status = response.get("status_code")
    body = response.get("body")
    if type(status) is not int or not 200 <= status < 300:
        details = body.get("error") if isinstance(body, dict) else None
        raise ValueError(f"HTTP status {status!r}" + (f": {details}" if details else ""))
    if not isinstance(body, dict):
        raise ValueError("missing or malformed Responses body")
    if body.get("status") != "completed":
        raise ValueError(
            f"Responses status {body.get('status')!r}: "
            f"{body.get('incomplete_details') or body.get('error') or 'no details'}"
        )
    if body.get("error") is not None or body.get("incomplete_details") is not None:
        raise ValueError("completed Responses body has an error or incomplete details")
    if not isinstance(body.get("output"), list):
        raise ValueError("Responses output must be an array")

    parts: list[str] = []
    for item in body["output"]:
        if not isinstance(item, dict):
            raise ValueError("malformed Responses output item")
        if item.get("type") != "message":
            continue
        if item.get("status") not in (None, "completed"):
            raise ValueError("Responses contains an incomplete message")
        if not isinstance(item.get("content"), list):
            raise ValueError("malformed Responses message content")
        for part in item["content"]:
            if not isinstance(part, dict):
                raise ValueError("malformed Responses content part")
            if part.get("type") == "refusal":
                raise ValueError("model refusal")
            if part.get("type") == "output_text" and item.get("phase") in (None, "final_answer"):
                if not isinstance(part.get("text"), str):
                    raise ValueError("malformed output_text")
                parts.append(part["text"])

    try:
        payload = json.loads("".join(parts))
    except ValueError as exc:
        raise ValueError("missing or malformed structured output") from exc
    if (
        not isinstance(payload, dict)
        or payload.keys() != {"markdown", "blank"}
        or not isinstance(payload["markdown"], str)
        or not isinstance(payload["blank"], bool)
    ):
        raise ValueError("structured output must contain only markdown and blank")

    markdown = payload["markdown"]
    markdown.encode("utf-8")
    blank = payload["blank"]
    if blank != (not markdown.strip()):
        raise ValueError("blank indicator disagrees with Markdown content")
    return body, "" if blank else markdown, blank


@dataclass(frozen=True, slots=True)
class PageResult:
    path: Path
    offset: int
    job: JobState
    fingerprint: bytes
    error: str | None

    def read(self, index: int, handles: dict[Path, IO[bytes]]) -> dict[str, Any]:
        if (handle := handles.get(self.path)) is None:
            handle = handles[self.path] = self.path.open("rb")
        handle.seek(self.offset)
        row = json.loads(handle.readline())
        body, markdown, blank = _page_payload(row)
        return {
            "index": index,
            "job_id": self.job.job_id,
            "custom_id": row["custom_id"],
            "request_id": row["response"].get("request_id"),
            "markdown": markdown,
            "blank": blank,
            "response": body,
        }


def _read_cache(path: Path, job: JobState, pages: dict[str, dict[int, PageResult]]) -> bool:
    problems = False
    with path.open("rb") as handle:
        number = 0
        while True:
            offset = handle.tell()
            if not (line := handle.readline()):
                break
            number += 1
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("custom_id"), str):
                    raise ValueError("malformed Batch result row")
                request = job.requests.get(row["custom_id"])
                if request is None:
                    raise ValueError(f"unexpected request ID {row['custom_id']!r} for {job.job_id}")
                canonical = json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
            except ValueError as exc:
                log.warning("%s:%d: %s", path.name, number, exc)
                problems = True
                continue

            error = None
            try:
                _page_payload(row)
            except ValueError as exc:
                error = str(exc)
            fingerprint = hashlib.sha256(canonical).digest()
            doc_pages = pages[request.document_id]
            previous = doc_pages.get(request.page_index)
            if previous is not None:
                if previous.fingerprint == fingerprint and previous.job.job_id == job.job_id:
                    continue
                error = f"conflicting duplicate result for {row['custom_id']}"
            doc_pages[request.page_index] = PageResult(path, offset, job, fingerprint, error)

    return problems


def reconcile(client: OpenAI, state: RunState, *, force: bool) -> bool:
    """Return whether any request or batch failed. Cache each download independently."""
    pages: dict[str, dict[int, PageResult]] = defaultdict(dict)
    waiting: set[str] = set()
    problems = False
    download_error: RemoteError | None = None

    for job in state.jobs:
        if job.terminal:
            try:
                for file_id, path in (
                    (job.output_file, state.results_path(job.job_id)),
                    (job.error_file, state.errors_path(job.job_id)),
                ):
                    if file_id:
                        if not path.is_file():
                            download_file(client, file_id, path)
                        problems |= _read_cache(path, job, pages)
            except RemoteError as exc:
                log.error("%s", exc)
                download_error = exc
                job.fetched = False
            else:
                job.fetched = True
                problems |= job.status != "completed" or bool(job.failed_requests)
            state.save()

        if not job.fetched:
            waiting.update(request.document_id for request in job.requests.values())

    for document_id, document in state.documents.items():
        count = document.expected_pages
        if document_id in waiting or count is None:
            continue
        paths = output_paths(state.output_dir, Path(document.relative_path))
        results = pages[document_id]
        if not results and document.ocr_error:
            continue
        missing = [index + 1 for index in range(count) if index not in results]
        errors = [
            f"page {index + 1}: {result.error}"
            for index, result in sorted(results.items())
            if result.error
        ]
        if missing:
            errors.append(f"missing page results: {', '.join(map(str, missing))}")
        if errors:
            document.ocr_error = "; ".join(errors)
            document.ocr_written = False
            log.error("OCR failed: %s: %s", document.relative_path, document.ocr_error)
            continue

        configuration = results[0].job.configuration
        if any(result.job.configuration != configuration for result in results.values()):
            document.ocr_error = "inconsistent OCR configuration across page requests"
            document.ocr_written = False
            log.error("OCR failed: %s: %s", document.relative_path, document.ocr_error)
            continue

        if (
            not force
            and document.ocr_written
            and not document.ocr_pending
            and paths.ocr_md.is_file()
            and paths.ocr_json.is_file()
        ):
            continue

        handles: dict[Path, IO[bytes]] = {}
        try:
            assembled = [results[index].read(index, handles) for index in range(count)]
        finally:
            for handle in handles.values():
                handle.close()
        data = {
            "provider": "openai",
            "document_id": document_id,
            "sha256": document.sha256,
            "configuration": configuration,
            "pages": assembled,
        }
        markdown = join_pages(
            [page_chunk(page["index"] + 1, page["markdown"].strip()) for page in assembled]
        )
        write_atomic_outputs(
            {
                paths.ocr_json: json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                paths.ocr_md: markdown,
            }
        )
        document.complete_ocr(configuration)

    state.save()
    state.write_manifest()
    if download_error:
        raise download_error
    return problems or any(doc.ocr_pending or doc.ocr_error for doc in state.documents.values())
