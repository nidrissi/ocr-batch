"""Split a batch JSONL result back into per-document files.

Read line by line: with block-level output the JSONL is the largest artifact of
a run and must never be loaded whole.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import output_paths
from .state import RunState
from .text import join_pages, page_chunk

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SplitSummary:
    written: int = 0
    failed: int = 0
    skipped: int = 0
    unknown: int = 0
    malformed: int = 0

    def __str__(self) -> str:
        parts = [f"{self.written} written"]

        for count, label in (
            (self.failed, "failed"),
            (self.skipped, "already done"),
            (self.unknown, "unknown id"),
            (self.malformed, "malformed"),
        ):
            if count:
                parts.append(f"{count} {label}")

        return ", ".join(parts)


def render_markdown(body: dict[str, Any]) -> str:
    """Render an OCR response body as page-separated markdown."""
    pages = sorted(
        body.get("pages") or [],
        key=lambda page: page.get("index", 0),
    )

    return join_pages(
        [
            page_chunk(page.get("index", 0) + 1, (page.get("markdown") or "").strip())
            for page in pages
        ]
    )


def _failure_reason(row: dict[str, Any]) -> str:
    for key in ("error", "errors"):
        if row.get(key):
            return json.dumps(row[key], ensure_ascii=False)

    response = row.get("response") or {}
    status = response.get("status_code")

    if status is not None and response.get("body"):
        return f"HTTP {status}: {json.dumps(response['body'], ensure_ascii=False)}"

    return f"no response body (status {status})" if status else "no response body"


def split_results(
    results_path: Path,
    state: RunState,
    *,
    force: bool = False,
) -> SplitSummary:
    """Write `.ocr.json` and `.ocr.md` for every row, recording failures in state.

    A row we cannot map or parse is reported and skipped -- one bad line must
    not abandon the rest of a completed batch.
    """
    summary = SplitSummary()

    with results_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except ValueError as exc:
                log.warning("%s:%d: malformed JSON: %s", results_path.name, number, exc)
                summary.malformed += 1
                continue

            if not isinstance(row, dict) or not isinstance(row.get("custom_id"), str):
                log.warning("%s:%d: malformed result row", results_path.name, number)
                summary.malformed += 1
                continue

            custom_id = row.get("custom_id")
            document = state.documents.get(custom_id) if custom_id else None

            if document is None:
                log.warning(
                    "%s:%d: result for unknown custom_id %r",
                    results_path.name,
                    number,
                    custom_id,
                )
                summary.unknown += 1
                continue

            response = row.get("response")
            if response is not None and not isinstance(response, dict):
                document.ocr_written = False
                document.ocr_error = "malformed response"
                summary.malformed += 1
                continue
            response = response or {}
            body = response.get("body")
            status = response.get("status_code")

            if (
                row.get("error")
                or row.get("errors")
                or not isinstance(body, dict)
                or (status is not None and (not isinstance(status, int) or not 200 <= status < 300))
            ):
                reason = _failure_reason(row)
                document.ocr_written = False
                document.ocr_error = reason
                log.error("OCR failed: %s: %s", document.relative_path, reason)
                summary.failed += 1
                continue

            try:
                if not isinstance(body.get("pages"), list):
                    raise ValueError("pages must be a list")
                markdown = render_markdown(body)
            except (AttributeError, TypeError, ValueError) as exc:
                document.ocr_written = False
                document.ocr_error = f"malformed OCR body: {exc}"
                log.warning("%s:%d: %s", results_path.name, number, document.ocr_error)
                summary.malformed += 1
                continue

            paths = output_paths(state.output_dir, Path(document.relative_path))

            if (
                document.ocr_written
                and paths.ocr_md.exists()
                and paths.ocr_json.exists()
                and not force
            ):
                summary.skipped += 1
                continue

            paths.ocr_md.parent.mkdir(parents=True, exist_ok=True)
            paths.ocr_json.write_text(
                json.dumps(body, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            paths.ocr_md.write_text(markdown, encoding="utf-8")

            document.ocr_written = True
            document.ocr_error = None
            summary.written += 1

    return summary
