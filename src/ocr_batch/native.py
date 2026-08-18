"""Local text extraction with PyMuPDF, executed in a process pool.

Workers never raise: a corrupt or password-protected PDF must not take down a
run whose expensive half has already been paid for.
"""

import logging
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import pymupdf

from .text import join_pages, page_chunk

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NativeResult:
    custom_id: str
    source: str
    pages: int = 0
    error: str | None = None


def extract_one(custom_id: str, pdf_path_name: str, output_path_name: str) -> NativeResult:
    """Extract one PDF's text layer. Runs in a child process; never raises."""
    pdf_path = Path(pdf_path_name)
    output_path = Path(output_path_name)
    chunks: list[str] = []

    try:
        with pymupdf.open(pdf_path) as doc:
            if doc.needs_pass:
                return NativeResult(custom_id, pdf_path_name, error="PDF is password-protected")

            for index in range(doc.page_count):
                page = doc.load_page(index)
                text = page.get_text("text", sort=True).strip()  # pyright: ignore[reportAttributeAccessIssue]

                chunks.append(page_chunk(index + 1, text))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(join_pages(chunks), encoding="utf-8")
    except Exception as exc:
        return NativeResult(custom_id, pdf_path_name, error=f"{type(exc).__name__}: {exc}")

    return NativeResult(custom_id, pdf_path_name, pages=len(chunks))


class NativePool:
    """A process pool that starts work immediately and is collected later.

    Used as a context manager so an exception on the network side still tears
    the children down instead of hanging the interpreter on exit.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self._pool = ProcessPoolExecutor(max_workers=max_workers)
        self._futures: list[Future[NativeResult]] = []

    def submit(self, custom_id: str, pdf: Path, output: Path) -> None:
        self._futures.append(self._pool.submit(extract_one, custom_id, str(pdf), str(output)))

    @property
    def submitted(self) -> int:
        return len(self._futures)

    def results(self) -> Iterator[NativeResult]:
        """Yield results as they complete."""
        for future in as_completed(self._futures):
            yield future.result()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._pool.shutdown(wait=exc_type is None, cancel_futures=exc_type is not None)
