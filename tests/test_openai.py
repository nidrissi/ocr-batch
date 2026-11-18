"""Exercise the OpenAI workflow with SDK-shaped fakes and generated PDFs."""

import base64
import json
import signal
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import pymupdf
import pytest
from conftest import write_pdf

from ocr_batch import cli, openai_backend
from ocr_batch.state import RunState


@dataclass
class FakeBatch:
    id: str
    request_counts: SimpleNamespace
    status: str = "validating"
    output_file_id: str | None = None
    error_file_id: str | None = None


class FakeFiles:
    def __init__(self, client: FakeOpenAI) -> None:
        self.client = client
        self.with_streaming_response = self

    def create(
        self,
        *,
        file: BinaryIO | tuple[str, BinaryIO, str],
        purpose: str,
        expires_after: dict[str, Any],
    ) -> SimpleNamespace:
        assert purpose == "batch"
        assert expires_after["anchor"] == "created_at"
        assert 24 * 3600 < expires_after["seconds"] <= 30 * 86400
        file_id = f"input-{len(self.client.uploads) + 1}"
        if isinstance(file, tuple):
            filename, file, content_type = file
            assert filename.endswith(".jsonl")
            assert content_type == "application/jsonl"
        self.client.uploads[file_id] = file.read()
        if self.client.interrupt == "upload":
            signal.raise_signal(signal.SIGINT)
        return SimpleNamespace(id=file_id)

    @contextmanager
    def content(self, file_id: str) -> Iterator[SimpleNamespace]:
        self.client.downloads.append(file_id)

        def iter_bytes(chunk_size: int = 65536) -> Iterator[bytes]:
            payload = self.client.payloads[file_id]
            for start in range(0, len(payload), chunk_size):
                if start == 0 and self.client.interrupt == file_id:
                    yield payload[:1]
                    raise KeyboardInterrupt
                yield payload[start : start + chunk_size]

        yield SimpleNamespace(iter_bytes=iter_bytes)

    def delete(self, file_id: str) -> SimpleNamespace:
        if file_id in self.client.fail_delete:
            raise RuntimeError("temporary delete failure")
        self.client.deleted.append(file_id)
        self.client.payloads.pop(file_id, None)
        return SimpleNamespace(deleted=True)


class FakeBatches:
    def __init__(self, client: FakeOpenAI) -> None:
        self.client = client

    def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, str],
        output_expires_after: dict[str, Any],
    ) -> FakeBatch:
        assert endpoint == "/v1/responses"
        assert completion_window == "24h"
        assert metadata["purpose"] == "ocr-batch"
        assert output_expires_after["anchor"] == "created_at"
        assert output_expires_after["seconds"] >= 7 * 86400
        state = RunState.load(self.client.out)
        assert input_file_id in {remote.file_id for remote in state.pending_remote_files()}
        live = {job.id for job in self.client.jobs.values() if job.status == "validating"}
        assert live <= {job.job_id for job in state.jobs}
        rows = [json.loads(line) for line in self.client.uploads[input_file_id].splitlines()]
        job_id = f"batch-{len(self.client.jobs) + 1}"
        job = FakeBatch(
            id=job_id, request_counts=SimpleNamespace(total=len(rows), completed=0, failed=0)
        )
        self.client.jobs[job_id] = job
        self.client.requests[job_id] = rows
        if self.client.interrupt == "create":
            signal.raise_signal(signal.SIGINT)
        return job

    def retrieve(self, batch_id: str) -> FakeBatch:
        self.client.retrieved.append(batch_id)
        if self.client.interrupt == "retrieve":
            raise KeyboardInterrupt
        return self.client.jobs[batch_id]

    def cancel(self, batch_id: str) -> FakeBatch:
        self.client.cancelled.append(batch_id)
        job = self.client.jobs[batch_id]
        job.status = "cancelling"
        return job


@dataclass
class FakeOpenAI:
    root: Path
    out: Path
    uploads: dict[str, bytes] = field(default_factory=dict)
    jobs: dict[str, FakeBatch] = field(default_factory=dict)
    requests: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    payloads: dict[str, bytes] = field(default_factory=dict)
    downloads: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    keys: list[str | None] = field(default_factory=list)
    fail_delete: set[str] = field(default_factory=set)
    interrupt: str | None = None
    files: FakeFiles = field(init=False)
    batches: FakeBatches = field(init=False)

    def __post_init__(self) -> None:
        self.files = FakeFiles(self)
        self.batches = FakeBatches(self)

    def submit(self, *args: str) -> int:
        return cli.main(
            ["submit", str(self.root), str(self.out), "--provider", "openai", "--no-native", *args]
        )

    def resume(self, command: str, *args: str) -> int:
        return cli.main([command, str(self.out), *args])

    def ids(self, job_id: str, document: str | None = None) -> list[str]:
        requests = [row["custom_id"] for row in self.requests[job_id]]
        if document is None:
            return requests
        state = RunState.load(self.out)
        saved_requests = next(job for job in state.jobs if job.job_id == job_id).requests
        return [
            custom_id
            for custom_id in requests
            if state.documents[saved_requests[custom_id].document_id].relative_path == document
        ]

    def finish(
        self,
        job_id: str,
        rows: list[dict[str, Any] | str],
        *,
        errors: list[dict[str, Any]] | None = None,
        status: str = "completed",
    ) -> None:
        job = self.jobs[job_id]
        job.status = status
        job.request_counts.completed = len(rows)
        if rows:
            file_id = job.output_file_id = f"output-{job_id}"
            self.payloads[file_id] = _jsonl(rows)
        if errors:
            file_id = job.error_file_id = f"error-{job_id}"
            self.payloads[file_id] = _jsonl(errors)


@pytest.fixture
def openai_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    client = FakeOpenAI(tmp_path / "in", tmp_path / "out")

    @contextmanager
    def make_client(api_key: str | None = None) -> Iterator[FakeOpenAI]:
        client.keys.append(api_key)
        yield client

    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai")
    monkeypatch.setattr(cli, "make_openai_client", make_client)
    return client


def _jsonl(rows: Iterable[dict[str, Any] | str]) -> bytes:
    return "".join(
        (row if isinstance(row, str) else json.dumps(row)) + "\n" for row in rows
    ).encode()


def _message(content: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [content],
    }


def response_row(
    custom_id: str,
    markdown: str = "Transcribed text",
    *,
    blank: bool = False,
    status_code: int = 200,
    **body_fields: Any,
) -> dict[str, Any]:
    return {
        "id": f"row-{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": status_code,
            "request_id": f"req-{custom_id}",
            "body": {
                "id": f"resp-{custom_id}",
                "status": "completed",
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        **_message({"type": "output_text", "text": "Working on transcription"}),
                        "phase": "commentary",
                    },
                    _message(
                        {
                            "type": "output_text",
                            "text": json.dumps({"markdown": markdown, "blank": blank}),
                        }
                    ),
                ],
                "usage": {"input_tokens": 81, "output_tokens": 12, "total_tokens": 93},
                **body_fields,
            },
        },
        "error": None,
    }


def test_renders_text_and_blank_pages_without_native_extraction(openai_client: FakeOpenAI):
    client = openai_client
    client.root.mkdir()
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Synthetic text")
        pdf.new_page()
        pdf.save(client.root / "scan.final.pdf")
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(client.root / "encrypted.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="x")
    (client.root / "broken.pdf").write_bytes(b"invalid PDF")

    assert client.submit("--api-key", "explicit") == cli.EXIT_PARTIAL
    assert client.keys == ["explicit"]
    assert not list(client.out.glob("*.native.txt"))
    assert len(client.requests) == 1
    rows = client.requests["batch-1"]
    assert len(rows) == 2
    submitted = RunState.load(client.out)
    (job,) = submitted.jobs
    assert job.configuration == submitted.configuration
    pages = [job.requests[row["custom_id"]] for row in rows]
    assert [page.page_index for page in pages] == [0, 1]
    document = submitted.documents[pages[0].document_id]
    assert document.expected_pages == 2
    assert document.native_pages is None

    for row in rows:
        assert row["method"] == "POST" and row["url"] == "/v1/responses"
        body = row["body"]
        assert body["model"] == "gpt-6.1-terra"
        assert body["store"] is False
        assert not body.get("tools")
        format_ = body["text"]["format"]
        assert format_["strict"] is True
        assert set(format_["schema"]["required"]) == {"markdown", "blank"}
        assert format_["schema"]["additionalProperties"] is False
        img, *_ = (
            item["image_url"]
            for message in body["input"]
            for item in message["content"]
            if item["type"] == "input_image"
        )
        prefix, b64_png = img.split(",", 1)
        assert prefix == "data:image/png;base64"
        pixmap = pymupdf.Pixmap(base64.b64decode(b64_png, validate=True))
        assert (pixmap.xres, pixmap.yres) == (200, 200)
        assert pixmap.n == 3 and not pixmap.alpha

    first, second = client.ids("batch-1")
    client.finish("batch-1", [response_row(second, "", blank=True), response_row(first)])
    assert client.resume("fetch") == cli.EXIT_PARTIAL
    markdown = (client.out / "scan.final.ocr.md").read_text()
    assert markdown.index("PAGE 1") < markdown.index("Transcribed text") < markdown.index("PAGE 2")
    output = json.loads((client.out / "scan.final.ocr.json").read_text())
    assert output["provider"] == "openai"
    first_page, blank_page = output["pages"]
    assert first_page["response"]["usage"]["total_tokens"] == 93
    assert blank_page["blank"]
    docs = {doc.relative_path: doc for doc in RunState.load(client.out).documents.values()}
    assert docs["broken.pdf"].ocr_error
    assert docs["encrypted.pdf"].ocr_error


def test_run_uses_the_model_override_through_fetch_and_cleanup(
    openai_client: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
):
    client = openai_client
    write_pdf(client.root / "a.pdf", "Synthetic")
    original_retrieve = client.batches.retrieve

    def complete_when_polled(job_id: str) -> FakeBatch:
        (custom_id,) = client.ids(job_id)
        client.finish(job_id, [response_row(custom_id)])
        return original_retrieve(job_id)

    monkeypatch.setattr(client.batches, "retrieve", complete_when_polled)
    model = "another-vision-model"
    assert (
        cli.main(
            [
                "run",
                str(client.root),
                str(client.out),
                "--provider",
                "openai",
                "--no-native",
                "--model",
                model,
            ]
        )
        == cli.EXIT_OK
    )
    body = client.requests["batch-1"][0]["body"]
    assert body["model"] == model
    assert "reasoning" not in body
    output = json.loads((client.out / "a.ocr.json").read_text())
    assert output["configuration"]["model"] == model
    assert set(client.deleted) == {"input-1", "output-batch-1"}


@pytest.mark.parametrize("limit", ["cli-count", "provider-count", "bytes"])
def test_partitions_the_serialized_requests(
    limit: str, openai_client: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
):
    client = openai_client
    write_pdf(client.root / "large.pdf", "Synthetic corpus", pages=3)
    assert client.submit() == cli.EXIT_OK
    original = next(iter(client.uploads.values())).splitlines(keepends=True)
    assert len(original) == 3

    comparison = FakeOpenAI(client.root, client.out.with_name("partitioned"))

    @contextmanager
    def make_client(api_key: str | None = None) -> Iterator[FakeOpenAI]:
        yield comparison

    monkeypatch.setattr(cli, "make_openai_client", make_client)
    if limit == "bytes":
        byte_limit = min(len(a) + len(b) for a, b in pairwise(original)) - 1
        monkeypatch.setattr(openai_backend, "MAX_BATCH_BYTES", byte_limit)
        options: list[str] = []
    elif limit == "provider-count":
        monkeypatch.setattr(openai_backend, "MAX_BATCH_REQUESTS", 1)
        options = []
    else:
        options = ["--batch-size", "1"]

    assert comparison.submit(*options) == cli.EXIT_OK
    assert list(comparison.uploads.values()) == original


def test_oversized_page_fails_the_document_without_resizing(
    openai_client: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
):
    client = openai_client
    write_pdf(client.root / "normal.pdf", "Synthetic")
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.new_page(width=1200, height=1200)
        pdf.save(client.root / "oversized.pdf")
    monkeypatch.setattr(openai_backend, "MAX_PAGE_DIMENSION", 3000)

    assert client.submit() == cli.EXIT_PARTIAL
    assert len(client.ids("batch-1")) == 1
    assert client.ids("batch-1") == client.ids("batch-1", "normal.pdf")
    state = RunState.load(client.out)
    oversized = next(
        doc for doc in state.documents.values() if doc.relative_path == "oversized.pdf"
    )
    assert "page 2" in (oversized.ocr_error or "")
    assert "image limits" in (oversized.ocr_error or "")


def test_assembles_a_document_across_jobs_and_downloads_partial_expiry(openai_client: FakeOpenAI):
    client = openai_client
    write_pdf(client.root / "a.pdf", "Synthetic", pages=3)
    assert client.submit("--batch-size", "2") == cli.EXIT_OK
    first, second = client.ids("batch-1")
    (third,) = client.ids("batch-2")
    repeated = response_row(first, "First")
    client.finish("batch-1", [response_row(second, "Second"), repeated, repeated])

    assert client.resume("fetch", "--no-wait") == cli.EXIT_ERROR
    (doc,) = RunState.load(client.out).documents.values()
    assert doc.ocr_error is None

    client.finish("batch-2", [response_row(third, "Third")], status="expired")
    assert client.resume("fetch") == cli.EXIT_PARTIAL
    output = json.loads((client.out / "a.ocr.json").read_text())
    assert [page["markdown"] for page in output["pages"]] == ["First", "Second", "Third"]
    assert [page["job_id"] for page in output["pages"]] == ["batch-1", "batch-1", "batch-2"]


def test_missing_and_conflicting_pages_leave_other_documents_usable(openai_client: FakeOpenAI):
    client = openai_client
    write_pdf(client.root / "a.pdf", "Synthetic", pages=2)
    write_pdf(client.root / "b.pdf", "Synthetic")
    write_pdf(client.root / "c.pdf", "Synthetic")
    assert client.submit() == cli.EXIT_OK
    first, second = client.ids("batch-1", "a.pdf")
    (good,) = client.ids("batch-1", "c.pdf")
    client.finish(
        "batch-1",
        [
            response_row(good),
            response_row(second),
            response_row(first, "First text"),
            response_row(first, "Conflicting text"),
            response_row("unrecognized-page"),
        ],
    )
    assert client.resume("fetch") == cli.EXIT_PARTIAL
    assert not (client.out / "a.ocr.json").exists()
    assert not (client.out / "b.ocr.json").exists()
    assert (client.out / "c.ocr.json").is_file()
    assert client.resume("fetch") == cli.EXIT_PARTIAL


def test_request_id_cannot_be_satisfied_by_another_jobs_output(openai_client: FakeOpenAI):
    client = openai_client
    write_pdf(client.root / "a.pdf", "Synthetic", pages=2)
    assert client.submit("--batch-size", "1") == cli.EXIT_OK
    (first,) = client.ids("batch-1")
    (second,) = client.ids("batch-2")
    client.finish("batch-1", [response_row(first), response_row(second)])
    client.finish("batch-2", [])

    assert client.resume("fetch") == cli.EXIT_PARTIAL
    assert not (client.out / "a.ocr.json").exists()


@pytest.mark.parametrize(
    ("status_code", "fields"),
    [
        (500, {}),
        (200, {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}),
        (200, {"output": [{"type": "reasoning", "summary": []}]}),
        (200, {"output": [_message({"type": "refusal", "refusal": "No"})]}),
        (200, {"output": [_message({"type": "output_text", "text": "{oops"})]}),
        (
            200,
            {
                "output": [
                    _message({"type": "output_text", "text": '{"markdown":"","blank":false}'})
                ]
            },
        ),
    ],
)
def test_invalid_response_does_not_count_as_a_page(
    status_code: int, fields: dict[str, Any], openai_client: FakeOpenAI
):
    client = openai_client
    write_pdf(client.root / "a.pdf", "Synthetic")
    write_pdf(client.root / "b.pdf", "Synthetic")
    assert client.submit() == cli.EXIT_OK
    (bad,) = client.ids("batch-1", "a.pdf")
    (good,) = client.ids("batch-1", "b.pdf")
    client.finish(
        "batch-1", [response_row(bad, status_code=status_code, **fields), "{", response_row(good)]
    )
    raw = client.payloads["output-batch-1"]
    assert client.resume("fetch") == cli.EXIT_PARTIAL
    assert (client.out / "_openai_batch_results-batch-1.jsonl").read_bytes() == raw
    assert not (client.out / "a.ocr.md").exists()
    assert (client.out / "b.ocr.md").is_file()


@pytest.mark.parametrize("interrupt", ["upload", "create"])
def test_submit_persists_ids_when_interrupted(interrupt: str, openai_client: FakeOpenAI):
    client = openai_client
    write_pdf(client.root / "a.pdf", "Synthetic", pages=2)
    client.interrupt = interrupt
    assert client.submit("--batch-size", "1") == cli.EXIT_INTERRUPTED

    state = RunState.load(client.out)
    (remote,) = state.remote_files
    assert remote.file_id == "input-1"
    if interrupt == "create":
        assert remote.job_id == state.jobs[0].job_id == "batch-1"
        assert not remote.deleted
        assert client.resume("cleanup") == cli.EXIT_ERROR
        assert not client.deleted
    else:
        assert not state.jobs
        assert remote.job_id is None
        assert client.resume("cleanup") == cli.EXIT_OK
        assert client.deleted == ["input-1"]


def test_interrupted_fetch_resumes_from_raw_downloads_and_restores_outputs(
    openai_client: FakeOpenAI,
):
    client = openai_client
    write_pdf(client.root / "a.pdf", "Synthetic", pages=2)
    assert client.submit("--batch-size", "1") == cli.EXIT_OK
    for job_id in client.jobs:
        (custom_id,) = client.ids(job_id)
        client.finish(job_id, [response_row(custom_id)])
    client.interrupt = "output-batch-2"

    assert client.resume("fetch") == cli.EXIT_INTERRUPTED
    assert (client.out / "_openai_batch_results-batch-1.jsonl").is_file()
    assert not (client.out / "_openai_batch_results-batch-2.jsonl").exists()
    client.interrupt = None
    assert client.resume("fetch") == cli.EXIT_OK
    assert client.downloads == ["output-batch-1", "output-batch-2", "output-batch-2"]

    (client.out / "a.ocr.md").unlink()
    (client.out / "a.ocr.json").unlink()
    for job in client.jobs.values():
        job.output_file_id = None
    assert client.resume("status") == cli.EXIT_OK
    client.jobs.clear()
    assert client.resume("fetch") == cli.EXIT_OK
    assert len(client.downloads) == 3
    assert len(client.uploads) == 2
    assert (client.out / "a.ocr.md").is_file()
    assert (client.out / "a.ocr.json").is_file()


def test_ambiguous_batch_creation_keeps_the_input(
    openai_client: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
):
    client = openai_client
    write_pdf(client.root / "a.pdf", "Synthetic")
    create = client.batches.create

    def create_without_reply(**kwargs: Any) -> FakeBatch:
        create(**kwargs)
        raise ConnectionError("batch reply was lost")

    monkeypatch.setattr(client.batches, "create", create_without_reply)
    assert client.submit() == cli.EXIT_ERROR
    state = RunState.load(client.out)
    assert not state.jobs
    assert state.pending_remote_files()[0].file_id == "input-1"
    assert client.resume("cleanup") == cli.EXIT_ERROR
    assert not client.deleted


def test_failed_replacement_retains_openai_output_and_its_provenance(openai_client: FakeOpenAI):
    client = openai_client
    write_pdf(client.root / "a.pdf", "Synthetic")
    assert client.submit() == cli.EXIT_OK
    (page,) = client.ids("batch-1")
    client.finish("batch-1", [response_row(page)])
    assert client.resume("fetch") == cli.EXIT_OK
    output = client.out / "a.ocr.json"
    original = output.read_bytes()
    old_config = json.loads(original)["configuration"]

    assert client.submit("--model", "another-vision-model") == cli.EXIT_OK
    (page,) = client.ids("batch-2")
    client.finish(
        "batch-2",
        [
            response_row(
                page, status="incomplete", incomplete_details={"reason": "max_output_tokens"}
            )
        ],
    )
    assert client.resume("fetch") == cli.EXIT_PARTIAL
    state = RunState.load(client.out)
    (document,) = state.documents.values()
    assert output.read_bytes() == original
    assert document.ocr_config == old_config
    assert state.jobs[0].configuration != old_config


def test_cleanup_never_deletes_unfetched_results(openai_client: FakeOpenAI):
    client = openai_client
    write_pdf(client.root / "a.pdf", "Synthetic")
    assert client.submit() == cli.EXIT_OK
    (page,) = client.ids("batch-1")
    client.finish(
        "batch-1",
        [],
        errors=[{"custom_id": page, "error": {"message": "request expired"}}],
        status="expired",
    )
    file_id = "error-batch-1"
    payload = client.payloads.pop(file_id)
    assert client.resume("fetch") == cli.EXIT_ERROR
    assert file_id not in client.deleted
    client.resume("cleanup", "--force")
    assert client.deleted == ["input-1"]

    client.payloads[file_id] = payload
    client.fail_delete.add(file_id)
    assert client.resume("fetch") == cli.EXIT_ERROR
    assert (client.out / "_openai_batch_errors-batch-1.jsonl").read_bytes() == payload
    client.fail_delete.clear()
    assert client.resume("cleanup") == cli.EXIT_OK
    assert client.resume("cleanup") == cli.EXIT_OK
    assert client.deleted == ["input-1", file_id]
    assert client.resume("fetch") == cli.EXIT_PARTIAL


@pytest.mark.parametrize(
    "flags",
    [
        ("--upload-workers", "8"),
        ("--url-expiry-hours", "168"),
        ("--upload-expiry-hours", "48"),
        ("--no-include-blocks",),
        ("--confidence-granularity", "block"),
        ("--table-format", "markdown"),
        ("--timeout-hours", "12"),
    ],
)
def test_rejects_explicitly_incompatible_options(
    flags: tuple[str, ...], openai_client: FakeOpenAI, caplog: pytest.LogCaptureFixture
):
    client = openai_client
    write_pdf(client.root / "a.pdf", "Synthetic")
    assert client.submit(*flags) == cli.EXIT_ERROR
    assert flags[0] in caplog.text
    assert client.keys == []
    assert not (client.out / "_state.json").exists()


def test_cancellation_in_progress_protects_inputs_and_cancelled_jobs_keep_outputs(
    openai_client: FakeOpenAI,
):
    client = openai_client
    write_pdf(client.root / "a.pdf", "Synthetic")
    assert client.submit("--timeout-hours", "24") == cli.EXIT_OK
    client.interrupt = "retrieve"
    assert client.resume("fetch", "--cancel-on-interrupt") == cli.EXIT_INTERRUPTED
    assert client.cancelled == ["batch-1"]
    client.interrupt = None
    assert client.resume("status") == cli.EXIT_OK
    assert client.resume("fetch", "--no-wait") == cli.EXIT_ERROR
    assert client.resume("cleanup") == cli.EXIT_ERROR
    assert not client.deleted

    (page,) = client.ids("batch-1")
    client.finish("batch-1", [response_row(page)], status="cancelled")
    assert client.resume("fetch", "--keep-remote") == cli.EXIT_PARTIAL
    assert (client.out / "a.ocr.md").is_file()
    assert not client.deleted
    assert client.resume("cleanup") == cli.EXIT_OK
    assert set(client.deleted) == {"input-1", "output-batch-1"}
