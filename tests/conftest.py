"""Shared fixtures: a fake Mistral client and a real (tiny) PDF generator."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# ----------------------------------------------------------------------
# A fake Mistral client covering exactly the call sites remote.py uses.
# ----------------------------------------------------------------------


@dataclass
class FakeUploaded:
    id: str


@dataclass
class FakeSignedURL:
    url: str


@dataclass
class FakeJob:
    id: str
    status: str = "QUEUED"
    total_requests: int = 0
    completed_requests: int = 0
    succeeded_requests: int = 0
    failed_requests: int = 0
    output_file: str | None = None
    error_file: str | None = None


class FakeDownload:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.closed = False

    def iter_bytes(self, size: int = 65536):
        for start in range(0, len(self._payload), size):
            yield self._payload[start : start + size]

    def close(self) -> None:
        self.closed = True


class FakeFiles:
    def __init__(self, owner: FakeMistral) -> None:
        self._owner = owner

    def upload(
        self, *, file: dict[str, Any], purpose: str, visibility: str, expiry: int | None = None
    ) -> FakeUploaded:
        if self._owner.fail_upload_for and file["file_name"] in self._owner.fail_upload_for:
            raise RuntimeError("upload boom")

        file_id = f"file-{len(self._owner.uploaded) + 1}"
        self._owner.uploaded[file_id] = file["file_name"]
        self._owner.upload_expiry = expiry

        return FakeUploaded(file_id)

    def get_signed_url(self, *, file_id: str, expiry: int = 24) -> FakeSignedURL:
        if file_id in self._owner.fail_signed_url_for:
            raise RuntimeError("signed url boom")

        self._owner.url_expiry = expiry

        return FakeSignedURL(f"https://example.invalid/{file_id}")

    def download(self, *, file_id: str) -> FakeDownload:
        download = FakeDownload(self._owner.downloads[file_id])
        self._owner.download_handles.append(download)

        return download

    def delete(self, *, file_id: str) -> None:
        if file_id in self._owner.fail_delete_for:
            raise RuntimeError("delete boom")

        self._owner.deleted.append(file_id)
        self._owner.uploaded.pop(file_id, None)


class FakeJobs:
    def __init__(self, owner: FakeMistral) -> None:
        self._owner = owner

    def create(
        self,
        *,
        requests: list[dict[str, Any]],
        model: str,
        endpoint: str,
        timeout_hours: int,
        metadata: dict[str, str],
    ) -> FakeJob:
        if len(self._owner.jobs) in self._owner.fail_create_at:
            raise RuntimeError("create boom")

        job = FakeJob(
            id=f"job-{len(self._owner.jobs) + 1}",
            status=self._owner.job_status,
            total_requests=len(requests),
            completed_requests=len(requests),
            succeeded_requests=len(requests),
            output_file=self._owner.output_file,
            error_file=self._owner.error_file,
        )

        self._owner.jobs[job.id] = job
        self._owner.submitted.extend(requests)
        self._owner.endpoint = endpoint
        self._owner.model = model

        return job

    def get(self, *, job_id: str) -> FakeJob:
        return self._owner.jobs[job_id]

    def cancel(self, *, job_id: str) -> FakeJob:
        job = self._owner.jobs[job_id]
        job.status = "CANCELLED"
        self._owner.cancelled.append(job_id)

        return job


class FakeBatch:
    def __init__(self, owner: FakeMistral) -> None:
        self.jobs = FakeJobs(owner)


class FakeMistral:
    def __init__(self) -> None:
        self.files = FakeFiles(self)
        self.batch = FakeBatch(self)

        self.uploaded: dict[str, str] = {}
        self.deleted: list[str] = []
        self.jobs: dict[str, FakeJob] = {}
        self.submitted: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.downloads: dict[str, bytes] = {}
        self.download_handles: list[FakeDownload] = []

        self.job_status = "SUCCESS"
        self.output_file: str | None = "out-1"
        self.error_file: str | None = None
        self.endpoint: str | None = None
        self.model: str | None = None
        self.upload_expiry: int | None = None
        self.url_expiry: int | None = None

        self.fail_upload_for: set[str] = set()
        self.fail_signed_url_for: set[str] = set()
        self.fail_delete_for: set[str] = set()
        self.fail_create_at: set[int] = set()

    def __enter__(self) -> FakeMistral:
        return self

    def __exit__(self, *args: object) -> None:
        return None


@pytest.fixture
def fake_client() -> FakeMistral:
    return FakeMistral()


@pytest.fixture
def patched_client(monkeypatch: pytest.MonkeyPatch, fake_client: FakeMistral) -> FakeMistral:
    """Make `cli.make_client` hand out the fake, with a credential in the env."""
    import contextlib

    from ocr_batch import cli

    @contextlib.contextmanager
    def _make_client(api_key: str | None = None):
        yield fake_client

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(cli, "make_client", _make_client)

    return fake_client


# ----------------------------------------------------------------------
# PDF fixtures, generated rather than committed.
# ----------------------------------------------------------------------


def write_pdf(path: Path, text: str, pages: int = 1) -> Path:
    import pymupdf

    path.parent.mkdir(parents=True, exist_ok=True)

    with pymupdf.open() as doc:
        for number in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"{text} {number + 1}")

        doc.save(path)

    return path


@dataclass
class Corpus:
    root: Path
    names: list[str] = field(default_factory=list)


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    """An input tree exercising dotted names, uppercase suffixes and a bad file."""
    root = tmp_path / "in"
    names = ["a.pdf", "nested/b.final.pdf", "C.PDF", "broken.pdf"]

    for name in names[:-1]:
        write_pdf(root / name, "hello")

    (root / "broken.pdf").write_bytes(b"not a pdf at all")

    return Corpus(root=root, names=names)


def batch_line(custom_id: str, markdown: str = "# Title", pages: int = 1) -> str:
    body = {
        "pages": [{"index": index, "markdown": f"{markdown} p{index}"} for index in range(pages)]
    }

    return json.dumps({"custom_id": custom_id, "response": {"status_code": 200, "body": body}})
