import threading
import time
from contextlib import suppress
from pathlib import Path

import pytest
from conftest import FakeMistral

from ocr_batch.errors import UploadError
from ocr_batch.remote import (
    MAX_URL_EXPIRY_HOURS,
    OcrOptions,
    Upload,
    build_request,
    delete_files,
    download_file,
    submit_jobs,
    upload_documents,
    wait_for_jobs,
)


def test_request_body_matches_the_ocr_api(fake_client: FakeMistral):
    request = build_request("id1", "https://x", OcrOptions(confidence_granularity=None))

    assert request["custom_id"] == "id1"
    assert request["body"]["document"] == {"type": "document_url", "document_url": "https://x"}
    assert "confidence_scores_granularity" not in request["body"]
    # Annotations stay off on purpose.
    assert "document_annotation_format" not in request["body"]
    assert "bbox_annotation_format" not in request["body"]


def test_signed_url_expiry_is_capped(tmp_path: Path, fake_client: FakeMistral):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-")

    upload_documents(
        fake_client,
        [("id1", tmp_path / "a.pdf")],
        workers=2,
        url_expiry_hours=1000,
        upload_expiry_hours=48,
        on_upload=lambda upload: None,
        on_orphan=lambda file_id: None,
    )

    assert fake_client.url_expiry == MAX_URL_EXPIRY_HOURS
    assert fake_client.upload_expiry == 48


def test_a_failed_signed_url_still_reports_the_stored_file(
    tmp_path: Path, fake_client: FakeMistral
):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-")
    fake_client.fail_signed_url_for = {"file-1"}
    orphans: list[str] = []

    with pytest.raises(UploadError):
        upload_documents(
            fake_client,
            [("id1", tmp_path / "a.pdf")],
            workers=1,
            url_expiry_hours=24,
            upload_expiry_hours=48,
            on_upload=lambda upload: None,
            on_orphan=orphans.append,
        )

    assert orphans == ["file-1"]


def test_upload_failure_still_reports_sibling_uploads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_client: FakeMistral
):
    barrier = threading.Barrier(3)
    created: list[str] = []

    def controlled_upload(client: object, *, custom_id: str, **kwargs: object) -> Upload:
        barrier.wait()

        if custom_id == "bad":
            raise UploadError("signed URL failed", file_id="file-bad")

        time.sleep(0.05)
        file_id = f"file-{custom_id}"
        created.append(file_id)
        return Upload(custom_id, file_id, f"https://example.invalid/{file_id}")

    monkeypatch.setattr("ocr_batch.remote.upload_document", controlled_upload)
    recorded: list[str] = []
    orphans: list[str] = []

    with pytest.raises(UploadError, match="signed URL failed"):
        upload_documents(
            fake_client,
            [(name, tmp_path / f"{name}.pdf") for name in ("bad", "a", "b")],
            workers=3,
            url_expiry_hours=24,
            upload_expiry_hours=48,
            on_upload=lambda upload: recorded.append(upload.file_id),
            on_orphan=orphans.append,
        )

    assert sorted(created) == ["file-a", "file-b"]
    assert sorted(recorded) == ["file-a", "file-b"]
    assert orphans == ["file-bad"]


def test_requests_are_chunked_into_several_jobs(fake_client: FakeMistral):
    requests = [build_request(f"id{n}", "https://x", OcrOptions()) for n in range(5)]
    seen: list[tuple[str, int]] = []

    job_ids = submit_jobs(
        fake_client,
        requests,
        model="m",
        timeout_hours=24,
        batch_size=2,
        metadata={"purpose": "test"},
        on_job=lambda job_id, custom_ids: seen.append((job_id, len(custom_ids))),
    )

    assert len(job_ids) == 3
    assert [count for _, count in seen] == [2, 2, 1]
    assert fake_client.endpoint == "/v1/ocr"


def test_wait_polls_with_backoff_until_terminal(fake_client: FakeMistral):
    fake_client.job_status = "RUNNING"
    submit_jobs(
        fake_client,
        [build_request("id1", "https://x", OcrOptions())],
        model="m",
        timeout_hours=24,
        batch_size=10,
        metadata={},
        on_job=lambda job_id, custom_ids: None,
    )

    delays: list[float] = []
    ticks = {"n": 0}

    def sleep(seconds: float) -> None:
        delays.append(seconds)
        ticks["n"] += 1

        if ticks["n"] == 3:
            fake_client.jobs["job-1"].status = "SUCCESS"

    jobs = wait_for_jobs(fake_client, ["job-1"], on_update=lambda jobs, p: None, sleep=sleep)

    assert jobs["job-1"].status == "SUCCESS"
    assert delays == [5.0, 7.5, 11.25]


def test_download_streams_to_disk_and_closes(tmp_path: Path, fake_client: FakeMistral):
    fake_client.downloads["out-1"] = b"a" * (1 << 21)

    path = download_file(fake_client, "out-1", tmp_path / "nested" / "r.jsonl")

    assert path.stat().st_size == (1 << 21)
    assert fake_client.download_handles[0].closed


def test_delete_reports_only_the_failures(fake_client: FakeMistral):
    fake_client.fail_delete_for = {"file-2"}

    failures = delete_files(fake_client, ["file-1", "file-2", "file-3"])

    assert set(failures) == {"file-2"}
    assert fake_client.deleted == ["file-1", "file-3"]


@pytest.mark.parametrize("signed_url_failure", [False, True])
def test_interrupt_during_collection_still_records_uploaded_files(
    tmp_path: Path,
    fake_client: FakeMistral,
    monkeypatch: pytest.MonkeyPatch,
    signed_url_failure: bool,
):
    from concurrent.futures import Future

    source = tmp_path / "a.pdf"
    source.write_bytes(b"pdf")
    if signed_url_failure:
        fake_client.fail_signed_url_for = {"file-1"}

    def interrupt(futures: list[Future[Upload]]) -> None:
        for future in futures:
            with suppress(UploadError):
                future.result()
        raise KeyboardInterrupt

    monkeypatch.setattr("ocr_batch.remote.as_completed", interrupt)
    recorded: list[str] = []
    orphans: list[str] = []

    with pytest.raises(KeyboardInterrupt):
        upload_documents(
            fake_client,
            [("id1", source)],
            workers=1,
            url_expiry_hours=24,
            upload_expiry_hours=48,
            on_upload=lambda upload: recorded.append(upload.file_id),
            on_orphan=orphans.append,
        )

    assert recorded + orphans == ["file-1"]
    assert bool(orphans) == signed_url_failure


def test_interrupted_download_keeps_the_previous_cache(
    tmp_path: Path, fake_client: FakeMistral, monkeypatch: pytest.MonkeyPatch
):
    from collections.abc import Generator

    from conftest import FakeDownload

    class BrokenDownload(FakeDownload):
        def iter_bytes(self, size: int = 65536) -> Generator[bytes]:
            yield b"truncated replacement"
            raise RuntimeError("stream interrupted")

    response = BrokenDownload(b"")
    monkeypatch.setattr(fake_client.files, "download", lambda **kwargs: response)
    destination = tmp_path / "results.jsonl"
    destination.write_bytes(b"complete previous cache")

    with pytest.raises(RuntimeError, match="stream interrupted"):
        download_file(fake_client, "out-1", destination)

    assert destination.read_bytes() == b"complete previous cache"
    assert response.closed
    assert list(tmp_path.iterdir()) == [destination]


def test_second_interrupt_while_draining_still_records_the_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_client: FakeMistral
):
    from concurrent.futures import Future
    from concurrent.futures import wait as real_wait

    release = threading.Event()

    def slow_upload(client: object, *, custom_id: str, **kwargs: object) -> Upload:
        release.wait(timeout=5)
        return Upload(custom_id, "file-1", "https://example.invalid/file-1")

    def interrupt(futures: list[Future[Upload]]) -> None:
        raise KeyboardInterrupt

    def interrupted_wait(futures: list[Future[Upload]]) -> object:
        if not release.is_set():
            release.set()
            raise KeyboardInterrupt
        return real_wait(futures)

    monkeypatch.setattr("ocr_batch.remote.upload_document", slow_upload)
    monkeypatch.setattr("ocr_batch.remote.as_completed", interrupt)
    monkeypatch.setattr("ocr_batch.remote.wait", interrupted_wait)
    recorded: list[str] = []

    with pytest.raises(KeyboardInterrupt):
        upload_documents(
            fake_client,
            [("id1", tmp_path / "a.pdf")],
            workers=1,
            url_expiry_hours=24,
            upload_expiry_hours=48,
            on_upload=lambda upload: recorded.append(upload.file_id),
            on_orphan=lambda file_id: None,
        )

    assert recorded == ["file-1"]
