import json
from pathlib import Path

import pytest
from conftest import Corpus, FakeMistral, batch_line, write_pdf

from ocr_batch import cli
from ocr_batch.errors import CollisionError, ConfigError, RemoteError, StateError
from ocr_batch.paths import file_sha256, output_paths
from ocr_batch.state import RunState


def options(**overrides: object) -> cli.SubmitOptions:
    base: dict[str, object] = {"upload_workers": 2, "jobs": 2}
    base.update(overrides)

    return cli.SubmitOptions(**base)  # type: ignore[arg-type]


def results_for(state: RunState) -> bytes:
    return (
        "\n".join(batch_line(custom_id) for custom_id in sorted(state.documents)) + "\n"
    ).encode("utf-8")


def test_submit_writes_state_before_anything_else(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"

    state = cli.do_submit(corpus.root, out, options())

    assert (out / "_state.json").exists()
    assert (out / "_manifest.json").exists()
    assert len(state.documents) == 4
    assert [job.job_id for job in state.jobs] == ["job-1"]
    # Every upload id is on disk, so cleanup can always find them again.
    assert len(RunState.load(out).remote_files) == 4


def test_native_outputs_keep_dotted_names_and_survive_a_broken_pdf(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"

    state = cli.do_submit(corpus.root, out, options(ocr=False))

    assert (out / "a.native.txt").exists()
    assert (out / "nested" / "b.final.native.txt").exists()
    assert (out / "C.native.txt").exists()
    assert not (out / "broken.native.txt").exists()

    broken = next(d for d in state.documents.values() if d.relative_path == "broken.pdf")

    assert broken.native_error
    assert not state.jobs


def test_run_end_to_end_writes_outputs_and_deletes_uploads(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    patched_client.output_file = "out-1"

    submitted = cli.do_submit(corpus.root, out, options())
    patched_client.downloads["out-1"] = results_for(submitted)

    code = cli.do_fetch(out)
    state = RunState.load(out)

    # broken.pdf failed local extraction, so the run reports partial success.
    assert code == cli.EXIT_PARTIAL
    assert (out / "nested" / "b.final.ocr.md").exists()
    assert (out / "nested" / "b.final.ocr.json").exists()
    assert (out / "_mistral_batch_results.jsonl").exists()
    assert all(document.ocr_written for document in state.documents.values())
    assert len(patched_client.deleted) == 4
    assert state.pending_remote_files() == []


def test_fetch_resumes_from_disk_without_re_uploading(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    submitted = cli.do_submit(corpus.root, out, options())
    patched_client.downloads["out-1"] = results_for(submitted)
    uploads_after_submit = len(patched_client.uploaded)

    # A fresh process would only have the state file -- which is all fetch uses.
    cli.do_fetch(out)

    assert uploads_after_submit == 4
    assert len(patched_client.submitted) == 4  # no second batch job


@pytest.mark.parametrize("force", [False, True])
def test_a_second_submit_refuses_to_re_pay_for_a_running_job(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral, force: bool
):
    out = tmp_path / "out"
    patched_client.job_status = "RUNNING"
    cli.do_submit(corpus.root, out, options())
    before = (out / "_state.json").read_bytes()

    with pytest.raises(StateError, match="already has running batch job"):
        cli.do_submit(corpus.root, out, options(force=force))

    assert (out / "_state.json").read_bytes() == before
    assert cli.do_cleanup(out) == cli.EXIT_ERROR
    assert patched_client.deleted == []


def test_completed_documents_are_skipped_unless_forced(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    submitted = cli.do_submit(corpus.root, out, options())
    patched_client.downloads["out-1"] = results_for(submitted)
    cli.do_fetch(out)

    uploads = len(patched_client.uploaded)
    state = cli.do_submit(corpus.root, out, options())

    assert len(patched_client.uploaded) == uploads  # nothing re-uploaded
    assert not state.jobs

    cli.do_submit(corpus.root, out, options(force=True, ocr=False))

    assert (out / "a.native.txt").exists()


def test_missing_ocr_json_is_not_treated_as_complete(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    submitted = cli.do_submit(corpus.root, out, options())
    patched_client.downloads["out-1"] = results_for(submitted)
    cli.do_fetch(out)
    (out / "a.ocr.json").unlink()

    state = cli.do_submit(corpus.root, out, options())

    assert state.jobs
    assert len(patched_client.submitted) == 5


def test_a_failure_before_any_job_exists_deletes_the_uploads(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    patched_client.fail_create_at = {0}

    with pytest.raises(Exception, match="create boom"):
        cli.do_submit(corpus.root, out, options())

    state = RunState.load(out)

    assert len(patched_client.deleted) == 4
    assert state.pending_remote_files() == []


def test_a_failure_after_a_job_exists_keeps_that_job_files(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    patched_client.fail_create_at = {1}

    with pytest.raises(Exception, match="create boom"):
        cli.do_submit(corpus.root, out, options(batch_size=2))

    state = RunState.load(out)
    kept = state.pending_remote_files()

    # The two files handed to job-1 stay; the two orphans are deleted.
    assert len(patched_client.deleted) == 2
    assert len(kept) == 2
    assert all(remote.job_id == "job-1" for remote in kept)


def test_cleanup_waits_for_a_running_job_unless_forced(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    patched_client.job_status = "RUNNING"
    cli.do_submit(corpus.root, out, options())

    assert cli.do_cleanup(out) == cli.EXIT_ERROR

    assert patched_client.deleted == []

    assert cli.do_cleanup(out, force=True) == cli.EXIT_OK

    assert len(patched_client.deleted) == 4


def test_cleanup_failure_is_an_error_and_remains_retryable(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    cli.do_submit(corpus.root, out, options())
    patched_client.fail_delete_for = {"file-1"}

    assert cli.do_cleanup(out) == cli.EXIT_ERROR
    assert [remote.file_id for remote in RunState.load(out).pending_remote_files()] == ["file-1"]


def test_fetch_download_failure_still_cleans_up(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    cli.do_submit(corpus.root, out, options())

    with pytest.raises(RemoteError, match="could not download"):
        cli.do_fetch(out)

    assert len(patched_client.deleted) == 4
    assert RunState.load(out).pending_remote_files() == []


def test_fetch_download_failure_honors_keep_remote(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    cli.do_submit(corpus.root, out, options())

    with pytest.raises(RemoteError, match="could not download"):
        cli.do_fetch(out, keep_remote=True)

    assert patched_client.deleted == []
    assert len(RunState.load(out).pending_remote_files()) == 4


def test_fetch_split_failure_still_cleans_up(
    corpus: Corpus,
    tmp_path: Path,
    patched_client: FakeMistral,
    monkeypatch: pytest.MonkeyPatch,
):
    out = tmp_path / "out"
    submitted = cli.do_submit(corpus.root, out, options())
    patched_client.downloads["out-1"] = results_for(submitted)

    def fail_split(*args: object, **kwargs: object) -> None:
        raise RuntimeError("split boom")

    monkeypatch.setattr(cli, "split_results", fail_split)

    with pytest.raises(RuntimeError, match="split boom"):
        cli.do_fetch(out)

    assert len(patched_client.deleted) == 4
    assert RunState.load(out).pending_remote_files() == []


def test_fetch_cleanup_failure_is_an_error(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    submitted = cli.do_submit(corpus.root, out, options())
    patched_client.downloads["out-1"] = results_for(submitted)
    patched_client.fail_delete_for = {"file-1"}

    assert cli.do_fetch(out) == cli.EXIT_ERROR
    assert [remote.file_id for remote in RunState.load(out).pending_remote_files()] == ["file-1"]


def test_fetch_without_wait_reports_a_running_job(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    patched_client.job_status = "RUNNING"
    cli.do_submit(corpus.root, out, options())

    assert cli.do_fetch(out, wait=False) == cli.EXIT_ERROR


def test_a_failed_job_still_downloads_its_error_file(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    patched_client.job_status = "TIMEOUT_EXCEEDED"
    patched_client.error_file = "err-1"

    submitted = cli.do_submit(corpus.root, out, options())
    patched_client.downloads["out-1"] = results_for(submitted)
    patched_client.downloads["err-1"] = b'{"custom_id": "x", "error": "boom"}\n'

    code = cli.do_fetch(out)

    assert code == cli.EXIT_PARTIAL
    assert (out / "_mistral_batch_errors.jsonl").exists()
    # Partial results were still salvaged rather than thrown away.
    assert (out / "a.ocr.md").exists()


def test_colliding_sources_are_refused(tmp_path: Path, patched_client: FakeMistral):
    root = tmp_path / "in"
    root.mkdir()
    (root / "a.pdf").write_bytes(b"%PDF-")
    (root / "a.PDF").write_bytes(b"%PDF-")

    with pytest.raises(CollisionError):
        cli.do_submit(root, tmp_path / "out", options())


def test_an_empty_input_directory_is_an_error(tmp_path: Path, patched_client: FakeMistral):
    root = tmp_path / "in"
    root.mkdir()

    with pytest.raises(ConfigError, match="no PDFs"):
        cli.do_submit(root, tmp_path / "out", options())


def test_missing_api_key_is_a_clean_message(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="MISTRAL_API_KEY"):
        cli.resolve_api_key()


def test_main_maps_errors_to_exit_codes(tmp_path: Path, patched_client: FakeMistral):
    assert cli.main(["status", str(tmp_path)]) == cli.EXIT_ERROR


def test_local_failures_are_reported_in_the_exit_code(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"

    assert cli.main(["run", str(corpus.root), str(out), "--no-ocr"]) == cli.EXIT_PARTIAL
    assert cli.main(["submit", str(corpus.root), str(out), "--no-ocr"]) == cli.EXIT_PARTIAL


def test_status_reports_a_run_without_jobs(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral, capsys: pytest.CaptureFixture[str]
):
    out = tmp_path / "out"
    cli.do_submit(corpus.root, out, options(ocr=False))

    assert cli.do_status(out) == cli.EXIT_OK

    printed = capsys.readouterr().out

    assert "documents: 4" in printed
    assert "jobs:      none" in printed


def test_status_reports_job_progress(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral, capsys: pytest.CaptureFixture[str]
):
    out = tmp_path / "out"
    patched_client.job_status = "RUNNING"
    cli.do_submit(corpus.root, out, options())

    assert cli.do_status(out) == cli.EXIT_OK
    assert "job job-1: RUNNING 4/4 succeeded" in capsys.readouterr().out


def test_a_missing_api_key_fails_before_any_work(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    out = tmp_path / "out"

    with pytest.raises(ConfigError, match="MISTRAL_API_KEY"):
        cli.do_submit(corpus.root, out, options())

    assert not (out / "_state.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jobs", 0),
        ("upload_workers", -1),
        ("batch_size", 0),
        ("timeout_hours", -1),
        ("url_expiry_hours", 0),
        ("upload_expiry_hours", -1),
    ],
)
def test_submit_options_reject_non_positive_numbers(field: str, value: int):
    with pytest.raises(ConfigError, match=field.replace("_", "-")):
        options(**{field: value})


def completed_run(tmp_path: Path, client: FakeMistral) -> tuple[Path, Path, RunState]:
    root = tmp_path / "in"
    write_pdf(root / "a.pdf", "OLD SOURCE")
    out = tmp_path / "out"
    state = cli.do_submit(root, out, options())
    custom_id = next(iter(state.documents))
    client.downloads["out-1"] = (batch_line(custom_id, "OLD OCR") + "\n").encode()
    assert cli.do_fetch(out) == cli.EXIT_OK
    return root, out, RunState.load(out)


def test_terminal_unfetched_job_is_preserved(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    state = cli.do_submit(corpus.root, out, options(native=False))
    cli.do_status(out)
    before = (out / "_state.json").read_bytes()

    with pytest.raises(StateError, match="unfetched"):
        cli.do_submit(corpus.root, out, options())

    assert (out / "_state.json").read_bytes() == before
    assert len(patched_client.jobs) == 1
    patched_client.downloads["out-1"] = results_for(state)
    assert cli.do_fetch(out) == cli.EXIT_OK


def test_force_discards_a_terminal_job_that_cannot_be_fetched(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    cli.do_submit(corpus.root, out, options(native=False))
    cli.do_status(out)

    with pytest.raises(RemoteError):
        cli.do_fetch(out, keep_remote=True)  # the output file is gone

    state = cli.do_submit(corpus.root, out, options(native=False, force=True))

    assert [job.job_id for job in state.jobs] == ["job-2"]
    assert len(state.pending_remote_files()) == 8  # job-1's uploads still tracked


def test_a_failed_forced_submit_keeps_the_record_of_existing_outputs(
    tmp_path: Path, patched_client: FakeMistral
):
    root, out, _ = completed_run(tmp_path, patched_client)
    patched_client.fail_upload_for = {"a.pdf"}

    with pytest.raises(Exception, match="upload boom"):
        cli.do_submit(root, out, options(force=True, native=False))

    assert all(document.ocr_written for document in RunState.load(out).documents.values())
    patched_client.fail_upload_for = set()
    uploads = len(patched_client.uploaded)
    assert not cli.do_submit(root, out, options(native=False)).jobs
    assert len(patched_client.uploaded) == uploads


def test_documents_that_never_reached_a_job_make_fetch_partial(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    patched_client.fail_create_at = {1}

    with pytest.raises(Exception, match="create boom"):
        cli.do_submit(corpus.root, out, options(native=False, batch_size=2))

    (job,) = RunState.load(out).jobs
    patched_client.downloads["out-1"] = (
        "\n".join(batch_line(custom_id) for custom_id in job.custom_ids) + "\n"
    ).encode()

    assert cli.do_fetch(out) == cli.EXIT_PARTIAL


def test_changed_source_redoes_native_and_ocr(tmp_path: Path, patched_client: FakeMistral):
    root, out, previous = completed_run(tmp_path, patched_client)
    write_pdf(root / "a.pdf", "NEW SOURCE")

    state = cli.do_submit(root, out, options())
    custom_id = next(iter(state.documents))
    paths = output_paths(out, Path("a.pdf"))
    assert state.documents[custom_id].sha256 != previous.documents[custom_id].sha256
    assert state.documents[custom_id].sha256 == file_sha256(root / "a.pdf")
    assert "NEW SOURCE" in paths.native.read_text()
    assert len(patched_client.submitted) == 2
    patched_client.downloads["out-1"] = (batch_line(custom_id, "NEW OCR") + "\n").encode()

    assert cli.do_fetch(out) == cli.EXIT_OK
    assert "NEW OCR" in paths.ocr_md.read_text()


def test_forced_submit_is_followed_by_an_ordinary_fetch(
    tmp_path: Path, patched_client: FakeMistral
):
    root, out, _ = completed_run(tmp_path, patched_client)
    state = cli.do_submit(root, out, options(force=True, native=False))
    custom_id = next(iter(state.documents))
    assert not state.documents[custom_id].ocr_written
    patched_client.downloads["out-1"] = (batch_line(custom_id, "NEW OCR") + "\n").encode()

    assert cli.do_fetch(out) == cli.EXIT_OK
    assert "NEW OCR" in (out / "a.ocr.md").read_text()


def test_fetch_counts_and_records_separate_request_errors(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    patched_client.error_file = "err-1"
    state = cli.do_submit(corpus.root, out, options(native=False))
    failed_id, *successful_ids = state.documents
    patched_client.jobs["job-1"].failed_requests = 1
    patched_client.downloads["out-1"] = (
        "\n".join(batch_line(custom_id) for custom_id in successful_ids) + "\n"
    ).encode()
    patched_client.downloads["err-1"] = (
        json.dumps({"custom_id": failed_id, "error": {"message": "OCR failed"}}) + "\n"
    ).encode()

    assert cli.do_fetch(out) == cli.EXIT_PARTIAL
    saved = RunState.load(out)
    assert "OCR failed" in (saved.documents[failed_id].ocr_error or "")
    assert not saved.documents[failed_id].ocr_written
    assert saved.jobs[0].failed_requests == 1
    patched_client.downloads.clear()
    patched_client.jobs.clear()
    assert cli.do_fetch(out) == cli.EXIT_PARTIAL


def test_missing_result_is_a_failure_on_every_fetch(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    state = cli.do_submit(corpus.root, out, options(native=False))
    missing_id, *successful_ids = state.documents
    patched_client.downloads["out-1"] = (
        "\n".join(batch_line(custom_id) for custom_id in successful_ids) + "\n"
    ).encode()

    assert cli.do_fetch(out) == cli.EXIT_PARTIAL
    assert RunState.load(out).documents[missing_id].ocr_error == "no successful OCR result returned"
    assert cli.do_fetch(out) == cli.EXIT_PARTIAL


def test_success_job_without_output_is_partial_and_can_be_retried(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral, caplog: pytest.LogCaptureFixture
):
    out = tmp_path / "out"
    patched_client.output_file = None
    cli.do_submit(corpus.root, out, options(native=False))

    assert cli.do_fetch(out) == cli.EXIT_PARTIAL
    assert "OCR results: 0 written, 4 failed" in caplog.text
    assert RunState.load(out).jobs[0].fetched
    assert cli.do_submit(corpus.root, out, options(native=False)).jobs


def test_cached_fetch_restores_outputs_without_remote_access(
    tmp_path: Path, patched_client: FakeMistral
):
    _, out, _ = completed_run(tmp_path, patched_client)
    (out / "a.ocr.md").unlink()
    (out / "a.ocr.json").unlink()
    patched_client.downloads.clear()
    patched_client.jobs.clear()

    assert cli.do_fetch(out) == cli.EXIT_OK
    assert "OLD OCR" in (out / "a.ocr.md").read_text()
    assert (out / "a.ocr.json").is_file()
    assert cli.do_fetch(out, force=True) == cli.EXIT_OK


def test_remote_failure_count_survives_cached_fetch(tmp_path: Path, patched_client: FakeMistral):
    root = tmp_path / "in"
    write_pdf(root / "a.pdf", "source")
    out = tmp_path / "out"
    state = cli.do_submit(root, out, options(native=False))
    patched_client.jobs["job-1"].failed_requests = 1
    patched_client.downloads["out-1"] = results_for(state)

    assert cli.do_fetch(out) == cli.EXIT_PARTIAL
    patched_client.jobs.clear()
    patched_client.downloads.clear()
    assert cli.do_fetch(out) == cli.EXIT_PARTIAL


def test_interrupted_uploads_are_saved_and_deleted(
    corpus: Corpus,
    tmp_path: Path,
    patched_client: FakeMistral,
    monkeypatch: pytest.MonkeyPatch,
):
    from concurrent.futures import Future

    from ocr_batch.remote import Upload

    def interrupt(futures: list[Future[Upload]]) -> None:
        for future in futures:
            future.result()
        raise KeyboardInterrupt

    monkeypatch.setattr("ocr_batch.remote.as_completed", interrupt)
    out = tmp_path / "out"

    with pytest.raises(KeyboardInterrupt):
        cli.do_submit(corpus.root, out, options(native=False))

    state = RunState.load(out)
    assert len(state.remote_files) == 4
    assert state.pending_remote_files() == []
    assert len(patched_client.deleted) == 4
    assert not patched_client.jobs


def test_fetch_resumes_multiple_jobs_after_a_later_download_fails(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    state = cli.do_submit(corpus.root, out, options(native=False, batch_size=2))
    for number, job in enumerate(state.jobs, 1):
        patched_client.jobs[job.job_id].output_file = f"out-{number}"
    first, second = state.jobs
    patched_client.downloads["out-1"] = (
        "\n".join(batch_line(custom_id) for custom_id in first.custom_ids) + "\n"
    ).encode()

    with pytest.raises(RemoteError):
        cli.do_fetch(out, keep_remote=True)

    assert RunState.load(out).jobs[0].fetched
    del patched_client.jobs[first.job_id]
    del patched_client.downloads["out-1"]
    patched_client.downloads["out-2"] = (
        "\n".join(batch_line(custom_id) for custom_id in second.custom_ids) + "\n"
    ).encode()

    assert cli.do_fetch(out) == cli.EXIT_OK
    assert all(job.fetched for job in RunState.load(out).jobs)
    assert len(patched_client.deleted) == 4
