from pathlib import Path

import pytest
from conftest import Corpus, FakeMistral, batch_line

from ocr_batch import cli
from ocr_batch.errors import CollisionError, ConfigError, RemoteError, StateError
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


def test_a_second_submit_refuses_to_re_pay_for_a_running_job(
    corpus: Corpus, tmp_path: Path, patched_client: FakeMistral
):
    out = tmp_path / "out"
    patched_client.job_status = "RUNNING"
    cli.do_submit(corpus.root, out, options())

    with pytest.raises(StateError, match="already has running batch job"):
        cli.do_submit(corpus.root, out, options())


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
