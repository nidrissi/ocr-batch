import json
from pathlib import Path

import pytest

from ocr_batch.errors import StateError
from ocr_batch.state import DocumentState, JobState, RemoteFile, RunState, write_atomic


def test_round_trip(tmp_path: Path):
    state = RunState.create(output_dir=tmp_path, input_dir=Path("/in"), model="m")
    state.documents["abc"] = DocumentState("abc", "a/b.pdf", "deadbeef", native_pages=3)
    state.jobs.append(JobState("job-1", ["abc"], status="SUCCESS", output_file="out-1"))
    state.remote_files.append(RemoteFile("file-1", custom_id="abc", job_id="job-1"))
    state.save()

    loaded = RunState.load(tmp_path)

    assert loaded.documents["abc"].native_pages == 3
    assert loaded.jobs[0].terminal
    assert loaded.pending_remote_files()[0].file_id == "file-1"


def test_load_without_state_file_explains_itself(tmp_path: Path):
    with pytest.raises(StateError, match="ocr-batch submit"):
        RunState.load(tmp_path)


def test_unknown_version_is_refused(tmp_path: Path):
    (tmp_path / "_state.json").write_text(json.dumps({"version": 99}), encoding="utf-8")

    with pytest.raises(StateError, match="state version"):
        RunState.load(tmp_path)


def test_write_atomic_leaves_no_temp_files(tmp_path: Path):
    target = tmp_path / "sub" / "f.json"
    write_atomic(target, "{}")

    assert target.read_text(encoding="utf-8") == "{}"
    assert list(tmp_path.glob("sub/.*tmp")) == []


def test_results_path_gains_job_id_only_when_multiple_jobs(tmp_path: Path):
    state = RunState.create(output_dir=tmp_path, input_dir=Path("/in"), model="m")
    state.jobs.append(JobState("job-1"))

    assert state.results_path("job-1").name == "_mistral_batch_results.jsonl"

    state.jobs.append(JobState("job-2"))

    assert state.results_path("job-2").name == "_mistral_batch_results-job-2.jsonl"


def test_manifest_is_written_sorted(tmp_path: Path):
    state = RunState.create(output_dir=tmp_path, input_dir=Path("/in"), model="m")
    state.documents["b"] = DocumentState("b", "b.pdf", "h2")
    state.documents["a"] = DocumentState("a", "a.pdf", "h1")
    state.write_manifest()

    manifest = json.loads((tmp_path / "_manifest.json").read_text(encoding="utf-8"))

    assert list(manifest) == ["a", "b"]
    assert manifest["a"]["relative_path"] == "a.pdf"
