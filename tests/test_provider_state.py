import json
from pathlib import Path

import pytest
from conftest import FakeJob, FakeMistral, batch_line, write_pdf

from ocr_batch import cli
from ocr_batch.errors import ConfigError, RemoteError, StateError
from ocr_batch.paths import file_sha256, make_id, output_paths
from ocr_batch.remote import OcrOptions
from ocr_batch.state import RemoteFile, RunState


def unexpected_client(*args: object, **kwargs: object) -> None:
    pytest.fail("unexpected remote client")


def test_v1_outstanding_job_and_upload_survive_migration(
    tmp_path: Path, patched_client: FakeMistral, monkeypatch: pytest.MonkeyPatch
):
    source = write_pdf(tmp_path / "in" / "a.pdf", "source")
    out = tmp_path / "out"
    out.mkdir()
    document_id = make_id(Path("a.pdf"))
    state_path = out / "_state.json"
    legacy = {
        "version": 1,
        "input_dir": str(source.parent),
        "model": cli.DEFAULT_MODEL,
        "documents": {
            document_id: {
                "custom_id": document_id,
                "relative_path": source.name,
                "sha256": file_sha256(source),
                "ocr_written": False,
            }
        },
        "jobs": [{"job_id": "old-batch", "custom_ids": [document_id], "status": "RUNNING"}],
        "remote_files": [
            {"file_id": "old-upload", "job_id": "old-batch", "custom_id": document_id}
        ],
    }
    state_path.write_text(json.dumps(legacy), encoding="utf-8")
    before = state_path.read_bytes()
    monkeypatch.setattr(cli, "make_openai_client", unexpected_client)
    patched_client.jobs["old-batch"] = FakeJob(
        "old-batch", status="SUCCESS", total_requests=1, succeeded_requests=1, output_file="old-out"
    )
    patched_client.downloads["old-out"] = (batch_line(document_id, "legacy OCR") + "\n").encode()

    loaded = RunState.load(out)
    assert loaded.provider == "mistral"
    assert loaded.jobs[0].configuration is None
    assert state_path.read_bytes() == before
    with pytest.raises(StateError, match="running batch job"):
        cli.do_submit(source.parent, out, cli.SubmitOptions(native=False, force=True))
    assert state_path.read_bytes() == before

    assert cli.do_status(out) == cli.EXIT_OK
    assert cli.do_fetch(out, keep_remote=True) == cli.EXIT_OK
    migrated = RunState.load(out)
    assert json.loads(state_path.read_text(encoding="utf-8"))["version"] == 2
    assert migrated.documents[document_id].ocr_written
    assert migrated.documents[document_id].ocr_config is None
    assert [remote.file_id for remote in migrated.pending_remote_files()] == ["old-upload"]
    assert cli.do_cleanup(out) == cli.EXIT_OK
    assert patched_client.deleted == ["old-upload"]

    paths = output_paths(out, Path("a.pdf"))
    assert "legacy OCR" in paths.ocr_md.read_text(encoding="utf-8")
    old_files = paths.ocr_md.read_bytes(), paths.ocr_json.read_bytes()
    state = cli.do_submit(source.parent, out, cli.SubmitOptions(native=False))

    # The legacy model string is known, but the old OCR options are unknown.
    assert len(state.jobs) == 1
    assert len(patched_client.submitted) == 1
    assert state.documents[document_id].ocr_config is None
    assert (paths.ocr_md.read_bytes(), paths.ocr_json.read_bytes()) == old_files


@pytest.mark.parametrize(
    ("provider", "env_var", "other_var", "default_model"),
    [
        ("mistral", "MISTRAL_API_KEY", "OPENAI_API_KEY", "mistral-ocr-4-1"),
        ("openai", "OPENAI_API_KEY", "MISTRAL_API_KEY", "gpt-6.1-terra"),
    ],
)
def test_credentials_and_defaults_follow_selected_provider(
    provider: str,
    env_var: str,
    other_var: str,
    default_model: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = write_pdf(tmp_path / "in" / "a.pdf", "source")
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "make_client", unexpected_client)
    monkeypatch.setattr(cli, "make_openai_client", unexpected_client)
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv(other_var, "wrong-provider")

    assert cli.SubmitOptions(provider=provider).model == default_model
    assert cli.resolve_api_key("override", provider=provider) == "override"
    with pytest.raises(ConfigError, match=env_var):
        cli.do_submit(source.parent, out, cli.SubmitOptions(provider=provider, native=False))
    assert not (out / "_state.json").exists()

    monkeypatch.setenv(env_var, "selected-provider")
    assert cli.resolve_api_key(provider=provider) == "selected-provider"
    assert cli.resolve_api_key("override", provider=provider) == "override"


def test_local_only_does_not_need_credentials_or_bind_remote_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = write_pdf(tmp_path / "in" / "a.pdf", "native text")
    out = tmp_path / "out"
    for env_var in ("OPENAI_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(cli, "make_client", unexpected_client)
    monkeypatch.setattr(cli, "make_openai_client", unexpected_client)

    assert (
        cli.main(
            ["run", str(source.parent), str(out), "--provider", "openai", "--no-ocr", "--jobs", "1"]
        )
        == cli.EXIT_OK
    )
    assert RunState.load(out).provider is None
    assert "native text" in (out / "a.native.txt").read_text(encoding="utf-8")
    assert cli.do_status(out) == cli.EXIT_OK
    assert cli.do_cleanup(out) == cli.EXIT_OK


@pytest.mark.parametrize(("previous", "selected"), [("mistral", "openai"), ("openai", "mistral")])
def test_provider_conflict_under_force_has_no_side_effects(
    previous: str, selected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = write_pdf(tmp_path / "in" / "a.pdf", "source")
    out = tmp_path / "out"
    state = RunState.create(
        output_dir=out, input_dir=source.parent, model="model", provider=previous
    )
    state.remote_files.append(RemoteFile("retained-upload"))
    state.save()
    state.write_manifest()
    (out / "a.ocr.md").write_text("existing output", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    monkeypatch.setattr(cli, "make_client", unexpected_client)
    monkeypatch.setattr(cli, "make_openai_client", unexpected_client)

    with pytest.raises(StateError, match="provider"):
        cli.do_submit(source.parent, out, cli.SubmitOptions(provider=selected, force=True), "key")

    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


@pytest.mark.parametrize("change", ["model", "options", "source"])
def test_retry_after_failed_replacement_keeps_ocr_provenance(
    change: str, tmp_path: Path, patched_client: FakeMistral
):
    source = write_pdf(tmp_path / "in" / "a.pdf", "old source")
    out = tmp_path / "out"
    paths = output_paths(out, Path("a.pdf"))
    first = cli.do_submit(source.parent, out, cli.SubmitOptions(native=False))
    (document_id,) = first.documents
    patched_client.downloads["out-1"] = (batch_line(document_id, "old OCR") + "\n").encode()
    assert cli.do_fetch(out) == cli.EXIT_OK
    completed = RunState.load(out).documents[document_id]
    assert completed.ocr_config is not None
    assert completed.ocr_config["provider"] == "mistral"
    assert completed.ocr_config["model"] == cli.DEFAULT_MODEL
    assert completed.ocr_config["ocr_options"]["table_format"] == "markdown"
    assert completed.ocr_sha256 == file_sha256(source)
    old_files = paths.ocr_md.read_bytes(), paths.ocr_json.read_bytes()

    if change == "source":
        write_pdf(source, "new source")
    options = cli.SubmitOptions(
        native=False,
        model="new-model" if change == "model" else None,
        ocr_options=OcrOptions(table_format="html" if change == "options" else "markdown"),
    )
    patched_client.fail_create_at = {1}
    with pytest.raises(RemoteError, match="create boom"):
        cli.do_submit(source.parent, out, options)

    def assert_prior_output() -> None:
        document = RunState.load(out).documents[document_id]
        assert document.ocr_config == completed.ocr_config
        assert document.ocr_sha256 == completed.ocr_sha256
        assert (paths.ocr_md.read_bytes(), paths.ocr_json.read_bytes()) == old_files

    assert_prior_output()
    patched_client.fail_create_at.clear()
    replacement = cli.do_submit(source.parent, out, options)
    (job,) = replacement.jobs
    assert job.configuration == replacement.configuration
    if change != "source":
        assert job.configuration != completed.ocr_config
    assert_prior_output()

    patched_client.downloads["out-1"] = (
        json.dumps({"custom_id": document_id, "error": "OCR failed"}) + "\n"
    ).encode()
    assert cli.do_fetch(out) == cli.EXIT_PARTIAL
    assert_prior_output()

    retried = cli.do_submit(source.parent, out, options)
    assert len(retried.jobs) == 1
    patched_client.downloads["out-1"] = (batch_line(document_id, "new OCR") + "\n").encode()
    assert cli.do_fetch(out) == cli.EXIT_OK
    final = RunState.load(out).documents[document_id]
    assert final.ocr_config == job.configuration
    assert final.ocr_sha256 == file_sha256(source)
    assert "new OCR" in paths.ocr_md.read_text(encoding="utf-8")
