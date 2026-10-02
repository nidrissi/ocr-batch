import json
from pathlib import Path

import pytest
from conftest import batch_line

from ocr_batch.paths import output_paths
from ocr_batch.results import render_markdown, split_results
from ocr_batch.state import DocumentState, RunState


def make_state(tmp_path: Path, *docs: tuple[str, str]) -> RunState:
    state = RunState.create(output_dir=tmp_path, input_dir=Path("/in"), model="m")

    for custom_id, rel in docs:
        state.documents[custom_id] = DocumentState(custom_id, rel, "hash")

    return state


def test_pages_are_ordered_and_separated():
    body = {
        "pages": [
            {"index": 1, "markdown": "second"},
            {"index": 0, "markdown": "first"},
        ]
    }

    assert render_markdown(body) == (
        "===== PAGE 1 =====\n\nfirst\n\n===== PAGE 2 =====\n\nsecond\n"
    )


def test_split_writes_both_outputs(tmp_path: Path):
    state = make_state(tmp_path, ("id1", "nested/smith.cv.final.pdf"))
    results = tmp_path / "r.jsonl"
    results.write_text(batch_line("id1", pages=2) + "\n", encoding="utf-8")

    summary = split_results(results, state)

    assert summary.written == 1
    assert (tmp_path / "nested/smith.cv.final.ocr.md").exists()
    assert json.loads((tmp_path / "nested/smith.cv.final.ocr.json").read_text())["pages"]
    assert state.documents["id1"].ocr_written


def test_one_bad_line_does_not_abandon_the_rest(tmp_path: Path):
    state = make_state(tmp_path, ("id1", "a.pdf"), ("id2", "b.pdf"))
    results = tmp_path / "r.jsonl"
    results.write_text(
        "\n".join(
            [
                "{not json",
                json.dumps({"custom_id": "ghost", "response": {"body": {"pages": []}}}),
                json.dumps({"custom_id": "id1", "response": {"status_code": 500, "body": None}}),
                "",
                batch_line("id2"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = split_results(results, state)

    assert (summary.written, summary.failed, summary.unknown, summary.malformed) == (1, 1, 1, 1)
    assert (tmp_path / "b.ocr.md").exists()
    assert state.documents["id1"].ocr_error is not None
    assert not state.documents["id1"].ocr_written


def test_split_is_idempotent_and_forceable(tmp_path: Path):
    state = make_state(tmp_path, ("id1", "a.pdf"))
    results = tmp_path / "r.jsonl"
    results.write_text(batch_line("id1") + "\n", encoding="utf-8")

    assert split_results(results, state).written == 1
    assert split_results(results, state).skipped == 1
    assert split_results(results, state, force=True).written == 1


@pytest.mark.parametrize("missing", ["ocr_md", "ocr_json"])
def test_split_restores_either_missing_output(tmp_path: Path, missing: str):
    state = make_state(tmp_path, ("id1", "a.pdf"))
    results = tmp_path / "r.jsonl"
    results.write_text(batch_line("id1") + "\n", encoding="utf-8")
    paths = output_paths(tmp_path, Path("a.pdf"))

    split_results(results, state)
    getattr(paths, missing).unlink()

    summary = split_results(results, state)

    assert summary.written == 1
    assert paths.ocr_md.exists()
    assert paths.ocr_json.exists()


def test_error_detail_is_recorded(tmp_path: Path):
    state = make_state(tmp_path, ("id1", "a.pdf"))
    results = tmp_path / "r.jsonl"
    results.write_text(
        json.dumps({"custom_id": "id1", "error": {"message": "rate limited"}}) + "\n",
        encoding="utf-8",
    )

    split_results(results, state)

    assert "rate limited" in (state.documents["id1"].ocr_error or "")


@pytest.mark.parametrize("status", [400, 429, 500])
def test_http_error_body_is_not_written_as_success(tmp_path: Path, status: int):
    state = make_state(tmp_path, ("id1", "a.pdf"))
    results = tmp_path / "r.jsonl"
    results.write_text(
        json.dumps(
            {"custom_id": "id1", "response": {"status_code": status, "body": {"message": "boom"}}}
        )
        + "\n"
    )

    summary = split_results(results, state)

    assert summary.failed == 1
    assert not state.documents["id1"].ocr_written
    assert "boom" in (state.documents["id1"].ocr_error or "")
    assert not (tmp_path / "a.ocr.md").exists()
    assert not (tmp_path / "a.ocr.json").exists()


@pytest.mark.parametrize(
    "bad_row",
    [
        [],
        None,
        42,
        {"custom_id": ["id1"]},
        {"custom_id": "id1", "response": "invalid"},
        {"custom_id": "id1", "response": {"body": {"message": "not OCR"}}},
        {"custom_id": "id1", "response": {"body": {"pages": [None]}}},
        {"custom_id": "id1", "response": {"body": {"pages": [{"index": "zero"}]}}},
        {"custom_id": "id1", "response": {"body": {"pages": [{"markdown": 42}]}}},
    ],
)
def test_wrong_shaped_rows_do_not_abandon_good_results(tmp_path: Path, bad_row: object):
    state = make_state(tmp_path, ("id1", "a.pdf"), ("id2", "b.pdf"))
    results = tmp_path / "r.jsonl"
    results.write_text(json.dumps(bad_row) + "\n" + batch_line("id2") + "\n")

    summary = split_results(results, state)

    assert summary.malformed == 1
    assert summary.written == 1
    assert (tmp_path / "b.ocr.md").is_file()
    assert not (tmp_path / "a.ocr.json").exists()
