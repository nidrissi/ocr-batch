import json
from pathlib import Path

from conftest import batch_line

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


def test_error_detail_is_recorded(tmp_path: Path):
    state = make_state(tmp_path, ("id1", "a.pdf"))
    results = tmp_path / "r.jsonl"
    results.write_text(
        json.dumps({"custom_id": "id1", "error": {"message": "rate limited"}}) + "\n",
        encoding="utf-8",
    )

    split_results(results, state)

    assert "rate limited" in (state.documents["id1"].ocr_error or "")
