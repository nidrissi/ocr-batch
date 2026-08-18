from pathlib import Path

from conftest import write_pdf

from ocr_batch.native import NativePool, extract_one


def test_extract_writes_page_separated_text(tmp_path: Path):
    source = write_pdf(tmp_path / "a.pdf", "hello", pages=2)
    destination = tmp_path / "out" / "a.native.txt"

    result = extract_one("id1", str(source), str(destination))

    assert result.error is None
    assert result.pages == 2

    text = destination.read_text(encoding="utf-8")

    assert text.startswith("===== PAGE 1 =====")
    assert "===== PAGE 2 =====" in text
    assert "hello 2" in text


def test_a_corrupt_pdf_is_reported_not_raised(tmp_path: Path):
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not a pdf at all")

    result = extract_one("id1", str(source), str(tmp_path / "broken.native.txt"))

    assert result.pages == 0
    assert result.error
    assert not (tmp_path / "broken.native.txt").exists()


def test_pool_runs_every_task_and_survives_a_bad_one(tmp_path: Path):
    write_pdf(tmp_path / "good.pdf", "hi")
    (tmp_path / "bad.pdf").write_bytes(b"nope")

    with NativePool(max_workers=2) as pool:
        pool.submit("good", tmp_path / "good.pdf", tmp_path / "good.native.txt")
        pool.submit("bad", tmp_path / "bad.pdf", tmp_path / "bad.native.txt")

        results = {result.custom_id: result for result in pool.results()}

    assert results["good"].pages == 1
    assert results["bad"].error
