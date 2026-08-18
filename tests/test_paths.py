from pathlib import Path

import pytest

from ocr_batch.paths import find_collisions, find_pdfs, make_id, output_paths


def test_dotted_names_keep_every_component():
    """Regression: with_suffix() used to turn smith.cv.final.pdf into smith.cv.*."""
    paths = output_paths(Path("/out"), Path("a/smith.cv.final.pdf"))

    assert paths.ocr_md == Path("/out/a/smith.cv.final.ocr.md")
    assert paths.ocr_json == Path("/out/a/smith.cv.final.ocr.json")
    assert paths.native == Path("/out/a/smith.cv.final.native.txt")


def test_dotted_and_plain_names_do_not_collide():
    dotted = output_paths(Path("/out"), Path("smith.cv.pdf"))
    plain = output_paths(Path("/out"), Path("smith.pdf"))

    assert dotted.ocr_md != plain.ocr_md


def test_uppercase_suffix_is_stripped():
    assert output_paths(Path("/out"), Path("SCAN.PDF")).ocr_md == Path("/out/SCAN.ocr.md")


def test_find_collisions_separates_exact_from_case_only():
    exact, folded = find_collisions([Path("a.pdf"), Path("a.pdf"), Path("A.PDF"), Path("b.pdf")])

    assert exact == [(Path("a.pdf"), Path("a.pdf"))]
    assert folded == [(Path("a.pdf"), Path("A.PDF"))]


def test_make_id_is_stable_and_platform_independent():
    assert make_id(Path("a/b.pdf")) == make_id(Path("a") / "b.pdf")
    assert make_id(Path("a/b.pdf")) != make_id(Path("a/c.pdf"))
    assert len(make_id(Path("a/b.pdf"))) == 24


def test_find_pdfs_is_case_insensitive_and_sorted(tmp_path: Path):
    for name in ("z.pdf", "a.PDF", "nested/m.Pdf", "notes.txt"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")

    found = [p.relative_to(tmp_path).as_posix() for p in find_pdfs(tmp_path)]

    assert found == ["a.PDF", "nested/m.Pdf", "z.pdf"]


@pytest.mark.parametrize("name", ["a.pdf", "a.b.c.pdf"])
def test_all_three_outputs_share_one_base(name: str):
    paths = output_paths(Path("/out"), Path(name))

    assert paths.native.name.startswith(paths.base.name)
    assert paths.ocr_md.name.startswith(paths.base.name)
    assert paths.ocr_json.name.startswith(paths.base.name)
