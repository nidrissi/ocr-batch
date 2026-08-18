"""Generate the live-test corpus.

Three trees, one per phase:

    <root>/smoke/      one 1-page PDF -- the Phase 1 API smoke test
    <root>/input/      the six-item main tree -- Phases 0 and 2
    <root>/collision/  a.pdf + a.PDF -- must be refused before any upload

Nothing here is committed: PDFs are generated so the corpus can be rebuilt
byte-for-byte on any machine.
"""

import argparse
import shutil
from pathlib import Path

import pymupdf

SCANNED_DPI = 150


def _text_pdf(path: Path, text: str, pages: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with pymupdf.open() as doc:
        for number in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"{text} {number + 1}")

        doc.save(path)


def _scanned_pdf(path: Path) -> None:
    """A page rendered to a bitmap, so the PDF carries no text layer at all."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with pymupdf.open() as source:
        page = source.new_page()
        page.insert_text((72, 72), "SCANNED PAGE ONE")
        pixmap = page.get_pixmap(dpi=SCANNED_DPI)
        # PNG rather than the raw pixmap: a bare pixmap embeds uncompressed.
        image = pixmap.tobytes("png")

    with pymupdf.open() as doc:
        page = doc.new_page(width=pixmap.width, height=pixmap.height)
        page.insert_image(pymupdf.Rect(0, 0, pixmap.width, pixmap.height), stream=image)
        doc.save(path, deflate=True)


def _locked_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "LOCKED PAGE ONE")
        doc.save(
            path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw="secret",
            owner_pw="secret",
        )


def _truncated_pdf(path: Path, source: Path) -> None:
    """Half a real PDF. MuPDF repairs this rather than refusing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source.read_bytes()[:500])


def build(root: Path, big_pages: int) -> None:
    if root.exists():
        shutil.rmtree(root)

    # Phase 1: the single page that answers the API-contract questions.
    _text_pdf(root / "smoke" / "smoke.pdf", "SMOKE PAGE", 1)

    # Phase 0 / 2: the main tree.
    tree = root / "input"
    _text_pdf(tree / "simple.pdf", "SIMPLE PAGE", 2)
    _scanned_pdf(tree / "scanned.pdf")
    _locked_pdf(tree / "locked.pdf")
    (tree / "broken.pdf").write_bytes(b"not a pdf at all")
    _truncated_pdf(tree / "truncated.pdf", tree / "simple.pdf")
    _text_pdf(tree / "dossier été" / "smith.cv.final.pdf", "SMITH PAGE", 1)
    _text_pdf(tree / "big.pdf", "BIG PAGE", big_pages)

    # Phase 0: both strip to the same output base and must abort the run.
    _text_pdf(root / "collision" / "a.pdf", "COLLIDE LOWER", 1)
    _text_pdf(root / "collision" / "a.PDF", "COLLIDE UPPER", 1)

    for pdf in sorted(root.rglob("*")):
        if pdf.is_file():
            print(f"{pdf.relative_to(root)}  ({pdf.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("corpus"))
    parser.add_argument(
        "--big-pages",
        type=int,
        default=24,
        help="pages in big.pdf; these dominate Phase 2 cost (default: %(default)s)",
    )
    args = parser.parse_args()

    build(args.root.resolve(), args.big_pages)


if __name__ == "__main__":
    main()
