"""Phase 0: everything that can be checked without spending a cent.

Runs the CLI against the generated corpus with `--no-ocr`, plus the collision
tree with OCR nominally enabled and a bogus API key -- if the run were to reach
the network, that key would fail it.

    uv run python scripts/make_corpus.py corpus
    uv run python scripts/phase0.py corpus
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pymupdf

EXIT_ERROR = 1
EXIT_PARTIAL = 3

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}{f' -- {detail}' if detail and not ok else ''}")

    if not ok:
        failures.append(label)


def run_cli(*args: str, api_key: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ | {"MISTRAL_API_KEY": api_key}

    return subprocess.run(
        [sys.executable, "-m", "ocr_batch", *args],
        capture_output=True,
        text=True,
        env=environment,
    )


def page_numbers(text: str) -> list[int]:
    return [int(number) for number in re.findall(r"^===== PAGE (\d+) =====$", text, re.M)]


def page_bodies(text: str) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"^===== PAGE \d+ =====$", text, flags=re.M)[1:]]


def source_pages(path: Path) -> int:
    with pymupdf.open(path) as doc:
        return doc.page_count


# ----------------------------------------------------------------------


def check_collision(corpus: Path, work: Path) -> None:
    print("\n-- collision tree: must abort before any upload --")

    out = work / "out-collision"
    result = run_cli("submit", str(corpus / "collision"), str(out), api_key="phase0-bogus-key")

    check("exits 1", result.returncode == EXIT_ERROR, f"got {result.returncode}")
    check(
        "names both colliding sources",
        "a.pdf" in result.stderr and "a.PDF" in result.stderr,
        result.stderr.strip()[-200:],
    )
    check("wrote no state file", not (out / "_state.json").exists())
    check(
        "never reached the API",
        "401" not in result.stderr and "Unauthorized" not in result.stderr,
        result.stderr.strip()[-200:],
    )


def check_main_tree(corpus: Path, work: Path) -> None:
    print("\n-- main tree, --no-ocr --")

    tree = corpus / "input"
    out = work / "out-native"
    result = run_cli("submit", str(tree), str(out), "--no-ocr", api_key="phase0-bogus-key")

    check(
        "exits 3 (locked.pdf + broken.pdf failed)",
        result.returncode == EXIT_PARTIAL,
        f"got {result.returncode}: {result.stderr.strip()[-300:]}",
    )

    # -- output layout ------------------------------------------------

    expected = {
        "simple.native.txt": source_pages(tree / "simple.pdf"),
        "scanned.native.txt": source_pages(tree / "scanned.pdf"),
        "big.native.txt": source_pages(tree / "big.pdf"),
        "truncated.native.txt": source_pages(tree / "truncated.pdf"),
        "dossier été/smith.cv.final.native.txt": source_pages(
            tree / "dossier été" / "smith.cv.final.pdf"
        ),
    }

    for name in expected:
        check(f"wrote {name}", (out / name).is_file())

    check(
        "did not truncate smith.cv.final to smith.cv",
        not (out / "dossier été" / "smith.cv.native.txt").exists(),
    )
    check("no .native.txt for locked.pdf", not (out / "locked.native.txt").exists())
    check("no .native.txt for broken.pdf", not (out / "broken.native.txt").exists())
    check(
        "wrote no OCR outputs",
        not list(out.rglob("*.ocr.md")) and not list(out.rglob("*.ocr.json")),
    )

    # -- page separators ----------------------------------------------

    for name, pages in expected.items():
        path = out / name

        if not path.is_file():
            continue

        numbers = page_numbers(path.read_text(encoding="utf-8"))
        check(
            f"{name}: pages 1..{pages} in order",
            numbers == list(range(1, pages + 1)),
            f"got {numbers}",
        )

    scanned = out / "scanned.native.txt"

    if scanned.is_file():
        check(
            "scanned.pdf yields an empty text layer",
            page_bodies(scanned.read_text(encoding="utf-8")) == [""],
        )

    # -- state and manifest -------------------------------------------

    state = json.loads((out / "_state.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    by_path = {doc["relative_path"]: doc for doc in state["documents"].values()}

    check("manifest covers 7 documents", len(manifest) == 7, f"got {len(manifest)}")
    check("state covers 7 documents", len(by_path) == 7, f"got {sorted(by_path)}")
    check("recorded no jobs", state["jobs"] == [])
    check("recorded no uploads", state["remote_files"] == [])

    check(
        "locked.pdf reports a password error",
        "password" in (by_path.get("locked.pdf", {}).get("native_error") or "").lower(),
        repr(by_path.get("locked.pdf", {}).get("native_error")),
    )
    check(
        "broken.pdf reports an extraction error",
        bool(by_path.get("broken.pdf", {}).get("native_error")),
        repr(by_path.get("broken.pdf", {}).get("native_error")),
    )

    for name, pages in expected.items():
        relative = name.replace(".native.txt", ".pdf")
        document = by_path.get(relative, {})

        check(
            f"{relative}: native_pages == {pages}, no error",
            document.get("native_pages") == pages and document.get("native_error") is None,
            f"pages={document.get('native_pages')} error={document.get('native_error')!r}",
        )

    check(
        "every document has a content hash",
        all(len(doc.get("sha256", "")) == 64 for doc in by_path.values()),
    )

    # -- idempotency ---------------------------------------------------

    print("\n-- rerun: successes untouched, then --force --")

    stamps = {name: (out / name).stat().st_mtime_ns for name in expected}
    again = run_cli("submit", str(tree), str(out), "--no-ocr", api_key="phase0-bogus-key")

    check(
        "rerun does not rewrite existing outputs",
        all((out / name).stat().st_mtime_ns == stamp for name, stamp in stamps.items()),
    )
    check("rerun still exits 3", again.returncode == EXIT_PARTIAL, f"got {again.returncode}")
    check(
        "rerun retries the two failures",
        again.stderr.count("native extraction failed") == 2,
        again.stderr.strip()[-300:],
    )

    stamp = stamps["simple.native.txt"]
    forced = run_cli(
        "submit", str(tree), str(out), "--no-ocr", "--force", api_key="phase0-bogus-key"
    )
    check("--force redoes extraction", (out / "simple.native.txt").stat().st_mtime_ns != stamp)
    check("--force still exits 3", forced.returncode == EXIT_PARTIAL, f"got {forced.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("corpus"))
    parser.add_argument("--work", type=Path, default=None, help="where to put output dirs")
    args = parser.parse_args()

    corpus = args.corpus.resolve()
    work = (args.work or corpus / "_phase0").resolve()

    if not (corpus / "input").is_dir():
        print(f"no corpus at {corpus}; run scripts/make_corpus.py first", file=sys.stderr)
        return EXIT_ERROR

    if work.exists():
        shutil.rmtree(work)

    work.mkdir(parents=True)

    check_collision(corpus, work)
    check_main_tree(corpus, work)

    print(f"\n{len(failures)} failed" if failures else "\nall checks passed")

    for label in failures:
        print(f"  {label}")

    return EXIT_ERROR if failures else 0


if __name__ == "__main__":
    sys.exit(main())
