"""Source discovery and the single source of truth for output file names.

Both the native and the OCR side derive their paths from `output_paths`, so a
document's three outputs can never disagree about how its name was built.
"""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ID_LENGTH = 24
_HASH_CHUNK = 1 << 20


def find_pdfs(input_dir: Path) -> list[Path]:
    """Return every PDF under `input_dir`, sorted, matching the suffix case-insensitively."""
    return sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"
    )


def make_id(rel: Path) -> str:
    """Derive the stable batch `custom_id` for a document.

    Hashes the POSIX form so ids computed on Windows match ids computed on
    Linux for the same relative path.
    """
    return hashlib.sha256(rel.as_posix().encode("utf-8")).hexdigest()[:ID_LENGTH]


def file_sha256(path: Path) -> str:
    """Content hash of a file, read in chunks so large PDFs stay off the heap."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)

    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class OutputPaths:
    base: Path
    native: Path
    ocr_md: Path
    ocr_json: Path


def output_paths(output_dir: Path, rel: Path) -> OutputPaths:
    """Map a source path relative to the input dir onto its three outputs.

    Only the `.pdf` suffix is stripped -- never `Path.with_suffix`, which would
    turn `smith.cv.final.pdf` into `smith.cv.ocr.md` and collide with
    `smith.cv.pdf`.
    """
    name = rel.name[: -len(rel.suffix)] if rel.suffix else rel.name
    base = output_dir / rel.parent / name

    return OutputPaths(
        base=base,
        native=base.with_name(f"{base.name}.native.txt"),
        ocr_md=base.with_name(f"{base.name}.ocr.md"),
        ocr_json=base.with_name(f"{base.name}.ocr.json"),
    )


def find_collisions(
    rels: Iterable[Path],
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Find source paths that share an output base.

    Returns `(exact, case_insensitive)`. Exact clashes always destroy data;
    case-only clashes destroy data solely on a case-insensitive filesystem, so
    the caller reports them as a warning.
    """
    exact: list[tuple[Path, Path]] = []
    folded: list[tuple[Path, Path]] = []
    by_key: dict[str, Path] = {}
    by_folded_key: dict[str, Path] = {}

    for rel in rels:
        key = output_paths(Path(), rel).base.as_posix()

        if (other := by_key.get(key)) is not None:
            exact.append((other, rel))
        else:
            by_key[key] = rel

            if (other := by_folded_key.get(key.lower())) is not None:
                folded.append((other, rel))
            else:
                by_folded_key[key.lower()] = rel

    return exact, folded
