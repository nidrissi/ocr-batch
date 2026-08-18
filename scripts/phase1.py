"""Phase 1: the one-page live smoke test.

Uploads a single 1-page PDF to Mistral and runs the whole pipeline against it,
then checks the four things only a real run can settle:

    1. the batch worker can read a signed URL minted for a `purpose: "ocr"` file
    2. the job-level `model` satisfies the per-request `model` field
    3. `files.upload(expiry=...)` is denominated in hours, not days
    4. `files.download(...).iter_bytes()` works on the real httpx response

This costs money and sends a PDF to a third party, so it will not run without
an explicit --yes.

    uv run python scripts/make_corpus.py corpus
    uv run python scripts/phase1.py corpus --yes
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from mistralai.client import Mistral

EXIT_ERROR = 1

# What the smoke PDF says, so we can prove the OCR really read our document.
SMOKE_TEXT = "SMOKE PAGE 1"

# Deliberately not 24 (the API default) and not 168 (the cap), so the echoed
# expires_at can only match a request that was read as hours.
PROBE_EXPIRY_HOURS = 48

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}{f' -- {detail}' if detail and not ok else ''}")

    if not ok:
        failures.append(label)


def note(label: str, value: object) -> None:
    print(f"note  {label}: {value}")


def as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, UTC)

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

    return None


# ----------------------------------------------------------------------


def probe_upload_expiry(client: Mistral, pdf: Path) -> None:
    """Question 3: upload one file, read back its expiry, delete it again."""
    print("\n-- probe: unit of files.upload(expiry=) --")

    uploaded = client.files.upload(
        file={"file_name": pdf.name, "content": pdf.read_bytes()},
        purpose="ocr",
        visibility="user",
        expiry=PROBE_EXPIRY_HOURS,
    )

    try:
        created = as_datetime(getattr(uploaded, "created_at", None))
        expires = as_datetime(getattr(uploaded, "expires_at", None))

        note("created_at", getattr(uploaded, "created_at", None))
        note("expires_at", getattr(uploaded, "expires_at", None))

        if created is None or expires is None:
            check(
                f"expiry of {PROBE_EXPIRY_HOURS} is read as hours",
                False,
                "API did not echo both timestamps -- check by hand",
            )
            return

        hours = (expires - created).total_seconds() / 3600
        note("lifetime in hours", round(hours, 2))
        check(
            f"expiry of {PROBE_EXPIRY_HOURS} is read as hours, not days",
            abs(hours - PROBE_EXPIRY_HOURS) < 1,
            f"got {hours:.2f}h; {PROBE_EXPIRY_HOURS} days would be {PROBE_EXPIRY_HOURS * 24}h",
        )
    finally:
        client.files.delete(file_id=uploaded.id)
        print(f"note  probe upload {uploaded.id} deleted")


def run_pipeline(smoke: Path, out: Path) -> subprocess.CompletedProcess[str]:
    print("\n-- ocr-batch run (one page) --")

    result = subprocess.run(
        [sys.executable, "-m", "ocr_batch", "run", str(smoke), str(out), "-v"],
        capture_output=True,
        text=True,
    )

    print(result.stderr.rstrip())
    check("exits 0", result.returncode == 0, f"got {result.returncode}")

    return result


def check_artifacts(out: Path) -> list[str]:
    """Check what landed on disk; returns the uploaded file ids for the sweep."""
    print("\n-- artifacts --")

    state = json.loads((out / "_state.json").read_text(encoding="utf-8"))
    documents = list(state["documents"].values())

    check("recorded exactly one job", len(state["jobs"]) == 1, f"got {len(state['jobs'])}")

    job = state["jobs"][0] if state["jobs"] else {}
    check("job status is SUCCESS", job.get("status") == "SUCCESS", repr(job.get("status")))
    check("job has an output file", bool(job.get("output_file")))
    check("job is marked fetched", job.get("fetched") is True)

    # Question 4: this file only exists if iter_bytes streamed the real response.
    results = out / "_mistral_batch_results.jsonl"
    check("downloaded the results JSONL", results.is_file() and results.stat().st_size > 0)

    md = out / "smoke.ocr.md"
    raw = out / "smoke.ocr.json"
    check("wrote smoke.ocr.md", md.is_file())
    check("wrote smoke.ocr.json", raw.is_file())
    check("wrote smoke.native.txt", (out / "smoke.native.txt").is_file())

    if md.is_file():
        text = md.read_text(encoding="utf-8")
        check("OCR markdown is page-separated", "===== PAGE 1 =====" in text)
        # Question 1: our own words coming back proves the batch worker fetched
        # the signed URL of a purpose="ocr" upload and read the PDF behind it.
        check(
            "OCR returned the text of our document",
            SMOKE_TEXT.lower() in re.sub(r"\s+", " ", text).lower(),
            f"looked for {SMOKE_TEXT!r} in: {text[:300]!r}",
        )

    if raw.is_file():
        body = json.loads(raw.read_text(encoding="utf-8"))
        pages = body.get("pages") or []

        # Question 2: a request whose `model` was never set by build_request
        # only produces pages if the job-level model was applied to it.
        check("response carries one page", len(pages) == 1, f"got {len(pages)}")
        note("response keys", sorted(body))

        if pages:
            page = pages[0]
            note("page keys", sorted(page))
            check("page has markdown", bool((page.get("markdown") or "").strip()))
            check("--include-blocks produced blocks", bool(page.get("blocks")))
            check(
                "--confidence-granularity produced scores",
                page.get("confidence_scores") is not None,
            )

        check("no annotation fields were requested", "document_annotation" not in body)

    for document in documents:
        check("document has no OCR error", not document.get("ocr_error"), repr(document))
        check("document marked ocr_written", document.get("ocr_written") is True)

    remote = state["remote_files"]
    check("all uploads marked deleted", all(entry["deleted"] for entry in remote), repr(remote))

    return [entry["file_id"] for entry in remote]


def check_nothing_left(client: Mistral, file_ids: list[str]) -> None:
    """Nothing in the CLI verifies the deletes landed, so verify them here."""
    print("\n-- remote sweep --")

    remaining = {file.id for file in client.files.list(purpose="ocr", page_size=100).data}

    check(
        "no smoke upload is still on Mistral",
        not (set(file_ids) & remaining),
        f"still present: {sorted(set(file_ids) & remaining)}",
    )
    note("other purpose=ocr files in this workspace", len(remaining - set(file_ids)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("corpus"))
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument(
        "--yes", action="store_true", help="required: uploads a PDF to Mistral and spends money"
    )
    args = parser.parse_args()

    if not args.yes:
        print(__doc__)
        print("refusing to run without --yes", file=sys.stderr)
        return EXIT_ERROR

    if not os.environ.get("MISTRAL_API_KEY"):
        print("MISTRAL_API_KEY is not set", file=sys.stderr)
        return EXIT_ERROR

    corpus = args.corpus.resolve()
    smoke = corpus / "smoke"
    pdf = smoke / "smoke.pdf"

    if not pdf.is_file():
        print(f"no {pdf}; run scripts/make_corpus.py first", file=sys.stderr)
        return EXIT_ERROR

    out = (args.work or corpus / "_phase1").resolve()

    if out.exists():
        shutil.rmtree(out)

    with Mistral(api_key=os.environ["MISTRAL_API_KEY"]) as client:
        probe_upload_expiry(client, pdf)
        run_pipeline(smoke, out)

        file_ids = check_artifacts(out) if (out / "_state.json").is_file() else []

        check_nothing_left(client, file_ids)

    print(f"\n{len(failures)} failed" if failures else "\nall checks passed")

    for label in failures:
        print(f"  {label}")

    if failures:
        print(f"\nartifacts kept in {out} for inspection")

    return EXIT_ERROR if failures else 0


if __name__ == "__main__":
    sys.exit(main())
