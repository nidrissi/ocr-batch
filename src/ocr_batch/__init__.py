import json
import os
import time
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf
from mistralai.client import Mistral


MODEL = "mistral-ocr-4-0"


# --------------------------------------------------------------------
# Local extraction -- executed in child processes
# --------------------------------------------------------------------


def pymupdf_extract(pdf_path: str, output_path: str):
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    pages = []

    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text", sort=True).strip()

            pages.append(f"===== PAGE {i + 1} =====\n\n{text}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n\n".join(pages) + "\n",
        encoding="utf-8",
    )

    return str(pdf_path), len(pages)


# --------------------------------------------------------------------
# Stable IDs so batch output can be mapped back to source files
# --------------------------------------------------------------------


def make_id(relpath: Path) -> str:
    return hashlib.sha256(str(relpath).encode("utf-8")).hexdigest()[:24]


# --------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------


def run(input_dir: Path, output_dir: Path):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.rglob("*.pdf"))

    print(f"{len(pdfs)} PDFs")

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # --------------------------------------------------------------
    # A. Immediately start local extraction in other processes
    # --------------------------------------------------------------

    pool = ProcessPoolExecutor()

    native_futures = []

    for pdf in pdfs:
        rel = pdf.relative_to(input_dir)

        native_out = output_dir / rel.parent / f"{pdf.stem}.native.txt"

        native_futures.append(
            pool.submit(
                pymupdf_extract,
                str(pdf),
                str(native_out),
            )
        )

    # --------------------------------------------------------------
    # B. Meanwhile upload PDFs for the Mistral OCR batch
    # --------------------------------------------------------------

    requests = []
    manifest = {}
    remote_file_ids = []

    for n, pdf in enumerate(pdfs, 1):
        rel = pdf.relative_to(input_dir)
        custom_id = make_id(rel)

        print(f"upload {n}/{len(pdfs)}: {rel}")

        with pdf.open("rb") as f:
            uploaded = client.files.upload(
                file={
                    "file_name": pdf.name,
                    "content": f,
                },
                purpose="ocr",
                visibility="user",
            )

        remote_file_ids.append(uploaded.id)

        # Maximum currently supported expiry: 168 hours.
        signed = client.files.get_signed_url(
            file_id=uploaded.id,
            expiry=168,
        )

        manifest[custom_id] = {
            "relative_path": str(rel),
            "remote_file_id": uploaded.id,
        }

        requests.append(
            {
                "custom_id": custom_id,
                "body": {
                    "document": {
                        "type": "document_url",
                        "document_url": signed.url,
                    },
                    # Useful for later QA/reconciliation.
                    "include_blocks": True,
                    "confidence_scores_granularity": "block",
                    # Don't inflate output with rendered images.
                    "include_image_base64": False,
                    # NOT using Mistral Annotations.
                    # No document_annotation_format.
                    # No bbox_annotation_format.
                    "extract_header": False,
                    "extract_footer": False,
                    "table_format": "markdown",
                },
            }
        )

    # Keep the manifest permanently.
    (output_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # C. Submit ONE asynchronous discounted OCR job
    # --------------------------------------------------------------

    job = client.batch.jobs.create(
        requests=requests,
        model=MODEL,
        endpoint="/v1/ocr",
        timeout_hours=24,
        metadata={"purpose": "math-job-applications-ocr"},
    )

    print(f"Mistral batch job: {job.id}")

    # --------------------------------------------------------------
    # D. PyMuPDF is already running while we wait for Mistral.
    # --------------------------------------------------------------

    while True:
        status = client.batch.jobs.get(job_id=job.id)

        print(
            f"Mistral: {status.status}: "
            f"{status.succeeded_requests}/"
            f"{status.total_requests}, "
            f"{status.failed_requests} failed"
        )

        if status.status not in ("QUEUED", "RUNNING"):
            break

        time.sleep(20)

    if status.status != "SUCCESS":
        raise RuntimeError(f"Batch ended with {status.status}")

    # --------------------------------------------------------------
    # E. By now, collect all PyMuPDF jobs too.
    # --------------------------------------------------------------

    for future in native_futures:
        filename, pages = future.result()
        print(f"native: {filename}: {pages} pages")

    pool.shutdown()

    # --------------------------------------------------------------
    # F. Download Mistral's JSONL batch result
    # --------------------------------------------------------------

    result_stream = client.files.download(file_id=status.output_file)

    result_bytes = result_stream.read()

    raw_batch_path = output_dir / "_mistral_batch_results.jsonl"

    raw_batch_path.write_bytes(result_bytes)

    # --------------------------------------------------------------
    # G. Split batch result back into per-PDF files
    # --------------------------------------------------------------

    for line in result_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        custom_id = row["custom_id"]

        response = row.get("response", {})
        body = response.get("body")

        if body is None:
            print(f"FAILED MISTRAL: {manifest[custom_id]['relative_path']}")
            continue

        rel = Path(manifest[custom_id]["relative_path"])

        base = output_dir / rel.parent / rel.stem
        base.parent.mkdir(parents=True, exist_ok=True)

        # Preserve raw document-level OCR response.
        (base.with_suffix(".ocr.json")).write_text(
            json.dumps(
                body,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # And produce the useful textual form.
        pages = sorted(
            body["pages"],
            key=lambda p: p["index"],
        )

        chunks = []

        for page in pages:
            page_no = page["index"] + 1

            chunks.append(
                f"===== PAGE {page_no} =====\n\n{page.get('markdown', '').strip()}"
            )

        (base.with_suffix(".ocr.md")).write_text(
            "\n\n".join(chunks) + "\n",
            encoding="utf-8",
        )

    # --------------------------------------------------------------
    # H. Delete confidential uploaded originals
    # --------------------------------------------------------------

    for file_id in remote_file_ids:
        try:
            client.files.delete(file_id=file_id)
        except Exception as exc:
            print(f"WARNING: couldn't delete {file_id}: {exc}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)

    args = parser.parse_args()

    run(args.input_dir, args.output_dir)
