"""Run one generated page through the live OpenAI Batch OCR workflow.

    uv run python scripts/phase1_openai.py --yes

This sends a synthetic PNG to OpenAI and incurs an API charge.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

SMOKE_TEXT = "OPENAI OCR SMOKE PAGE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="submit one paid OpenAI page request")
    parser.add_argument("--work", type=Path, help="put the smoke run under this existing directory")
    args = parser.parse_args()
    if not args.yes:
        parser.error("pass --yes to submit a paid OpenAI Batch job")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is not set")

    work = Path(tempfile.mkdtemp(prefix="ocr-batch-openai-smoke-", dir=args.work))
    source = work / "input"
    out = work / "output"
    source.mkdir()
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), SMOKE_TEXT, fontsize=16)
        pdf.save(source / "smoke.pdf")

    print(f"Smoke artifacts: {work}", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "ocr_batch", "run", str(source), str(out), "--provider", "openai"],
        check=False,
    )
    if result.returncode:
        return result.returncode

    try:
        state = json.loads((out / "_state.json").read_text(encoding="utf-8"))
        (job,) = state["jobs"]
        envelope = json.loads((out / "smoke.ocr.json").read_text(encoding="utf-8"))
        (page,) = envelope["pages"]
        markdown = (out / "smoke.ocr.md").read_text(encoding="utf-8")
        checks = {
            "completed Batch job": job["status"] == "completed" and job["fetched"],
            "OpenAI page response": envelope["provider"] == "openai" and not page["blank"],
            "returned usage": page["response"].get("usage") is not None,
            "transcribed page": "===== PAGE 1 =====" in markdown
            and SMOKE_TEXT in re.sub(r"\s+", " ", markdown).upper(),
            "cached raw output": (out / f"_openai_batch_results-{job['job_id']}.jsonl").is_file(),
            "remote files deleted": bool(state["remote_files"])
            and all(remote["deleted"] for remote in state["remote_files"]),
        }
    except (OSError, KeyError, ValueError) as exc:
        print(f"Invalid smoke outputs: {exc}", file=sys.stderr)
        return 1

    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {label}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
