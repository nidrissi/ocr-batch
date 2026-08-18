# Changelog

## 0.1.0b1

First beta. The pipeline from the initial draft is unchanged in spirit -- local
PyMuPDF extraction running alongside one discounted Mistral batch OCR job -- but
it is now resumable, restartable and packaged.

### Added

- `run`, `submit`, `status`, `fetch` and `cleanup` subcommands over a durable
  `_state.json`, updated atomically as remote ids arrive. An interrupted run can
  resume a recorded submitted (and paid for) batch job.
- Uploaded originals are deleted on handled terminal-job exit paths, not only on
  success, and carry a server-side expiry as a backstop. `cleanup` retries failed
  deletions and refuses to delete files a running job still needs.
- Partial results are salvaged: a `FAILED` or `TIMEOUT_EXCEEDED` job still has
  its `output_file` split into per-document outputs, and its `error_file` is
  downloaded to `_mistral_batch_errors.jsonl`.
- Concurrent uploads, chunking into several batch jobs above `--batch-size`
  requests, and polling with exponential backoff that logs only on change.
- Documents whose outputs already exist are skipped unless `--force`.
- CLI flags for model, OCR options, concurrency and timeouts; `--version`;
  `python -m ocr_batch`; logging with `-v`/`-q`; meaningful exit codes.
- Test suite, ruff and pyright configuration, this changelog and a README.

### Fixed

- Output names are built by appending to the full stem instead of
  `Path.with_suffix`, which silently turned `smith.cv.final.pdf` into
  `smith.cv.ocr.md` and let `smith.cv.pdf` overwrite it. Colliding sources are
  now detected before anything is uploaded.
- `pymupdf` and `mistralai` are declared as dependencies; the installed package
  used to fail at import.
- PDF discovery matches the suffix case-insensitively, so `.PDF` files are no
  longer skipped.
- A corrupt or password-protected PDF is reported and skipped instead of killing
  the run after the batch has already completed.
- Results for an unknown `custom_id`, malformed JSONL lines and missing response
  bodies are reported per row instead of aborting the split.
- The batch JSONL result is streamed to disk and parsed line by line rather than
  held in memory twice.
- A missing `MISTRAL_API_KEY` produces a one-line message instead of a
  `KeyError` traceback.
- `custom_id`s hash the POSIX form of the relative path, so they match across
  platforms.
