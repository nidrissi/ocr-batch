# Changelog

## 0.2.0

### Added

- `run` and `submit` accept `--provider openai` with `OPENAI_API_KEY`. The
  packaged SDK submits 200 DPI PNG pages to Responses Batch using
  `gpt-6.1-terra` by default. Mistral remains the default provider.
- OpenAI fetch assembles page-separated Markdown and a JSON envelope with
  configuration, blank-page markers, and raw Responses bodies with usage.
  Durable Batch output and error files support interrupted fetch and
  reconstruction. Cleanup tracks input and result files independently.

### Changed

- The default Mistral OCR model is now `mistral-ocr-4-1`.
- State version 2 records the provider and successful output provenance
  independently from replacement work. Version-1 Mistral jobs and uploads
  remain resumable. The next OCR submission refreshes legacy outputs with
  unknown settings.
- `--force` can discard completed but unfetched batch jobs. Running jobs still
  must be fetched before another submission.
- The README covers PyPI installation with `uvx`, separate provider output
  directories, resume commands, supported settings, service retention, and a
  manual OCR comparison. `scripts/phase1_openai.py --yes` is an opt-in live
  smoke test.

### Fixed

- Interrupted uploads are drained and recorded so uploaded files can be cleaned
  up. Interrupted downloads preserve the previous cache until replacement is
  complete.
- Mistral fetch can resume from cached result files, process partial jobs, and
  report documents that never reached a job or returned a successful result.
- Malformed result rows and non-success HTTP responses are reported without
  discarding valid rows. Server-reported failed requests are included in the
  final failure count.
- Failed submissions preserve existing OCR state until a replacement job is
  created.

### Dependencies

- Added the OpenAI SDK with a minimum version of 2.54. The Mistral SDK now
  supports 3.x and is locked to 3.7.0. PyMuPDF is locked to 2.1 and the
  transitive `opentelemetry-api` dependency to 1.45.2.
- The build backend now supports `uv_build` 0.13. The development lockfile uses
  Ruff 0.17.5, pytest 9.2.0, and Pyright 1.1.414.

[Full comparison from v0.1.0b2 to v0.2.0](https://github.com/nidrissi/ocr-batch/compare/v0.1.0b2...v0.2.0)

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
