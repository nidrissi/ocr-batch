# ocr-batch

Batch OCR a directory tree of PDFs with local and remote extraction.

- **locally**, with PyMuPDF, extracting whatever text layer each PDF already has
  (fast, free, exact for born-digital documents);
- **remotely**, with Mistral's asynchronous OCR API or OpenAI's asynchronous
  Responses Batch API.

Both runs use the same page separators, so the two renderings of a document can
be diffed line by line.

## Install

Requires Python 3.14+.

With [uv](https://docs.astral.sh/uv/), no checkout needed:

```bash
uvx ocr-batch run ./applications ./ocr-output
```

Or from a clone:

```bash
git clone https://github.com/nidrissi/ocr-batch
cd ocr-batch
uv sync
uv run ocr-batch run ./applications ./ocr-output
```

The installation includes both provider SDKs. Set the API key for your chosen
provider or pass it with `--api-key`.

```bash
export MISTRAL_API_KEY=...
export OPENAI_API_KEY=...
```

## Usage

```bash
ocr-batch run ./applications ./ocr-output
ocr-batch run ./applications ./openai-output --provider openai
ocr-batch run ./applications ./native-output --no-ocr
```

`run` starts local extraction, prepares and submits the selected provider's
batch requests, waits, writes the per-document outputs, and deletes eligible
remote files. Mistral is the default. Local-only runs with `--no-ocr` need no
API key. Each output directory is bound to a single remote provider, including
under `--force`. Use separate output directories to compare providers.

| Command | What it does |
| --- | --- |
| `ocr-batch submit IN OUT` | Local extraction, uploads, batch job creation. Records job ids and uploaded file ids. |
| `ocr-batch status OUT` | One-shot progress report for a submitted run. |
| `ocr-batch fetch OUT` | Waits (`--no-wait` to fail fast), downloads results, writes per-PDF outputs, cleans up. Idempotent. |
| `ocr-batch cleanup OUT` | Deletes eligible remote files and retries failed deletions. Safe to re-run. |
| `ocr-batch run IN OUT` | `submit` + `fetch`. |

Commands after submission read the provider from `OUT/_state.json`. This
sequence submits an OpenAI batch and resumes it later.

```bash
ocr-batch submit ./applications ./openai-output --provider openai --no-native
ocr-batch status ./openai-output
ocr-batch fetch ./openai-output
ocr-batch cleanup ./openai-output
```

By default, Ctrl-C leaves jobs running. `run` and `fetch` accept
`--cancel-on-interrupt` to request cancellation during the wait. OpenAI's
`cancelling` status remains active until the provider reaches a terminal
status. `fetch` can recover completed results from expired and cancelled
batches. With OpenAI, `fetch --no-wait` also processes available terminal
batches and exits 1 while work is still live.

Use `fetch` before submitting again. It reuses raw downloads, can reconstruct
missing outputs, and never submits paid work. `submit` rejects recorded live
jobs even with `--force`. Finished but unfetched jobs also block submission
unless you explicitly discard them with `--force`. After a partial failure,
an ordinary `submit` resubmits each incomplete document in full, including
any of its pages that succeeded before. It skips completed documents whose
PDF hash, effective OCR settings, and two output files match. `--force`
requests replacements for all documents. Previous outputs and their recorded
provenance survive failed replacement work.
Run state-changing commands one at a time for a given output directory.

Version-1 state loads as Mistral, preserving outstanding jobs and upload IDs.
Existing outputs remain readable. Their old settings are unknown, so the next
remote submission conservatively refreshes those outputs. The migration is
written on the next state save.

### Output layout

For `applications/nested/smith.cv.final.pdf`, under `OUT/`:

```
nested/smith.cv.final.native.txt    local PyMuPDF text, page-separated
nested/smith.cv.final.ocr.md        remote OCR Markdown, page-separated
nested/smith.cv.final.ocr.json      provider-specific document response
_manifest.json                      custom_id -> relative path, hash, page count
_state.json                         provider, provenance, job and file IDs
```

Every suffix is appended to the full name, so `smith.cv.final.pdf` and
`smith.pdf` never collide. Sources that *would* collide (`a.pdf` and `a.PDF`)
are refused before anything is uploaded.

Mistral's `.ocr.json` is the raw OCR response body, including provider fields
such as blocks and confidence scores when requested. Raw batch files use
`_mistral_batch_results.jsonl` and `_mistral_batch_errors.jsonl`, with a
`-<job-id>` suffix for multiple jobs.

OpenAI's `.ocr.json` is an envelope with `provider`, `document_id`, `sha256`,
`configuration` (model, prompt and schema versions, rendering and generation
settings), and ordered `pages`. Each page contains its zero-based `index`,
`job_id`, `custom_id`, API `request_id`, `markdown`, `blank`, and the raw
Responses `response` body, which includes usage when provided. A blank page
has a page separator, empty Markdown, and `blank: true`. Confidence scores
and bounding boxes are specific to Mistral. OpenAI raw output and error files
use `_openai_batch_results-<job-id>.jsonl` and
`_openai_batch_errors-<job-id>.jsonl`. Keep these files and `_state.json` for
recovery. The CLI replaces each output atomically and marks the document
complete after both files are written.

### Useful flags

```
--provider mistral|openai      remote backend (default: mistral)
--model ID                    model override (defaults below)
--no-ocr / --no-native        run only one of the two extraction paths
--force                       redo completed work, discarding unfetched finished jobs
--jobs N                      local extraction processes (default: one per CPU)
--upload-workers N            concurrent Mistral uploads (default: 8)
--batch-size N                max documents (Mistral) or pages (OpenAI), default 500
--timeout-hours N             batch job timeout (default: 24)
--no-include-blocks           smaller responses: no per-block bounding boxes
--confidence-granularity ...  none | page | word | block (default: block)
--keep-remote                 keep remote files after run/fetch
--cancel-on-interrupt         cancel the job on Ctrl-C instead of leaving it running
```

Exit codes are `0` for success, `1` for operational errors or a `--no-wait`
fetch with active jobs, `3` for partial results or document failures, and
`130` for interruption.

### OpenAI defaults and limits

Mistral defaults to `mistral-ocr-4-1`. OpenAI defaults to `gpt-6.1-terra`, with
a 24-hour Batch window, `detail: auto`, low reasoning effort, and a maximum of
16,384 output tokens per page. The transcription prompt and schema are both
version 1. Model overrides must support images, the
Responses Batch endpoint, Structured Outputs, and the requested output cap.
Overrides use their own reasoning defaults. No tools, sampling, or verbosity
settings are requested. [Terra model](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

OpenAI receives one page per request, rendered with PyMuPDF as a 200 DPI RGB
PNG with annotations and no alpha channel. Page counts are read even with
`--no-native`. The CLI applies no resizing. It rejects unreadable, encrypted,
and damaged PDFs and pages over 65,535 pixels on either side or over 30,000
32-by-32 pixel patches. The API can apply its own model-specific resizing and
input limits. Batches are capped at 49,999 requests and 195,000,000 serialized
UTF-8 bytes, including base64 data and newlines. An oversized page request fails
its document. Other documents can proceed. [Batch API](https://developers.openai.com/api/docs/guides/batch) · [Images and vision](https://developers.openai.com/api/docs/guides/images-vision)

For OpenAI, `--timeout-hours` accepts 24 only. The Mistral-only flags
`--upload-workers`, `--url-expiry-hours`, `--upload-expiry-hours`,
`--no-include-blocks`, `--confidence-granularity`, and `--table-format` are
rejected when explicitly passed. OpenAI transcribes visible figure labels
and inserts a figure marker. OCR can still omit text, misread notation, or
change reading order. No automatic retries or provider fallback are performed.

## Data handling

With Mistral, PDFs are uploaded (`purpose: ocr`, `visibility: user`) and made
readable to the batch job through a signed URL valid for up to 168 hours. Once a
job is terminal, the CLI attempts deletion even if fetching or splitting its
results fails; `ocr-batch cleanup` re-runs any deletion that failed.
As a backstop for a crash between an upload and its id reaching disk, uploads
also carry a server-side expiry (job timeout + 24h by default).

Mistral Annotations are deliberately not used: no `document_annotation_format`
and no `bbox_annotation_format` are ever sent.

With OpenAI, the input Batch JSONL contains the rendered pages as inline PNG
data URLs. Inputs are eligible for deletion when their jobs reach a terminal
status and expire seven days after upload as a backstop. Result and error
files expire 30 days after file creation. The CLI deletes those files only
after an atomic, durable local download. A failed download leaves its remote
file available for a later fetch. `--keep-remote` skips the post-fetch
deletion pass. `cleanup` can delete eligible files later, including unattached
uploads left by an interrupted submit. If a batch-create reply is lost, the CLI
reports the input file ID and leaves that potentially attached input to expire.
Check OpenAI's Batch dashboard for the file ID before resubmitting. OpenAI
cleanup protects active inputs and undownloaded results even with `--force`.
The local raw JSONL can contain sensitive transcriptions. `store=false`
applies to Responses storage. Batch
files and other service retention still apply. [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data)

## Compare OCR quality

1. Use the same representative set of application PDFs and mathematical pages
   in both runs. Include scans, dense equations, tables, footnotes, and blank
   pages. Prepare a literal reference transcription for the sampled pages.
2. Time each run from submission through fetch, using fresh output directories
   and the same input files. Record models, settings, elapsed time, and actual
   provider usage. For OpenAI, sum input, output, and total tokens from
   `pages[].response.usage`, including reasoning tokens where reported. Use
   raw Batch rows for usage on incomplete documents. For Mistral, record the
   usage fields the response or provider reports (including pages processed)
   and mark unavailable token counts as unavailable.
3. Assess operational results first by recording exit codes, completed
   documents, and missing or failed pages. Compare the completed outputs
   page by page against the reference. Count omissions, additions, and
   symbol errors, including accents, signs, subscripts, and superscripts.
   Record reading-order errors such as swapped columns or misplaced footnotes.
   Report counts with the number of pages scored for each provider.

## Development

```console
$ uv run pytest
$ uv run ruff check . && uv run ruff format --check .
$ uv run pyright
```

The test suite drives the whole pipeline through a fake client and never touches
the network.

The OpenAI smoke script generates a non-sensitive PDF. It requires `--yes`
and an OpenAI API key to submit a paid batch.

```bash
uv run python scripts/phase1_openai.py --yes
```
