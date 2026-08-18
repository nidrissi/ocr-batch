# ocr-batch

Batch OCR a directory tree of PDFs two ways:

- **locally**, with PyMuPDF, extracting whatever text layer each PDF already has
  (fast, free, exact for born-digital documents);
- **remotely**, with the Mistral asynchronous batch OCR API (discounted, handles
  scans and handwriting).

Both runs use the same page separators, so the two renderings of a document can
be diffed line by line.

## Install

Requires Python 3.14+.

```console
$ uv sync
$ export MISTRAL_API_KEY=...        # or pass --api-key
```

## Usage

```console
$ ocr-batch run ./applications ./ocr-output
```

`run` is the whole pipeline: it starts local extraction, uploads the PDFs,
creates the batch job(s), waits for them, downloads and splits the results, and
deletes the uploaded originals. The individual stages are also available, and
submitted jobs are resumable because their ids and uploaded file ids are kept in
`<output_dir>/_state.json` as they arrive:

| Command | What it does |
| --- | --- |
| `ocr-batch submit IN OUT` | Local extraction, uploads, batch job creation. Records job ids and uploaded file ids. |
| `ocr-batch status OUT` | One-shot progress report for a submitted run. |
| `ocr-batch fetch OUT` | Waits (`--no-wait` to fail fast), downloads results, writes per-PDF outputs, cleans up. Idempotent. |
| `ocr-batch cleanup OUT` | Deletes the uploaded originals from Mistral. Safe to re-run. |
| `ocr-batch run IN OUT` | `submit` + `fetch`. |

If a `run` is interrupted, the job keeps going on Mistral's side; pick it back up
with `ocr-batch fetch OUT`. Recorded live jobs are not submitted twice: `submit`
refuses to start while a job from a previous run is still live, and skips
documents whose complete outputs already exist (`--force` overrides both).

### Output layout

For `applications/nested/smith.cv.final.pdf`, under `OUT/`:

```
nested/smith.cv.final.native.txt    local PyMuPDF text, page-separated
nested/smith.cv.final.ocr.md        Mistral OCR markdown, page-separated
nested/smith.cv.final.ocr.json      raw per-document OCR response
_manifest.json                      custom_id -> relative path, hash, page count
_state.json                         resumable run state (job ids, upload ids)
_mistral_batch_results.jsonl        raw batch output as downloaded
_mistral_batch_errors.jsonl         per-request errors, when the batch reports any
```

Every suffix is appended to the full name, so `smith.cv.final.pdf` and
`smith.pdf` never collide. Sources that *would* collide (`a.pdf` and `a.PDF`)
are refused before anything is uploaded.

### Useful flags

```
--no-ocr / --no-native        run only one of the two extraction paths
--force                       redo work whose outputs already exist
--jobs N                      local extraction processes (default: one per CPU)
--upload-workers N            concurrent uploads (default: 8)
--batch-size N                requests per batch job (default: 500)
--timeout-hours N             batch job timeout (default: 24)
--no-include-blocks           smaller responses: no per-block bounding boxes
--confidence-granularity ...  none | page | word | block (default: block)
--keep-remote                 keep the uploaded originals on Mistral
--cancel-on-interrupt         cancel the job on Ctrl-C instead of leaving it running
```

Exit codes: `0` success, `1` error, `3` finished with per-document failures,
`130` interrupted.

## Data handling

The PDFs are uploaded to Mistral (`purpose: ocr`, `visibility: user`) and made
readable to the batch job through a signed URL valid for up to 168 hours. Once a
job is terminal, the CLI attempts deletion even if fetching or splitting its
results fails; `ocr-batch cleanup` re-runs any deletion that failed.
As a backstop for a crash between an upload and its id reaching disk, uploads
also carry a server-side expiry (job timeout + 24h by default).

Mistral Annotations are deliberately not used: no `document_annotation_format`
and no `bbox_annotation_format` are ever sent.

## Development

```console
$ uv run pytest
$ uv run ruff check . && uv run ruff format --check .
$ uv run pyright
```

The test suite drives the whole pipeline through a fake client and never touches
the network.
