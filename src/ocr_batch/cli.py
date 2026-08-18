"""Command line interface.

Five subcommands over one durable state file:

    submit   upload + create the batch job(s), and run local extraction
    status   one-shot progress report
    fetch    download and split results (idempotent, resumable)
    cleanup  delete the uploaded originals from Mistral
    run      submit -> wait -> fetch -> cleanup
"""

import argparse
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._version import __version__
from .errors import CollisionError, ConfigError, OcrBatchError, StateError
from .native import NativePool
from .paths import file_sha256, find_collisions, find_pdfs, make_id, output_paths
from .remote import (
    MistralClient,
    OcrOptions,
    Progress,
    aggregate,
    build_request,
    cancel_jobs,
    delete_files,
    download_file,
    refresh_jobs,
    submit_jobs,
    upload_documents,
    wait_for_jobs,
)
from .results import SplitSummary, split_results
from .state import DocumentState, JobState, RemoteFile, RunState

log = logging.getLogger("ocr_batch")

DEFAULT_MODEL = "mistral-ocr-4-0"
DEFAULT_BATCH_SIZE = 500
DEFAULT_UPLOAD_WORKERS = 8
DEFAULT_TIMEOUT_HOURS = 24
DEFAULT_URL_EXPIRY_HOURS = 168

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PARTIAL = 3
EXIT_INTERRUPTED = 130


@dataclass(slots=True)
class SubmitOptions:
    model: str = DEFAULT_MODEL
    native: bool = True
    ocr: bool = True
    force: bool = False
    jobs: int | None = None
    upload_workers: int = DEFAULT_UPLOAD_WORKERS
    batch_size: int = DEFAULT_BATCH_SIZE
    timeout_hours: int = DEFAULT_TIMEOUT_HOURS
    url_expiry_hours: int = DEFAULT_URL_EXPIRY_HOURS
    upload_expiry_hours: int | None = None
    label: str = "ocr-batch"
    ocr_options: OcrOptions = field(default_factory=OcrOptions)

    @property
    def effective_upload_expiry_hours(self) -> int:
        """Server-side expiry for uploads, comfortably outliving the job itself."""
        if self.upload_expiry_hours is not None:
            return self.upload_expiry_hours

        return min(168, self.timeout_hours + 24)


# ----------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------


def resolve_api_key(api_key: str | None = None) -> str:
    """Resolve the API key, failing with a message rather than a KeyError."""
    key = api_key or os.environ.get("MISTRAL_API_KEY")

    if not key:
        raise ConfigError("MISTRAL_API_KEY is not set; export it or pass --api-key")

    return key


@contextmanager
def make_client(api_key: str | None = None) -> Iterator[MistralClient]:
    from mistralai.client import Mistral

    with Mistral(api_key=resolve_api_key(api_key)) as client:
        yield client


# ----------------------------------------------------------------------
# submit
# ----------------------------------------------------------------------


def _prepare_state(input_dir: Path, output_dir: Path, opts: SubmitOptions) -> RunState:
    if not input_dir.is_dir():
        raise ConfigError(f"input directory does not exist: {input_dir}")

    pdfs = find_pdfs(input_dir)

    if not pdfs:
        raise ConfigError(f"no PDFs found under {input_dir}")

    rels = [pdf.relative_to(input_dir) for pdf in pdfs]
    exact, folded = find_collisions(rels)

    if exact:
        first, second = exact[0]
        raise CollisionError(
            f"{first} and {second} would write to the same output files; rename one"
        )

    for first, second in folded:
        log.warning(
            "%s and %s differ only by case and collide on case-insensitive filesystems",
            first,
            second,
        )

    previous = RunState.load_if_exists(output_dir)

    if previous is not None and previous.active_jobs() and not opts.force:
        active = ", ".join(job.job_id for job in previous.active_jobs())
        raise StateError(
            f"{output_dir} already has running batch job(s): {active}. "
            "Use `ocr-batch status` / `ocr-batch fetch`, or --force to start over."
        )

    state = RunState.create(output_dir=output_dir, input_dir=input_dir, model=opts.model)

    # Carry over uploads the previous run never managed to delete, so cleanup
    # still knows about them.
    if previous is not None:
        state.remote_files.extend(previous.pending_remote_files())

    log.info("%d PDFs under %s", len(pdfs), input_dir)

    for pdf, rel in zip(pdfs, rels, strict=True):
        custom_id = make_id(rel)
        document = DocumentState(
            custom_id=custom_id,
            relative_path=rel.as_posix(),
            sha256=file_sha256(pdf),
        )

        old = previous.documents.get(custom_id) if previous is not None else None

        if old is not None and old.sha256 == document.sha256:
            document.native_pages = old.native_pages
            document.native_error = old.native_error
            document.ocr_written = old.ocr_written

        state.documents[custom_id] = document

    state.save()
    state.write_manifest()

    return state


def _abort_uploads(client: MistralClient, state: RunState) -> None:
    """Delete uploads that never made it into a batch job.

    Files already attached to a created job stay: that job is still running and
    still needs them.
    """
    orphans = [remote for remote in state.pending_remote_files() if remote.job_id is None]

    if not orphans:
        return

    log.warning("deleting %d uploaded file(s) not attached to a job", len(orphans))
    failures = delete_files(client, [remote.file_id for remote in orphans])

    for remote in orphans:
        remote.deleted = remote.file_id not in failures

    state.save()


def _submit_ocr(
    client: MistralClient, state: RunState, targets: Sequence[tuple[str, Path]], opts: SubmitOptions
) -> None:
    uploaded_since_save = 0

    def on_upload(upload: Any) -> None:
        nonlocal uploaded_since_save

        state.remote_files.append(RemoteFile(file_id=upload.file_id, custom_id=upload.custom_id))
        uploaded_since_save += 1

        # Persist in small batches: the window between an upload and its id
        # reaching disk is covered by the server-side expiry.
        if uploaded_since_save >= 10:
            state.save()
            uploaded_since_save = 0

    def on_orphan(file_id: str) -> None:
        state.remote_files.append(RemoteFile(file_id=file_id))
        state.save()

    try:
        uploads = upload_documents(
            client,
            targets,
            workers=opts.upload_workers,
            url_expiry_hours=opts.url_expiry_hours,
            upload_expiry_hours=opts.effective_upload_expiry_hours,
            on_upload=on_upload,
            on_orphan=on_orphan,
        )
    finally:
        state.save()

    by_custom_id = {remote.custom_id: remote for remote in state.remote_files}

    def on_job(job_id: str, custom_ids: list[str]) -> None:
        state.jobs.append(JobState(job_id=job_id, custom_ids=custom_ids))

        for custom_id in custom_ids:
            if (remote := by_custom_id.get(custom_id)) is not None:
                remote.job_id = job_id

        state.save()

    requests = [
        build_request(upload.custom_id, upload.signed_url, opts.ocr_options)
        for upload in sorted(uploads, key=lambda upload: upload.custom_id)
    ]

    submit_jobs(
        client,
        requests,
        model=opts.model,
        timeout_hours=opts.timeout_hours,
        batch_size=opts.batch_size,
        metadata={"purpose": opts.label},
        on_job=on_job,
    )


def do_submit(
    input_dir: Path, output_dir: Path, opts: SubmitOptions, api_key: str | None = None
) -> RunState:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if opts.ocr:
        # Fail before hashing and uploading anything if we have no credentials.
        resolve_api_key(api_key)

    output_dir.mkdir(parents=True, exist_ok=True)

    state = _prepare_state(input_dir, output_dir, opts)

    native_targets: list[tuple[str, Path, Path]] = []
    ocr_targets: list[tuple[str, Path]] = []

    for custom_id, document in state.documents.items():
        rel = Path(document.relative_path)
        source = input_dir / rel
        paths = output_paths(output_dir, rel)

        if opts.native and (opts.force or not paths.native.exists()):
            native_targets.append((custom_id, source, paths.native))

        if opts.ocr and (opts.force or not paths.ocr_md.exists()):
            ocr_targets.append((custom_id, source))

    if not native_targets and not ocr_targets:
        log.info("nothing to do: every output already exists (use --force to redo)")
        return state

    # Local extraction starts first and runs while the uploads are in flight.
    with NativePool(opts.jobs) as pool:
        for custom_id, source, destination in native_targets:
            pool.submit(custom_id, source, destination)

        if ocr_targets:
            log.info("uploading %d PDFs", len(ocr_targets))

            with make_client(api_key) as client:
                try:
                    _submit_ocr(client, state, ocr_targets, opts)
                except BaseException:
                    _abort_uploads(client, state)
                    raise

        if native_targets:
            log.info("extracting text locally from %d PDFs", len(native_targets))

            for result in pool.results():
                document = state.documents.get(result.custom_id)

                if document is None:
                    continue

                if result.error:
                    document.native_error = result.error
                    log.error("native extraction failed: %s: %s", result.source, result.error)
                else:
                    document.native_pages = result.pages
                    document.native_error = None
                    log.debug("native: %s: %d pages", result.source, result.pages)

    state.save()
    state.write_manifest()

    for job in state.jobs:
        log.info("batch job %s submitted (%d requests)", job.job_id, len(job.custom_ids))

    if state.jobs:
        log.info("resume with: ocr-batch fetch %s", output_dir)

    return state


# ----------------------------------------------------------------------
# fetch / status / cleanup
# ----------------------------------------------------------------------


def native_failures(state: RunState) -> int:
    return sum(1 for document in state.documents.values() if document.native_error)


def _record_jobs(state: RunState, jobs: dict[str, Any]) -> None:
    for job_id, remote_job in jobs.items():
        job = state.job(job_id)

        if job is None:
            continue

        job.status = str(remote_job.status)
        job.output_file = getattr(remote_job, "output_file", None)
        job.error_file = getattr(remote_job, "error_file", None)

    state.save()


def cleanup_remote(client: MistralClient, state: RunState, *, force: bool = False) -> None:
    """Delete uploaded originals once no job needs them any more."""
    pending = state.pending_remote_files()

    if not pending:
        return

    if not force and state.active_jobs():
        log.warning(
            "%d uploaded file(s) kept: job(s) %s are still running",
            len(pending),
            ", ".join(job.job_id for job in state.active_jobs()),
        )
        return

    failures = delete_files(client, [remote.file_id for remote in pending])

    for remote in pending:
        remote.deleted = remote.file_id not in failures

    state.save()

    if failures:
        for file_id, error in failures.items():
            log.warning("could not delete %s: %s", file_id, error)

        log.warning("re-run `ocr-batch cleanup %s` to retry", state.output_dir)
    else:
        log.info("deleted %d uploaded file(s) from Mistral", len(pending))


def do_fetch(
    output_dir: Path,
    *,
    wait: bool = True,
    force: bool = False,
    keep_remote: bool = False,
    cancel_on_interrupt: bool = False,
    api_key: str | None = None,
) -> int:
    output_dir = output_dir.resolve()
    state = RunState.load(output_dir)

    if not state.jobs:
        raise StateError(f"no batch jobs recorded in {output_dir}")

    job_ids = [job.job_id for job in state.jobs]

    with make_client(api_key) as client:

        def on_update(jobs: dict[str, Any], progress: Progress) -> None:
            _record_jobs(state, jobs)
            log.info("mistral: %s", _describe(jobs, progress))

        try:
            if wait:
                jobs = wait_for_jobs(client, job_ids, on_update=on_update)
            else:
                jobs = refresh_jobs(client, job_ids)
                on_update(jobs, aggregate(jobs))
        except KeyboardInterrupt:
            if cancel_on_interrupt:
                log.warning("cancelling %d batch job(s)", len(job_ids))
                cancel_jobs(client, job_ids)
            else:
                log.warning(
                    "interrupted; job(s) still running -- resume with: ocr-batch fetch %s",
                    output_dir,
                )

            return EXIT_INTERRUPTED

        _record_jobs(state, jobs)

        still_running = state.active_jobs()

        if still_running:
            log.error(
                "job(s) not finished: %s",
                ", ".join(f"{job.job_id} [{job.status}]" for job in still_running),
            )
            return EXIT_ERROR

        summary = SplitSummary()

        for job in state.jobs:
            if job.error_file:
                path = download_file(client, job.error_file, state.errors_path(job.job_id))
                log.warning("per-request errors written to %s", path)

            if not job.output_file:
                log.error("job %s [%s] produced no output file", job.job_id, job.status)
                continue

            results = download_file(client, job.output_file, state.results_path(job.job_id))
            part = split_results(results, state, force=force)

            summary.written += part.written
            summary.failed += part.failed
            summary.skipped += part.skipped
            summary.unknown += part.unknown
            summary.malformed += part.malformed

            job.fetched = True

        state.save()
        state.write_manifest()

        log.info("OCR results: %s", summary)

        if not keep_remote:
            cleanup_remote(client, state)

    failed_jobs = [job for job in state.jobs if job.status != "SUCCESS"]

    if failed_jobs:
        log.error(
            "job(s) ended badly: %s",
            ", ".join(f"{job.job_id} [{job.status}]" for job in failed_jobs),
        )
        return EXIT_PARTIAL

    problems = summary.failed + summary.unknown + summary.malformed
    native_failures = sum(1 for document in state.documents.values() if document.native_error)

    if problems or native_failures:
        if native_failures:
            log.error("%d document(s) failed local extraction", native_failures)

        return EXIT_PARTIAL

    return EXIT_OK


def _describe(jobs: dict[str, Any], progress: Progress) -> str:
    statuses = ", ".join(f"{job_id[:8]}={job.status}" for job_id, job in jobs.items())

    return f"{statuses}: {progress}"


def do_status(output_dir: Path, api_key: str | None = None) -> int:
    output_dir = output_dir.resolve()
    state = RunState.load(output_dir)

    documents = state.documents.values()

    print(f"input:     {state.input_dir}")
    print(f"model:     {state.model}")
    print(f"created:   {state.created_at}")
    print(f"documents: {len(state.documents)}")
    print(
        f"  native:  {sum(1 for d in documents if d.native_pages is not None)} extracted, "
        f"{sum(1 for d in documents if d.native_error)} failed"
    )
    print(
        f"  ocr:     {sum(1 for d in documents if d.ocr_written)} written, "
        f"{sum(1 for d in documents if d.ocr_error)} failed"
    )
    print(f"uploads:   {len(state.pending_remote_files())} still on Mistral")

    if not state.jobs:
        print("jobs:      none")
        return EXIT_OK

    with make_client(api_key) as client:
        jobs = refresh_jobs(client, [job.job_id for job in state.jobs])
        _record_jobs(state, jobs)

        for job_id, remote_job in jobs.items():
            print(
                f"job {job_id}: {remote_job.status} "
                f"{remote_job.succeeded_requests}/{remote_job.total_requests} succeeded, "
                f"{remote_job.failed_requests} failed"
            )

    return EXIT_OK


def do_cleanup(output_dir: Path, *, force: bool = False, api_key: str | None = None) -> int:
    output_dir = output_dir.resolve()
    state = RunState.load(output_dir)

    if not state.pending_remote_files():
        log.info("no uploaded files left to delete")
        return EXIT_OK

    with make_client(api_key) as client:
        if state.jobs and not force:
            _record_jobs(state, refresh_jobs(client, [job.job_id for job in state.jobs]))

        cleanup_remote(client, state, force=force)

    return EXIT_OK


def _submit_exit_code(state: RunState) -> int:
    """Exit code for a run that produced no batch job (e.g. --no-ocr)."""
    failed = native_failures(state)

    if failed:
        log.error("%d document(s) failed local extraction", failed)
        return EXIT_PARTIAL

    return EXIT_OK


def do_run(
    input_dir: Path,
    output_dir: Path,
    opts: SubmitOptions,
    *,
    keep_remote: bool = False,
    cancel_on_interrupt: bool = False,
    api_key: str | None = None,
) -> int:
    state = do_submit(input_dir, output_dir, opts, api_key=api_key)

    if not state.jobs:
        return _submit_exit_code(state)

    return do_fetch(
        output_dir,
        wait=True,
        force=opts.force,
        keep_remote=keep_remote,
        cancel_on_interrupt=cancel_on_interrupt,
        api_key=api_key,
    )


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-key", help="Mistral API key (default: $MISTRAL_API_KEY)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")


def _add_submit_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OCR model (default: %(default)s)")
    parser.add_argument(
        "--no-native", dest="native", action="store_false", help="skip local PyMuPDF extraction"
    )
    parser.add_argument(
        "--no-ocr", dest="ocr", action="store_false", help="skip the Mistral batch OCR entirely"
    )
    parser.add_argument(
        "--force", action="store_true", help="redo work whose output files already exist"
    )
    parser.add_argument(
        "--jobs", type=int, default=None, help="local extraction processes (default: one per CPU)"
    )
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=DEFAULT_UPLOAD_WORKERS,
        help="concurrent uploads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="requests per batch job (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout-hours",
        type=int,
        default=DEFAULT_TIMEOUT_HOURS,
        help="batch job timeout (default: %(default)s)",
    )
    parser.add_argument(
        "--url-expiry-hours",
        type=int,
        default=DEFAULT_URL_EXPIRY_HOURS,
        help="signed URL lifetime, max 168 (default: %(default)s)",
    )
    parser.add_argument(
        "--upload-expiry-hours",
        type=int,
        default=None,
        help="server-side expiry of uploads (default: timeout + 24h)",
    )
    parser.add_argument("--label", default="ocr-batch", help="job metadata purpose label")
    parser.add_argument(
        "--no-include-blocks",
        dest="include_blocks",
        action="store_false",
        help="omit per-block bounding boxes from the OCR response",
    )
    parser.add_argument(
        "--confidence-granularity",
        default="block",
        choices=("none", "page", "word", "block"),
        help="confidence score granularity (default: %(default)s)",
    )
    parser.add_argument("--table-format", default="markdown", choices=("markdown", "html"))
    parser.add_argument(
        "--keep-remote", action="store_true", help="do not delete the uploaded originals afterwards"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-batch",
        description="Batch OCR a tree of PDFs with PyMuPDF and the Mistral batch API.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="submit, wait, fetch and clean up")
    _add_submit_options(run)
    run.add_argument(
        "--cancel-on-interrupt",
        action="store_true",
        help="cancel the batch job(s) on Ctrl-C instead of leaving them running",
    )
    _add_common(run)

    submit = subparsers.add_parser("submit", help="upload and create the batch job(s)")
    _add_submit_options(submit)
    _add_common(submit)

    status = subparsers.add_parser("status", help="report progress of a submitted run")
    status.add_argument("output_dir", type=Path)
    _add_common(status)

    fetch = subparsers.add_parser("fetch", help="download and split results")
    fetch.add_argument("output_dir", type=Path)
    fetch.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="fail instead of waiting if the job is still running",
    )
    fetch.add_argument("--force", action="store_true", help="rewrite outputs that already exist")
    fetch.add_argument(
        "--keep-remote", action="store_true", help="do not delete the uploaded originals afterwards"
    )
    fetch.add_argument(
        "--cancel-on-interrupt",
        action="store_true",
        help="cancel the batch job(s) on Ctrl-C instead of leaving them running",
    )
    _add_common(fetch)

    cleanup = subparsers.add_parser("cleanup", help="delete uploaded originals from Mistral")
    cleanup.add_argument("output_dir", type=Path)
    cleanup.add_argument(
        "--force", action="store_true", help="delete even while a job is still running"
    )
    _add_common(cleanup)

    return parser


def _submit_options(args: argparse.Namespace) -> SubmitOptions:
    granularity = None if args.confidence_granularity == "none" else args.confidence_granularity

    return SubmitOptions(
        model=args.model,
        native=args.native,
        ocr=args.ocr,
        force=args.force,
        jobs=args.jobs,
        upload_workers=args.upload_workers,
        batch_size=args.batch_size,
        timeout_hours=args.timeout_hours,
        url_expiry_hours=args.url_expiry_hours,
        upload_expiry_hours=args.upload_expiry_hours,
        label=args.label,
        ocr_options=OcrOptions(
            include_blocks=args.include_blocks,
            confidence_granularity=granularity,
            table_format=args.table_format,
        ),
    )


def _configure_logging(args: argparse.Namespace) -> None:
    level = logging.INFO

    if getattr(args, "verbose", False):
        level = logging.DEBUG
    elif getattr(args, "quiet", False):
        level = logging.WARNING

    logging.basicConfig(level=level, stream=sys.stderr, format="%(levelname)s: %(message)s")
    logging.getLogger("ocr_batch").setLevel(level)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args)

    try:
        match args.command:
            case "run":
                return do_run(
                    args.input_dir,
                    args.output_dir,
                    _submit_options(args),
                    keep_remote=args.keep_remote,
                    cancel_on_interrupt=args.cancel_on_interrupt,
                    api_key=args.api_key,
                )
            case "submit":
                return _submit_exit_code(
                    do_submit(
                        args.input_dir,
                        args.output_dir,
                        _submit_options(args),
                        api_key=args.api_key,
                    )
                )
            case "status":
                return do_status(args.output_dir, api_key=args.api_key)
            case "fetch":
                return do_fetch(
                    args.output_dir,
                    wait=args.wait,
                    force=args.force,
                    keep_remote=args.keep_remote,
                    cancel_on_interrupt=args.cancel_on_interrupt,
                    api_key=args.api_key,
                )
            case "cleanup":
                return do_cleanup(args.output_dir, force=args.force, api_key=args.api_key)
            case _:  # pragma: no cover -- argparse rejects anything else
                raise ConfigError(f"unknown command {args.command!r}")
    except OcrBatchError as exc:
        log.error("%s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:
        log.warning("interrupted")
        return EXIT_INTERRUPTED
