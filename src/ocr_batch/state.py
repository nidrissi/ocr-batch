"""Durable run state.

The whole point of this module is that a crash, a Ctrl-C or a closed laptop lid
must never lose a submitted -- and paid for -- batch job, and must never leave
confidential uploads sitting on Mistral's servers with no record of their ids.
Everything the recovery path needs lives in `_state.json`, written atomically
before it is acted on.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from .errors import StateError

STATE_FILENAME = "_state.json"
MANIFEST_FILENAME = "_manifest.json"
RESULTS_FILENAME = "_mistral_batch_results.jsonl"
ERRORS_FILENAME = "_mistral_batch_errors.jsonl"

STATE_VERSION = 1

TERMINAL_STATUSES = frozenset({"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"})


def write_atomic(path: Path, data: str) -> None:
    """Write `data` to `path` via a temp file in the same directory plus a rename.

    A partially written state file is worse than no state file at all, so the
    rename is the only thing that ever makes new content visible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


@dataclass(slots=True)
class DocumentState:
    custom_id: str
    relative_path: str
    sha256: str
    native_pages: int | None = None
    native_error: str | None = None
    ocr_written: bool = False
    ocr_error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            custom_id=data["custom_id"],
            relative_path=data["relative_path"],
            sha256=data.get("sha256", ""),
            native_pages=data.get("native_pages"),
            native_error=data.get("native_error"),
            ocr_written=bool(data.get("ocr_written", False)),
            ocr_error=data.get("ocr_error"),
        )


@dataclass(slots=True)
class RemoteFile:
    file_id: str
    custom_id: str | None = None
    job_id: str | None = None
    deleted: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            file_id=data["file_id"],
            custom_id=data.get("custom_id"),
            job_id=data.get("job_id"),
            deleted=bool(data.get("deleted", False)),
        )


@dataclass(slots=True)
class JobState:
    job_id: str
    custom_ids: list[str] = field(default_factory=list)
    status: str = "QUEUED"
    output_file: str | None = None
    error_file: str | None = None
    fetched: bool = False

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            job_id=data["job_id"],
            custom_ids=list(data.get("custom_ids", [])),
            status=data.get("status", "QUEUED"),
            output_file=data.get("output_file"),
            error_file=data.get("error_file"),
            fetched=bool(data.get("fetched", False)),
        )


@dataclass(slots=True)
class RunState:
    output_dir: Path
    input_dir: str = ""
    model: str = ""
    created_at: str = ""
    version: int = STATE_VERSION
    documents: dict[str, DocumentState] = field(default_factory=dict)
    jobs: list[JobState] = field(default_factory=list)
    remote_files: list[RemoteFile] = field(default_factory=list)

    # -- construction -------------------------------------------------

    @classmethod
    def create(cls, *, output_dir: Path, input_dir: Path, model: str) -> Self:
        return cls(
            output_dir=output_dir,
            input_dir=str(input_dir),
            model=model,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    @classmethod
    def load(cls, output_dir: Path) -> Self:
        state = cls.load_if_exists(output_dir)

        if state is None:
            raise StateError(f"no {STATE_FILENAME} in {output_dir}; run `ocr-batch submit` first")

        return state

    @classmethod
    def load_if_exists(cls, output_dir: Path) -> Self | None:
        path = output_dir / STATE_FILENAME

        if not path.is_file():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StateError(f"could not read {path}: {exc}") from exc

        if data.get("version") != STATE_VERSION:
            raise StateError(
                f"{path} has state version {data.get('version')!r}, expected {STATE_VERSION}"
            )

        return cls(
            output_dir=output_dir,
            input_dir=data.get("input_dir", ""),
            model=data.get("model", ""),
            created_at=data.get("created_at", ""),
            documents={
                custom_id: DocumentState.from_dict(entry)
                for custom_id, entry in data.get("documents", {}).items()
            },
            jobs=[JobState.from_dict(entry) for entry in data.get("jobs", [])],
            remote_files=[RemoteFile.from_dict(entry) for entry in data.get("remote_files", [])],
        )

    # -- persistence --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "input_dir": self.input_dir,
            "model": self.model,
            "created_at": self.created_at,
            "jobs": [
                {
                    "job_id": job.job_id,
                    "custom_ids": job.custom_ids,
                    "status": job.status,
                    "output_file": job.output_file,
                    "error_file": job.error_file,
                    "fetched": job.fetched,
                }
                for job in self.jobs
            ],
            "remote_files": [
                {
                    "file_id": remote.file_id,
                    "custom_id": remote.custom_id,
                    "job_id": remote.job_id,
                    "deleted": remote.deleted,
                }
                for remote in self.remote_files
            ],
            "documents": {
                custom_id: {
                    "custom_id": doc.custom_id,
                    "relative_path": doc.relative_path,
                    "sha256": doc.sha256,
                    "native_pages": doc.native_pages,
                    "native_error": doc.native_error,
                    "ocr_written": doc.ocr_written,
                    "ocr_error": doc.ocr_error,
                }
                for custom_id, doc in self.documents.items()
            },
        }

    def save(self) -> None:
        write_atomic(
            self.output_dir / STATE_FILENAME,
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
        )

    def write_manifest(self) -> None:
        """Write the human-facing id -> path mapping kept alongside the state."""
        manifest = {
            custom_id: {
                "relative_path": doc.relative_path,
                "sha256": doc.sha256,
                "native_pages": doc.native_pages,
            }
            for custom_id, doc in sorted(self.documents.items())
        }

        write_atomic(
            self.output_dir / MANIFEST_FILENAME,
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )

    # -- queries ------------------------------------------------------

    def job(self, job_id: str) -> JobState | None:
        return next((job for job in self.jobs if job.job_id == job_id), None)

    def active_jobs(self) -> list[JobState]:
        return [job for job in self.jobs if not job.terminal]

    def pending_remote_files(self) -> list[RemoteFile]:
        return [remote for remote in self.remote_files if not remote.deleted]

    def results_path(self, job_id: str) -> Path:
        """Where a job's JSONL output lands.

        A single-job run keeps the plain, predictable name; only multi-job runs
        need the id in the filename.
        """
        if len(self.jobs) <= 1:
            return self.output_dir / RESULTS_FILENAME

        return self.output_dir / f"_mistral_batch_results-{job_id}.jsonl"

    def errors_path(self, job_id: str) -> Path:
        if len(self.jobs) <= 1:
            return self.output_dir / ERRORS_FILENAME

        return self.output_dir / f"_mistral_batch_errors-{job_id}.jsonl"
