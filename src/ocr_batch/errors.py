"""Exception hierarchy for ocr-batch.

Every error the CLI raises on purpose derives from `OcrBatchError`, so `main`
can turn it into a one-line message and an exit code instead of a traceback.
"""


class OcrBatchError(Exception):
    """Base class for expected, user-facing failures."""


class ConfigError(OcrBatchError):
    """Missing credentials, bad paths, nothing to do."""


class StateError(OcrBatchError):
    """The run state on disk is missing, unreadable, or in the wrong phase."""


class CollisionError(OcrBatchError):
    """Two source PDFs would write to the same output files."""


class RemoteError(OcrBatchError):
    """The Mistral API refused or a batch job ended badly."""


class UploadError(RemoteError):
    """An upload failed after the file was already stored remotely.

    Carries `file_id` when one exists so the caller can record it for cleanup
    instead of leaking a confidential document on Mistral's servers.
    """

    def __init__(self, message: str, *, file_id: str | None = None) -> None:
        super().__init__(message)
        self.file_id = file_id
