"""Batch OCR a tree of PDFs with PyMuPDF and the Mistral batch API."""

from ._version import __version__
from .cli import main

__all__ = ["__version__", "main"]
