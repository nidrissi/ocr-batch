"""Shared page-rendering helpers.

The native (PyMuPDF) and OCR (Mistral) outputs use the same page separator on
purpose, so the two renderings of a document can be diffed line by line.
"""

PAGE_HEADER = "===== PAGE {number} ====="


def page_chunk(number: int, body: str) -> str:
    """Render one page as a separator line followed by its text."""
    return f"{PAGE_HEADER.format(number=number)}\n\n{body}"


def join_pages(chunks: list[str]) -> str:
    """Join rendered pages into a document, always newline-terminated."""
    return "\n\n".join(chunks) + "\n"
