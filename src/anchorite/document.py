import dataclasses
import hashlib
import io
import mimetypes
import pathlib
from collections.abc import Iterator
from typing import TypeAlias

import fitz
import fsspec

DocumentInput: TypeAlias = pathlib.Path | str | bytes | io.IOBase


@dataclasses.dataclass
class DocumentChunk:
    """A chunk of a document (e.g., a subset of pages extracted from a PDF)."""

    document_sha256: str
    """SHA256 hash of the original document."""
    start_page: int
    """Start page number of this chunk in the original document."""
    end_page: int
    """End page number (exclusive) of this chunk."""
    data: bytes
    """Raw bytes of the chunk (PDF or image)."""
    mime_type: str
    """MIME type of the chunk data."""


def _split_pdf_bytes(file_bytes: bytes, page_count: int | None = None) -> Iterator[DocumentChunk]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    doc_page_count = len(doc)
    document_sha256 = hashlib.sha256(file_bytes).hexdigest()
    if page_count is None:
        yield DocumentChunk(document_sha256, 0, doc_page_count, file_bytes, "application/pdf")
        return

    for start_page in range(0, doc_page_count, page_count):
        new_doc = fitz.open()
        end_page = min(start_page + page_count, doc_page_count)
        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)
        yield DocumentChunk(document_sha256, start_page, end_page, new_doc.tobytes(), "application/pdf")
        new_doc.close()


def _resolve_input(input_source: DocumentInput, mime_type: str | None) -> tuple[bytes, str | None]:
    """Resolves input source to bytes and mime_type."""
    file_bytes: bytes

    match input_source:
        case str() if "://" in input_source:
            with fsspec.open(input_source, "rb") as f:
                file_bytes = f.read()  # type: ignore[attr-defined]
            if mime_type is None:
                mime_type, _ = mimetypes.guess_type(input_source)
        case str() | pathlib.Path() as path:
            path_obj = pathlib.Path(path)
            file_bytes = path_obj.read_bytes()
            if mime_type is None:
                mime_type, _ = mimetypes.guess_type(path_obj)
        case bytes():
            file_bytes = input_source
        case _ if hasattr(input_source, "read"):
            file_bytes = input_source.read()
        case _:
            raise ValueError(f"Unsupported input source: {input_source}")

    return file_bytes, mime_type


def chunks(
    input_source: DocumentInput,
    *,
    page_count: int | None = None,
    mime_type: str | None = None,
) -> Iterator[DocumentChunk]:
    """Splits a Document into chunks.

    Supports PDF (splits by pages) and Images (single chunk).
    """
    file_bytes, mime_type = _resolve_input(input_source, mime_type)

    # Auto-detect PDF if mime_type is unknown
    if mime_type is None:
        if file_bytes.startswith(b"%PDF"):
            mime_type = "application/pdf"
        elif file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            mime_type = "image/png"
        elif file_bytes.startswith(b"\xff\xd8"):
            mime_type = "image/jpeg"
        elif file_bytes.startswith(b"RIFF") and file_bytes[8:12] == b"WEBP":
            mime_type = "image/webp"

    if mime_type and mime_type.startswith("image/"):
        yield DocumentChunk(hashlib.sha256(file_bytes).hexdigest(), 0, 1, file_bytes, mime_type)
        return

    if mime_type != "application/pdf":
        raise ValueError(f"Unsupported file type: {mime_type}")

    yield from _split_pdf_bytes(file_bytes, page_count)
