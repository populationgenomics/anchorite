"""Protocols for Markdown and anchor generation providers."""

from typing import Protocol, runtime_checkable

from . import anchors, document


class MarkdownProvider(Protocol):
    """Protocol for generating markdown from a document chunk."""

    async def generate_markdown(self, chunk: document.DocumentChunk) -> str: ...


class AnchorProvider(Protocol):
    """Protocol for generating anchors from a document chunk."""

    async def generate_anchors(self, chunk: document.DocumentChunk) -> list[anchors.Anchor]: ...


@runtime_checkable
class MarkdownAnchorProvider(Protocol):
    """Protocol for generating anchors by aligning assembled Markdown to PDF char data.

    Separates PDF reading (chunked, parallelisable) from alignment (full document):

    * ``process_chunk`` extracts and caches per-character bounding boxes from a
      PDF chunk.  It has no Markdown dependency and can run concurrently with
      ``MarkdownProvider.generate_markdown``.
    * ``finalize`` receives the fully assembled Markdown (after all chunks are
      joined) and performs the alignment, returning one ``Anchor`` per matched
      Markdown segment with page number and bounding boxes already populated.

    Unlike ``AnchorProvider``, a ``MarkdownAnchorProvider`` implementation does
    not require a separate ``align`` step — pagination and span assignment are
    both outputs of ``finalize``.
    """

    async def process_chunk(self, chunk: document.DocumentChunk) -> None:
        """Extract and cache character bbox data from this chunk's PDF pages."""
        ...

    async def finalize(self, markdown: str) -> list[anchors.Anchor]:
        """Align assembled Markdown against accumulated char data.

        Args:
            markdown: The complete assembled Markdown with ``<!--page-->``
                page-break markers.

        Returns:
            One ``Anchor`` per matched Markdown segment, in document order.
        """
        ...
