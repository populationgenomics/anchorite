"""Protocols for Markdown and anchor generation providers."""

from typing import Protocol

from . import anchors, document


class MarkdownProvider(Protocol):
    """Protocol for generating markdown from a document chunk."""

    async def generate_markdown(self, chunk: document.DocumentChunk) -> str: ...


class AnchorProvider(Protocol):
    """Protocol for generating anchors from a document chunk."""

    async def generate_anchors(self, chunk: document.DocumentChunk) -> list[anchors.Anchor]: ...
