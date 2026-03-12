from typing import Protocol

from .document import DocumentChunk
from .types import Anchor


class MarkdownProvider(Protocol):
    """Protocol for generating markdown from a document chunk."""

    async def generate_markdown(self, chunk: DocumentChunk) -> str: ...


class AnchorProvider(Protocol):
    """Protocol for generating anchors from a document chunk."""

    async def generate_anchors(self, chunk: DocumentChunk) -> list[Anchor]: ...
