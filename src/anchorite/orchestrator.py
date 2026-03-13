import asyncio
import dataclasses
from collections.abc import Iterable

from . import document
from .bbox_alignment import align_anchors as align_fn
from .providers import AnchorProvider, MarkdownProvider
from .types import Anchor


@dataclasses.dataclass
class AlignmentResult:
    """The result of aligning anchors to generated markdown."""

    markdown_content: str
    """The generated markdown content."""
    anchor_spans: dict[Anchor, tuple[int, int]]
    """Mapping of anchors to their span ranges in the markdown."""
    coverage_percent: float
    """Percentage of markdown content covered by aligned anchors."""

    def annotate(self) -> str:
        """Annotates the markdown content with anchor spans."""
        # Import here to avoid circular import
        from . import annotate as annotate_fn

        return annotate_fn(self.markdown_content, self.anchor_spans)


async def process_document(
    chunks: Iterable[document.DocumentChunk],
    markdown_provider: MarkdownProvider,
    anchor_provider: AnchorProvider | None = None,
    *,
    alignment_uniqueness_threshold: float = 0.5,
    alignment_min_overlap: float = 0.9,
    renumber: bool = True,
) -> AlignmentResult:
    """Align anchors to Markdown across a set of pre-chunked document pages.

    Generates Markdown and anchors concurrently for all chunks, joins the Markdown
    with ``<!--page-->`` separators, then runs alignment across the full document.

    Args:
        chunks: Pre-chunked document pages, e.g. from ``anchorite.document.chunks()``.
            Markdown and anchors are generated for each chunk independently and in
            parallel, then assembled before alignment.
        markdown_provider: Generates Markdown text for a chunk (e.g. an LLM call).
            Run concurrently with anchor generation when ``anchor_provider`` is set.
        anchor_provider: Generates anchors for a chunk (e.g. an OCR call). If
            ``None``, alignment is skipped and an empty ``AlignmentResult`` is returned.
        alignment_uniqueness_threshold: Passed to ``align``. An anchor is accepted
            only when its best-match score exceeds this fraction of its second-best
            score.
        alignment_min_overlap: Passed to ``align``. Minimum fraction of the anchor's
            normalised length that must be covered.
        renumber: If ``True``, renumber ``<!--table-->`` and ``<!--figure-->``
            markers across chunks before joining them (so numbering is document-wide
            rather than per-chunk).

    Returns:
        ``AlignmentResult`` with the assembled Markdown, the anchor→span mapping,
        and the fraction of Markdown characters covered by aligned anchors.
    """
    chunk_list = list(chunks)

    markdown_tasks = [markdown_provider.generate_markdown(chunk) for chunk in chunk_list]

    if anchor_provider is not None:
        anchor_tasks = [anchor_provider.generate_anchors(chunk) for chunk in chunk_list]
        results = await asyncio.gather(*markdown_tasks, *anchor_tasks)
        markdown_chunks = list(results[: len(chunk_list)])
        all_anchors = results[len(chunk_list) :]
        flat_anchors = [anchor for chunk_anchors in all_anchors for anchor in chunk_anchors]
    else:
        markdown_chunks = list(await asyncio.gather(*markdown_tasks))

    if renumber:
        from .markdown import renumber_markers

        markdown_chunks = renumber_markers(markdown_chunks)

    markdown_content = "\n\n<!--page-->\n\n".join(markdown_chunks)

    if anchor_provider is None:
        return AlignmentResult(markdown_content, {}, 0.0)

    # Align
    anchor_spans = align_fn(
        markdown_content,
        flat_anchors,
        uniqueness_threshold=alignment_uniqueness_threshold,
        min_overlap=alignment_min_overlap,
    )

    # Calculate coverage
    coverage_percent = 0.0
    if markdown_content:
        from . import range_ops

        spans = sorted(anchor_spans.values())
        covered_ranges = range_ops.union_ranges(spans, [])
        covered_len = sum(end - start for start, end in covered_ranges)
        coverage_percent = covered_len / len(markdown_content)

    return AlignmentResult(
        markdown_content=markdown_content,
        anchor_spans=anchor_spans,
        coverage_percent=coverage_percent,
    )
