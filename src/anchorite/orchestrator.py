"""High-level orchestration of concurrent Markdown and anchor generation followed by alignment."""

import asyncio
import dataclasses
from collections.abc import Iterable

from . import anchors, bbox_alignment, document, markdown, providers


@dataclasses.dataclass
class AlignmentResult:
    """The result of aligning anchors to generated markdown."""

    markdown_content: str
    """The generated markdown content."""
    anchor_spans: dict[anchors.Anchor, tuple[int, int]]
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
    markdown_provider: providers.MarkdownProvider,
    anchor_provider: providers.AnchorProvider | providers.MarkdownAnchorProvider | None = None,
    *,
    alignment_uniqueness_threshold: float = 0.5,
    alignment_min_overlap: float = 0.9,
    renumber: bool = True,
) -> AlignmentResult:
    """Align anchors to Markdown across a set of pre-chunked document pages.

    Generates Markdown and anchors concurrently for all chunks, joins the Markdown
    with ``<!--page-->`` separators, then runs alignment across the full document.

    Supports two anchor provider styles:

    * ``AnchorProvider``: generates anchors independently per chunk, in parallel
      with Markdown generation.  Anchors are then aligned to the assembled Markdown
      via Smith-Waterman.
    * ``MarkdownAnchorProvider``: ``process_chunk`` (PDF char extraction) runs
      concurrently with Markdown generation; ``finalize`` receives the fully
      assembled Markdown and returns anchors with page info already set.  The
      anchors are then aligned to the Markdown to produce span positions.

    Args:
        chunks: Pre-chunked document pages, e.g. from ``anchorite.document.chunks()``.
            Markdown and anchors are generated for each chunk independently and in
            parallel, then assembled before alignment.
        markdown_provider: Generates Markdown text for a chunk (e.g. an LLM call).
            Run concurrently with anchor generation when ``anchor_provider`` is set.
        anchor_provider: Generates anchors for a chunk (e.g. an OCR call), or a
            ``MarkdownAnchorProvider`` that uses the assembled Markdown to guide
            anchor extraction from PDF char data.  If ``None``, alignment is skipped
            and an empty ``AlignmentResult`` is returned.
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

    if isinstance(anchor_provider, providers.MarkdownAnchorProvider):
        # Case 2: PDF char extraction runs concurrently with Markdown generation.
        # finalize() receives the assembled Markdown and returns anchors with pages.
        process_tasks = [anchor_provider.process_chunk(chunk) for chunk in chunk_list]
        markdown_chunks, _ = await asyncio.gather(
            asyncio.gather(*markdown_tasks),
            asyncio.gather(*process_tasks),
        )
        markdown_chunks = list(markdown_chunks)
        if renumber:
            markdown_chunks = markdown.renumber_markers(markdown_chunks)
        markdown_content = "\n\n<!--page-->\n\n".join(markdown_chunks)
        flat_anchors = await anchor_provider.finalize(markdown_content)
    elif anchor_provider is not None:
        # Case 1: OCR-style provider generates anchors independently per chunk.
        anchor_tasks = [anchor_provider.generate_anchors(chunk) for chunk in chunk_list]
        markdown_chunks, all_anchors = await asyncio.gather(
            asyncio.gather(*markdown_tasks),
            asyncio.gather(*anchor_tasks),
        )
        flat_anchors = [anchor for chunk_anchors in all_anchors for anchor in chunk_anchors]
        if renumber:
            markdown_chunks = markdown.renumber_markers(list(markdown_chunks))
        markdown_content = "\n\n<!--page-->\n\n".join(markdown_chunks)
    else:
        markdown_chunks = list(await asyncio.gather(*markdown_tasks))
        if renumber:
            markdown_chunks = markdown.renumber_markers(markdown_chunks)
        markdown_content = "\n\n<!--page-->\n\n".join(markdown_chunks)

    if anchor_provider is None:
        return AlignmentResult(markdown_content, {}, 0.0)

    # Align anchors to span positions in the assembled Markdown.
    anchor_spans = bbox_alignment.align_anchors(
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
