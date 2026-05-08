"""Anchorite: spatial text alignment connecting Markdown to PDF bounding boxes."""

from collections.abc import Sequence

from . import (
    annotation,
    bbox_alignment,
    document,
    markdown,
    md_association,
    md_segments,
    normalize,
    orchestrator,
    pdf_index,
    providers,
    quote_resolution,
    range_ops,
)
from .anchors import Anchor, BBox
from .annotation import StrippedMarkdown, annotate, strip
from .md_segments import MarkdownSegment, parse_markdown_segments
from .normalize import normalize_strict
from .orchestrator import AlignmentResult, process_document
from .pdf_index import PdfIndex
from .quote_resolution import SpanAnchor, quote_locates, resolve, resolve_quote

__all__ = [
    "AlignmentResult",
    "Anchor",
    "BBox",
    "MarkdownSegment",
    "PdfIndex",
    "SpanAnchor",
    "StrippedMarkdown",
    "align",
    "annotate",
    "annotation",
    "document",
    "markdown",
    "md_association",
    "md_segments",
    "normalize",
    "normalize_strict",
    "orchestrator",
    "parse_markdown_segments",
    "pdf_index",
    "process_document",
    "providers",
    "quote_locates",
    "quote_resolution",
    "range_ops",
    "resolve",
    "resolve_quote",
    "strip",
]


def align(
    anchors: Sequence[Anchor],
    markdown: str,
    uniqueness_threshold: float = 0.5,
    min_overlap: float = 0.9,
) -> dict[Anchor, tuple[int, int]]:
    """Align OCR anchors to character positions in a Markdown string.

    Iterative: ungapped alignment first, then gapped, until convergence.
    Filters by uniqueness, minimum overlap, and page consistency.

    Args:
        anchors: OCR-derived anchors to align.
        markdown: The Markdown string to align against.
        uniqueness_threshold: An anchor is accepted only when its best-match
            score exceeds this fraction of its second-best score.
        min_overlap: Minimum fraction of the anchor's normalised length that
            must be covered by the alignment.

    Returns:
        Mapping of Anchor -> (start_char, end_char) in markdown.
    """
    return bbox_alignment.align_anchors(
        markdown,
        anchors,
        uniqueness_threshold=uniqueness_threshold,
        min_overlap=min_overlap,
    )
