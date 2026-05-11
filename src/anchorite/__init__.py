"""Anchorite: spatial text alignment connecting Markdown to PDF bounding boxes."""

import bisect
import dataclasses
import logging
import re
import string
from collections.abc import Mapping, Sequence

import seq_smith

from . import (
    bbox_alignment,
    document,
    markdown,
    md_association,
    md_segments,
    normalize,
    orchestrator,
    pdf_index,
    providers,
    range_ops,
)
from .anchors import Anchor, BBox
from .md_segments import MarkdownSegment, parse_markdown_segments
from .normalize import normalize_strict
from .orchestrator import AlignmentResult, process_document
from .pdf_index import PdfIndex

__all__ = [
    "AlignmentResult",
    "Anchor",
    "BBox",
    "MarkdownSegment",
    "PdfIndex",
    "SpanAnchor",
    "align",
    "annotate",
    "document",
    "markdown",
    "md_association",
    "md_segments",
    "normalize",
    "orchestrator",
    "parse_markdown_segments",
    "pdf_index",
    "process_document",
    "providers",
    "quote_locates",
    "range_ops",
    "resolve",
    "resolve_quote",
    "strip",
]

logger = logging.getLogger(__name__)

# --- Internal Alignment Constants & Helpers ---

# Sentinel character used to mask already-matched portions of the query.
# It must not appear in normalised text. Its row and column in the score matrix
# are set to a large negative value so the aligner never matches through it.
_MASK_CHAR = "#"

# Alignment alphabet used by the resolvers: the strict normaliser's alphabet
# (lowercase ASCII + digits + space) plus the mask sentinel ``#``.  Bytes 0..36
# round-trip identically against ``normalize.ALIGN_ALPHABET_STRICT`` so the
# resolver's score matrix accepts directly-normalised quotes without
# re-encoding.
_ALIGN_ALPHABET = string.ascii_lowercase + string.digits + " " + _MASK_CHAR
_NON_WORD_CHARS = seq_smith.encode(" ", _ALIGN_ALPHABET)
_SCORE_MATRIX = seq_smith.make_score_matrix(_ALIGN_ALPHABET, +1, -1)
_MASK_BYTE: int = seq_smith.encode(_MASK_CHAR, _ALIGN_ALPHABET)[0]
_SPACE_BYTE: int = seq_smith.encode(" ", _ALIGN_ALPHABET)[0]
_SCORE_MATRIX[_MASK_BYTE, :] = -100
_SCORE_MATRIX[:, _MASK_BYTE] = -100

_GAP_OPEN, _GAP_EXTEND = -2, -2

# Minimum alignment score to accept a quote match (roughly 15 matched chars).
_MIN_ALIGNMENT_SCORE = 15
# Coverage thresholds: warn below 50%, reject below 30%.
_WARN_COVERAGE = 0.5
_FAIL_COVERAGE = 0.3


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


def annotate(
    markdown: str,
    alignment: Mapping[Anchor, tuple[int, int]],
) -> str:
    """Inject coordinate ``<span>`` tags into Markdown at aligned positions.

    Produces ``<span data-bbox="t,l,b,r" data-page="N">text</span>`` for each
    anchor in ``alignment``. Span boundaries are snapped outward to the edges of
    any enclosing math block (``$...$`` or ``$$...$$``) to avoid splitting LaTeX.
    Overlapping and nested spans are handled by inserting tags in sorted order.

    Args:
        markdown: The plain Markdown string to annotate.
        alignment: Mapping of Anchor -> ``(start, end)`` character offsets,
            as returned by ``align``.

    Returns:
        Annotated Markdown string with embedded ``<span>`` tags.
    """
    math_ranges = []
    # Pattern matches $$...$$ (DOTALL) or $...$ (inline, allowing newlines for wrapped text)
    pattern = re.compile(r"(\$\$[\s\S]+?\$\$|\$[^$]+?\$)")
    for m in pattern.finditer(markdown):
        math_ranges.append((m.start(), m.end()))

    insertions = []
    for i, (anchor, (span_start, span_end)) in enumerate(alignment.items()):
        if span_start == span_end:
            continue
        start, end = span_start, span_end
        # Check for overlap with math ranges
        for m_start, m_end in math_ranges:
            if max(start, m_start) < min(end, m_end):
                # Snap to the math range
                start = min(start, m_start)
                end = max(end, m_end)
                break

        length = end - start
        box_str = ";".join(f"{b.top},{b.left},{b.bottom},{b.right}" for b in anchor.boxes)
        start_tag = f'<span data-bbox="{box_str}" data-page="{anchor.page}">'
        end_tag = "</span>"

        insertions.append((start, False, length, i, start_tag))
        insertions.append((end, True, length, i, end_tag))

    # Sorting rules:
    # * we wish to process in descending position order
    # * if there are position ties (n.b. later insertions end up to the left of earlier):
    #   * we want to insert closing spans to the left of opening spans, so we want to process opening spans first.
    #   * if there are opening span ties:
    #     * we want to insert longer spans to the left of shorter (so we want to process shorter spans first).
    #   * if there are closing span ties:
    #     * we want to insert longer spans to the right of shorter (so we want to process longer spans first).
    #   * if position, kind, and length are all equal (identical ranges):
    #     * the first-seen span (lower i) is treated as outer.

    def _key(x: tuple[int, bool, int, int, str]) -> tuple[int, bool, int, int]:
        position, is_closing, length, span_index, _tag = x
        return (
            -position,
            is_closing,  # opening spans sort before closing.
            -length if is_closing else +length,  # longer closing spans and shorter opening spans first.
            +span_index if is_closing else -span_index,  # first-seen span is outer.
        )

    insertions.sort(key=_key)

    chars = list(markdown)
    for index, _, _, _, text in insertions:
        chars.insert(index, text)

    return "".join(chars)


@dataclasses.dataclass(frozen=True)
class StrippedMarkdown:
    """Markdown content with tags stripped and a validation map."""

    plain_text: str
    """The plain text with all anchor spans removed."""
    validation_map: list[tuple[int, int, Anchor]]
    """A list of (start, end, Anchor) ranges in plain_text."""


def strip(annotated_md: str) -> StrippedMarkdown:
    """Remove ``<span>`` annotation tags and build a validation map.

    Returns a ``StrippedMarkdown`` with two fields:

    - ``plain_text``: the Markdown with all anchor spans removed.
    - ``validation_map``: sorted list of ``(start, end, Anchor)`` tuples giving
      each anchor's character range in ``plain_text``.

    The validation map can be used to verify that a generated quote is grounded
    in the source document — see ``resolve`` for the higher-level interface.
    """
    # Regex to find <span data-bbox="..." data-page="...">...</span>
    token_pattern = re.compile(
        r'(?P<start><span data-bbox="(?P<bbox>-?\d+,-?\d+,-?\d+,-?\d+(?:;-?\d+,-?\d+,-?\d+,-?\d+)*)"'
        r' data-page="(?P<page>\d+)">)|(?P<end></span>)',
    )

    plain_chars = []
    validation_map = []
    # Stack stores (start_index_in_plain_text, anchor_object)
    stack = []
    last_pos = 0
    current_plain_pos = 0

    for match in token_pattern.finditer(annotated_md):
        # Text before the tag
        before = annotated_md[last_pos : match.start()]
        plain_chars.append(before)
        current_plain_pos += len(before)

        if match.group("start"):
            bbox_str = match.group("bbox")
            page = int(match.group("page"))
            boxes = tuple(BBox(*[int(x) for x in group.split(",")]) for group in bbox_str.split(";"))
            anchor = Anchor(text="", page=page, boxes=boxes)
            stack.append((current_plain_pos, anchor))
        elif stack:
            start_plain_pos, anchor = stack.pop()
            validation_map.append((start_plain_pos, current_plain_pos, anchor))

        last_pos = match.end()

    plain_chars.append(annotated_md[last_pos:])
    return StrippedMarkdown(
        plain_text="".join(plain_chars),
        validation_map=sorted(validation_map),
    )


def _collect_overlapping_anchors(
    text_start: int,
    text_end: int,
    validation_map: list[tuple[int, int, Anchor]],
    found_locations: list[tuple[int, BBox]],
) -> None:
    """Internal helper to find and collect all anchors overlapping with a text range."""
    for b_start, b_end, anchor in validation_map:
        if b_start >= text_end:
            break
        if b_end > text_start:
            found_locations.extend((anchor.page, box) for box in anchor.boxes)


def _process_alignment(
    alignment: seq_smith.Alignment,
    text_mapping: Sequence[int],
    validation_map: list[tuple[int, int, Anchor]],
    current_norm_quote: bytearray,
    found_locations: list[tuple[int, BBox]],
) -> int:
    """Process alignment fragments, mask query, and collect overlapping anchors.

    Returns the total number of matched characters in the query for this alignment.
    """
    iteration_matched_len = 0
    for frag in alignment.fragments:
        if frag.fragment_type == seq_smith.FragmentType.BGap:
            continue

        # Mask the consumed portion of the query
        for i in range(frag.sb_start, frag.sb_start + frag.len):
            current_norm_quote[i] = _MASK_BYTE

        if frag.fragment_type == seq_smith.FragmentType.Match:
            _collect_overlapping_anchors(
                text_mapping[frag.sa_start],
                text_mapping[frag.sa_start + frag.len],
                validation_map,
                found_locations,
            )
            iteration_matched_len += frag.len
        else:
            # AGap also consumes query length but doesn't map to text
            iteration_matched_len += frag.len

    return iteration_matched_len


def _fuzzy_resolve_quote(
    norm_text: bytes,
    text_mapping: Sequence[int],
    validation_map: list[tuple[int, int, Anchor]],
    quote: str,
) -> list[tuple[int, BBox]]:
    """Internal helper to resolve a single quote using iterative fuzzy matching."""
    clean_quote = quote.strip()
    if not clean_quote:
        return []

    norm_quote, _ = normalize_strict(clean_quote, strip_html=True)
    if not norm_quote:
        return []

    found_locations: list[tuple[int, BBox]] = []
    current_norm_quote = bytearray(norm_quote)
    matched_len = 0
    total_len = len(norm_quote)

    for _ in range(10):  # Cap iterations
        if all(b == _MASK_BYTE for b in current_norm_quote):
            break

        alignment = seq_smith.local_align(norm_text, bytes(current_norm_quote), _SCORE_MATRIX, _GAP_OPEN, _GAP_EXTEND)
        if alignment.score < _MIN_ALIGNMENT_SCORE:
            break

        iteration_matched = _process_alignment(
            alignment,
            text_mapping,
            validation_map,
            current_norm_quote,
            found_locations,
        )
        if iteration_matched == 0:
            break
        matched_len += iteration_matched

    if matched_len < total_len * _WARN_COVERAGE:
        logger.warning("Low coverage for quote alignment: %d/%d for quote '%s'", matched_len, total_len, quote)
        if matched_len < total_len * _FAIL_COVERAGE:
            return []

    return sorted(set(found_locations))


# TODO: flag non-unique matches in the resolve() return value so callers can warn
# when a quote is ambiguous.  The practical mitigation is to ask the LLM to supply
# enough context to make every citation unique in the document.
def resolve(
    annotated_md: str,
    quotes: Sequence[str],
) -> dict[str, list[tuple[int, BBox]]]:
    """Resolve verbatim quotes to bounding boxes using fuzzy iterative matching.

    Strips the annotation tags, then for each quote runs iterative Smith-Waterman
    local alignment against the plain text. Matched regions are masked after each
    alignment so the same span is not claimed twice. Quotes that cannot be matched
    with sufficient confidence (score < ``_MIN_ALIGNMENT_SCORE`` or coverage <
    ``_FAIL_COVERAGE``) return an empty list.

    Args:
        annotated_md: Annotated Markdown produced by ``annotate``.
        quotes: Verbatim strings to locate (e.g. citations extracted by an LLM).

    Returns:
        Mapping of quote -> list of ``(page, BBox)`` for every anchor that
        overlaps the matched region. A single quote may span multiple anchors.
    """
    stripped = strip(annotated_md)
    norm_text, text_mapping = normalize_strict(stripped.plain_text, strip_html=True)

    return {quote: _fuzzy_resolve_quote(norm_text, text_mapping, stripped.validation_map, quote) for quote in quotes}


def quote_locates(
    markdown: str,
    quote: str,
    *,
    min_score: int = _MIN_ALIGNMENT_SCORE,
    fail_coverage: float = _FAIL_COVERAGE,
) -> bool:
    """Return ``True`` iff ``quote`` aligns to ``markdown`` with sufficient confidence.

    Uses the same SW + masking pipeline as ``resolve_quote`` but skips the
    span-overlap step.  Suitable for validating that an LLM-emitted quote
    is grounded in the document before returning it to the user.
    """
    clean_quote = quote.strip()
    if not clean_quote:
        return False

    norm_quote, _ = normalize_strict(clean_quote, strip_html=True)
    if not norm_quote:
        return False

    norm_text, _ = normalize_strict(markdown, strip_html=True)

    current = bytearray(norm_quote)
    matched_len = 0
    total_len = len(norm_quote)
    for _ in range(10):
        if all(b == _MASK_BYTE for b in current):
            break
        alignment = seq_smith.local_align(
            norm_text,
            bytes(current),
            _SCORE_MATRIX,
            _GAP_OPEN,
            _GAP_EXTEND,
        )
        if alignment.score < min_score:
            break
        progressed = False
        for frag in alignment.fragments:
            if frag.fragment_type == seq_smith.FragmentType.BGap:
                continue
            for j in range(frag.sb_start, frag.sb_start + frag.len):
                current[j] = _MASK_BYTE
            matched_len += frag.len
            progressed = True
        if not progressed:
            break

    return matched_len >= total_len * fail_coverage


@dataclasses.dataclass(frozen=True)
class SpanAnchor:
    """A bounding box paired with its character range in the source Markdown.

    The span is a half-open ``[start, end)`` interval into the Markdown text
    used to derive the box.  ``resolve_quote`` walks a sequence of these,
    finding every entry whose span overlaps where the quote aligns.
    """

    span: tuple[int, int]
    """``(start, end)`` half-open char range in the source Markdown."""
    page: int
    """0-indexed PDF page."""
    box: BBox
    """Bounding box on that page."""


def resolve_quote(  # noqa: C901, PLR0912
    markdown: str,
    spans: Sequence[SpanAnchor],
    quote: str,
    *,
    min_score: int = _MIN_ALIGNMENT_SCORE,
    warn_coverage: float = _WARN_COVERAGE,
    fail_coverage: float = _FAIL_COVERAGE,
) -> list[tuple[int, BBox]]:
    """Locate ``quote`` in ``markdown`` and return overlapping anchor boxes.

    Iteratively SW-aligns the normalised quote against the normalised
    Markdown, masking matched query bytes after each pass so the same span
    isn't claimed twice.  For every matched fragment, every ``SpanAnchor``
    whose ``span`` overlaps the matched Markdown char range contributes its
    ``(page, box)`` to the output.  Duplicate ``(page, box)`` pairs are
    de-duplicated.

    The same normalisation that ``md_association.associate`` uses to derive
    the spans is used here: NFKD decomposition, HTML-tag stripping,
    zero-width combining marks.  Quote text that aligned cleanly when the
    bboxes were generated will therefore align cleanly here too — that
    consistency is the whole point of routing the resolver through this
    function rather than each consumer rolling its own SW pass.

    Args:
        markdown: The Markdown the bboxes were derived from.
        spans: Sequence of ``SpanAnchor`` — typically built from a stored
            ``bboxes.json``-style record list at load time.
        quote: A verbatim quote (LLM-extracted, etc.) to locate.
        min_score: Reject SW alignments scoring below this.  Defaults to
            ``_MIN_ALIGNMENT_SCORE``.
        warn_coverage: Log a warning when matched coverage falls below this
            fraction of the normalised quote.
        fail_coverage: Return ``[]`` when matched coverage falls below this
            fraction.

    Returns:
        Sorted ``[(page, box), ...]`` for every anchor whose span overlaps
        the matched markdown range.  Empty list on coverage failure.
    """
    clean_quote = quote.strip()
    if not clean_quote:
        return []

    norm_quote, _ = normalize_strict(clean_quote, strip_html=True)
    if not norm_quote:
        return []

    norm_text, text_mapping = normalize_strict(markdown, strip_html=True)

    # Sort spans by start once.  Multiple spans may share the same start —
    # one ``SpanAnchor`` per visual line of a multi-line anchor produces N
    # records all keyed at the same Markdown-character start position — so
    # the iteration uses ``bisect_left`` to bound the *upper* end of the
    # candidate window and then walks every span whose start is < text_end,
    # checking the overlap condition explicitly.  A naive
    # ``bisect_right - 1`` would land on the last duplicate-start record
    # and skip the earlier ones, returning only one bbox for a wrapped
    # sentence.
    sorted_spans = sorted(spans, key=lambda sa: sa.span[0])
    span_starts = [sa.span[0] for sa in sorted_spans]
    # Prefix max of span ends, in start order.  Used to skip spans whose
    # ends are all <= text_start: bisect_right on this monotonic
    # non-decreasing array yields the lowest index whose max_end exceeds
    # text_start, so every earlier span ended before the match and can be
    # ignored.
    max_end_prefix: list[int] = []
    running_max = 0
    for sa in sorted_spans:
        running_max = max(running_max, sa.span[1])
        max_end_prefix.append(running_max)

    found_locations: list[tuple[int, BBox]] = []
    current_norm_quote = bytearray(norm_quote)
    matched_len = 0
    total_len = len(norm_quote)

    for _ in range(10):  # iteration cap
        if all(b == _MASK_BYTE for b in current_norm_quote):
            break

        alignment = seq_smith.local_align(
            norm_text,
            bytes(current_norm_quote),
            _SCORE_MATRIX,
            _GAP_OPEN,
            _GAP_EXTEND,
        )
        if alignment.score < min_score:
            break

        match_found = False
        for frag in alignment.fragments:
            if frag.fragment_type == seq_smith.FragmentType.BGap:
                continue

            # Mask the consumed query bytes.
            for j in range(frag.sb_start, frag.sb_start + frag.len):
                current_norm_quote[j] = _MASK_BYTE
            matched_len += frag.len
            match_found = True

            if frag.fragment_type == seq_smith.FragmentType.Match:
                # Trim trailing / leading space bytes from the matched
                # reference run before mapping to Markdown char positions.
                # ``normalize_strict`` collapses any non-alnum run to a single
                # space byte and records the *first* original char of the
                # run.  A trailing-space match therefore advances
                # ``text_end`` past the entire whitespace run — into the
                # *next* segment's span on the page.  Trim back to the
                # last non-space byte so ``text_end`` points at one past
                # the last actual alnum char of the match.
                ms, me = frag.sa_start, frag.sa_start + frag.len
                while ms < me and norm_text[ms] == _SPACE_BYTE:
                    ms += 1
                while me > ms and norm_text[me - 1] == _SPACE_BYTE:
                    me -= 1
                if ms >= me:
                    continue
                text_start = text_mapping[ms]
                text_end = text_mapping[me]
                # All overlap candidates have ``span_start < text_end`` and
                # ``span_end > text_start``.  ``bisect_left`` on
                # ``span_starts`` bounds the upper end; ``bisect_right`` on
                # ``max_end_prefix`` skips the prefix of spans that have all
                # ended before ``text_start``.  The remaining window still
                # has individual spans that may not overlap, so each is
                # checked explicitly.
                hi = bisect.bisect_left(span_starts, text_end)
                lo = bisect.bisect_right(max_end_prefix, text_start)
                for sa in sorted_spans[lo:hi]:
                    if sa.span[1] > text_start:
                        found_locations.append((sa.page, sa.box))

        if not match_found:
            break

    if matched_len < total_len * warn_coverage:
        logger.warning(
            "Low coverage for quote alignment: %d/%d for quote %r",
            matched_len,
            total_len,
            quote,
        )
        if matched_len < total_len * fail_coverage:
            return []

    return sorted(set(found_locations))
