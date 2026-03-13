import dataclasses
import logging
import re
import string
from collections.abc import Sequence

import seq_smith

from . import bbox_alignment, document, markdown, orchestrator, providers, range_ops
from .orchestrator import AlignmentResult, process_document
from .types import Anchor, BBox

__all__ = [
    "AlignmentResult",
    "Anchor",
    "BBox",
    "align",
    "annotate",
    "document",
    "markdown",
    "orchestrator",
    "process_document",
    "providers",
    "range_ops",
    "resolve",
    "strip",
]

logger = logging.getLogger(__name__)

# --- Internal Alignment Constants & Helpers ---

# Sentinel character used to mask already-matched portions of the query.
# It must not appear in normalised text. Its row and column in the score matrix
# are set to a large negative value so the aligner never matches through it.
_MASK_CHAR = "#"

_ALIGN_ALPHABET = string.ascii_lowercase + string.digits + " " + _MASK_CHAR
_NON_WORD_CHARS = seq_smith.encode(" ", _ALIGN_ALPHABET)
_SCORE_MATRIX = seq_smith.make_score_matrix(_ALIGN_ALPHABET, +1, -1)
_MASK_BYTE: int = seq_smith.encode(_MASK_CHAR, _ALIGN_ALPHABET)[0]
_SCORE_MATRIX[_MASK_BYTE, :] = -100
_SCORE_MATRIX[:, _MASK_BYTE] = -100

_GAP_OPEN, _GAP_EXTEND = -2, -2

# Minimum alignment score to accept a quote match (roughly 15 matched chars).
_MIN_ALIGNMENT_SCORE = 15
# Coverage thresholds: warn below 50%, reject below 30%.
_WARN_COVERAGE = 0.5
_FAIL_COVERAGE = 0.3


@dataclasses.dataclass(frozen=True)
class _NormalizedSpan:
    source: str
    normalized: bytes
    normalized_to_source: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.normalized)

    def _trim(self) -> None:
        normalized = self.normalized.lstrip(_NON_WORD_CHARS)
        left_trimmed = len(self.normalized) - len(normalized)
        normalized = normalized.rstrip(_NON_WORD_CHARS)
        right_trimmed = len(self.normalized) - len(normalized) - left_trimmed
        if right_trimmed == 0:
            normalized_to_source = self.normalized_to_source[left_trimmed:]
        else:
            normalized_to_source = self.normalized_to_source[left_trimmed:-right_trimmed]
        object.__setattr__(self, "normalized", normalized)
        object.__setattr__(self, "normalized_to_source", normalized_to_source)

    def __post_init__(self) -> None:
        self._trim()


@dataclasses.dataclass(frozen=True)
class _AnchorFragment(_NormalizedSpan):
    anchor: Anchor


@dataclasses.dataclass(frozen=True)
class _DocumentFragment(_NormalizedSpan):
    page_range: tuple[int, int]


def _normalize(source: str, span: tuple[int, int] = (-1, -1)) -> tuple[bytes, tuple[int, ...]]:
    if span == (-1, -1):
        span = (0, len(source))

    def _normalize_char(c: str) -> str:
        if c.lower() in string.ascii_letters + string.digits:
            return c.lower()
        return " "

    s, e = span
    normalized: list[str] = []
    normalized_to_source: list[int] = []

    for i in range(s, e):
        n = _normalize_char(source[i])
        if n == " " and normalized and normalized[-1] == " ":
            continue
        normalized.append(n)
        normalized_to_source.append(i)

    normalized_to_source.append(e)
    return seq_smith.encode("".join(normalized), _ALIGN_ALPHABET), tuple(normalized_to_source)


def align(
    anchors: Sequence[Anchor],
    markdown: str,
    uniqueness_threshold: float = 0.5,
    min_overlap: float = 0.9,
) -> dict[Anchor, tuple[int, int]]:
    """
    Alignment of anchors to Markdown text.

    Iterative: ungapped alignment first, then gapped, until convergence.
    Filters by uniqueness, minimum overlap, and page consistency.

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
    alignment: dict[Anchor, tuple[int, int]],
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
    pattern = re.compile(r"(\$\$[\s\S]+?\$\$|\$(?:\.|[^$])+?\$)")
    for m in pattern.finditer(markdown):
        math_ranges.append((m.start(), m.end()))

    insertions = []
    for anchor, (span_start, span_end) in alignment.items():
        start, end = span_start, span_end
        # Check for overlap with math ranges
        for m_start, m_end in math_ranges:
            if max(start, m_start) < min(end, m_end):
                # Snap to the math range
                start = min(start, m_start)
                end = max(end, m_end)
                break

        length = end - start
        box_str = f"{anchor.box.top},{anchor.box.left},{anchor.box.bottom},{anchor.box.right}"
        start_tag = f'<span data-bbox="{box_str}" data-page="{anchor.page}">'
        end_tag = "</span>"

        insertions.append((start, False, length, start_tag))
        insertions.append((end, True, length, end_tag))

    insertions.sort(key=lambda x: (x[0], x[1], -x[2] if not x[1] else x[2]), reverse=True)

    chars = list(markdown)
    for index, _, _, text in insertions:
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
        r'(?P<start><span data-bbox="(?P<bbox>-?\d+,-?\d+,-?\d+,-?\d+)" data-page="(?P<page>\d+)">)|(?P<end></span>)',
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
            coords = [int(x) for x in bbox_str.split(",")]
            anchor = Anchor(text="", page=page, box=BBox(*coords))
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
            found_locations.append((anchor.page, anchor.box))


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

    norm_quote, _ = _normalize(clean_quote)
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


def resolve(
    annotated_md: str,
    quotes: list[str],
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
    norm_text, text_mapping = _normalize(stripped.plain_text)

    return {quote: _fuzzy_resolve_quote(norm_text, text_mapping, stripped.validation_map, quote) for quote in quotes}
