"""Iterative Smith-Waterman alignment of OCR anchors to Markdown text."""

import collections
import dataclasses
import logging
import re
import string
from collections.abc import Iterator, Sequence

import seq_smith

from . import anchors, range_ops

_ALIGN_ALPHABET = string.ascii_lowercase + string.digits + " "
_NON_WORD_CHARS = seq_smith.encode(" ", _ALIGN_ALPHABET)
_GAP_OPEN, _GAP_EXTEND = -2, -2
_SCORE_MATRIX = seq_smith.make_score_matrix(_ALIGN_ALPHABET, +1, -1)
_UNIQUENESS_THRESHOLD = 0.5
_MIN_OVERLAP = 0.9


@dataclasses.dataclass(frozen=True)
class _NormalizedSpan:
    """A span of text with its normalized form and mapping."""

    source: str
    """The original source text."""
    normalized: bytes
    """The normalized byte sequence used for alignment."""
    normalized_to_source: tuple[int, ...]
    """Mapping from normalized indices to source indices."""

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
    """A normalized span derived from an anchor."""

    anchor: anchors.Anchor
    """The original anchor object."""


@dataclasses.dataclass(frozen=True)
class _DocumentFragment(_NormalizedSpan):
    """A normalized span derived from the markdown document."""

    page_range: tuple[int, int]
    """Range of pages (start, end) this fragment might span."""


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

    span_bytes = seq_smith.encode("".join(normalized), _ALIGN_ALPHABET)
    return span_bytes, tuple(normalized_to_source)


def _make_anchor_fragment(anchor: anchors.Anchor) -> _AnchorFragment:
    span_bytes, normalized_to_source = _normalize(anchor.text)
    return _AnchorFragment(anchor.text, span_bytes, normalized_to_source, anchor)


def _make_document_fragment(
    source: str,
    page_range: tuple[int, int],
    span: tuple[int, int] = (-1, -1),
) -> _DocumentFragment:
    span_bytes, normalized_to_source = _normalize(source, span)
    return _DocumentFragment(source, span_bytes, normalized_to_source, page_range)


def _a_end(f: seq_smith.AlignmentFragment) -> int:
    return f.sa_start if f.fragment_type == seq_smith.FragmentType.AGap else f.sa_start + f.len


def _b_end(f: seq_smith.AlignmentFragment) -> int:
    return f.sb_start if f.fragment_type == seq_smith.FragmentType.BGap else f.sb_start + f.len


def _aligned_range(alignment: seq_smith.Alignment) -> tuple[int, int]:
    return (alignment.fragments[0].sa_start, _a_end(alignment.fragments[-1]))


def _make_document_fragments(markdown_content: str, page_range: tuple[int, int]) -> Iterator[_DocumentFragment]:
    start = 0
    for m in re.finditer(r"<!-{2,3}.*?-->", markdown_content):
        span = _make_document_fragment(markdown_content, page_range, (start, m.start()))
        if len(span):
            yield span
        start = m.end()
    if start < len(markdown_content):
        span = _make_document_fragment(markdown_content, page_range, (start, len(markdown_content)))
        if len(span):
            yield span


def _slice_document_fragment(
    span: _DocumentFragment,
    start: int,
    end: int,
    page_range: tuple[int, int],
) -> _DocumentFragment:
    if start >= end:
        return _DocumentFragment(span.source, b"", (), page_range)

    n_sub = span.normalized[start:end]
    # normalized_to_source has length len(normalized) + 1.
    # slice it from start to end + 1 to include the end boundary
    nts_sub = span.normalized_to_source[start : end + 1]

    return _DocumentFragment(span.source, n_sub, nts_sub, page_range)


def _compute_candidate_alignments(
    document_fragments: list[_DocumentFragment],
    anchor_spans: list[_AnchorFragment],
    with_ungapped: bool,
) -> dict[_AnchorFragment, list[tuple[int, seq_smith.Alignment]]]:
    anchor_span_hsps: dict[_AnchorFragment, list[tuple[int, seq_smith.Alignment]]] = collections.defaultdict(list)

    for i, fragment in enumerate(document_fragments):
        spans = [b for b in anchor_spans if range_ops.in_range(b.anchor.page, fragment.page_range)]
        if not spans:
            continue
        if with_ungapped:
            alignments = seq_smith.top_k_ungapped_local_align_many(
                fragment.normalized,
                [s.normalized for s in spans],
                _SCORE_MATRIX,
                k=2,
                filter_overlap_a=False,
                filter_overlap_b=False,
            )
            for s, a in zip(spans, alignments, strict=True):
                anchor_span_hsps[s].extend((i, hsp) for hsp in a)
        else:
            alignments = seq_smith.local_global_align_many(
                fragment.normalized,
                [s.normalized for s in spans],
                _SCORE_MATRIX,
                _GAP_OPEN,
                _GAP_EXTEND,
            )
            for s, a in zip(spans, alignments, strict=True):
                anchor_span_hsps[s].append((i, a))
    return anchor_span_hsps


def _assign_high_confidence_spans(
    document_fragments: list[_DocumentFragment],
    anchor_spans: Sequence[_AnchorFragment],
    uniqueness_threshold: float = _UNIQUENESS_THRESHOLD,
    min_overlap: float = _MIN_OVERLAP,
    with_ungapped: bool = True,
) -> Iterator[tuple[_DocumentFragment, list[_AnchorFragment]]]:
    anchor_spans_list = list(anchor_spans)
    anchor_span_hsps = _compute_candidate_alignments(document_fragments, anchor_spans_list, with_ungapped)

    assignments = collections.defaultdict(list)
    for anchor_span in anchor_spans_list:
        anchor_hsps = sorted(
            anchor_span_hsps[anchor_span],
            key=lambda x: x[1].score,
            reverse=True,
        )

        if not anchor_hsps:
            continue

        span_idx, hsp = anchor_hsps[0]
        if len(anchor_hsps) > 1 and anchor_hsps[1][1].score >= hsp.score * uniqueness_threshold:
            continue
        if hsp.stats.len >= min_overlap * len(anchor_span):
            assignments[span_idx].append((anchor_span, hsp))

    for i, assigned_anchor_spans in sorted(assignments.items()):
        assigned_anchor_spans.sort(key=lambda x: x[1].score, reverse=True)
        yield document_fragments[i], [x[0] for x in assigned_anchor_spans]


def _page_range_for_range(
    page_range: tuple[int, int],
    span_range: tuple[int, int],
    page_ranges: dict[int, tuple[int, int]],
) -> tuple[int, int]:
    """Return the page range for the span range."""
    page_start, page_end = page_range

    for page, r in page_ranges.items():
        if r[0] <= span_range[0]:
            page_start = max(page_start, page)
        if r[1] >= span_range[1]:
            page_end = min(page_end, page + 1)
    return page_start, page_end


def _is_consistent_with_page_ranges(
    r: tuple[int, int],
    page: int,
    page_ranges: dict[int, Sequence[tuple[int, int]]],
) -> bool:
    """Return true if range `r` inferred from a span on page `page` is consistent with `page_ranges`."""
    pre_ranges = [pr for p, pr in page_ranges.items() if p < page]
    post_ranges = [pr for p, pr in page_ranges.items() if p > page]
    lower_bound = max(pr[-1][1] for pr in pre_ranges) if pre_ranges else r[0]
    upper_bound = min(pr[0][0] for pr in post_ranges) if post_ranges else r[1]
    return range_ops.contained(r, (lower_bound, upper_bound))


def _assign_spans(
    document_fragment: _DocumentFragment,
    candidates: list[_AnchorFragment],
    match_fraction: float = 0.9,
    new_coverage_fraction: float = 0.9,
) -> tuple[list[tuple[_AnchorFragment, tuple[int, int]]], list[_DocumentFragment]]:
    candidates = [c for c in candidates if range_ops.in_range(c.anchor.page, document_fragment.page_range)]

    alignments = seq_smith.local_global_align_many(
        document_fragment.normalized,
        [c.normalized for c in candidates],
        _SCORE_MATRIX,
        _GAP_OPEN,
        _GAP_EXTEND,
    )

    candidates_alignments = sorted(zip(candidates, alignments, strict=True), key=lambda x: x[1].score, reverse=True)
    candidates = [x[0] for x in candidates_alignments]
    alignments = [x[1] for x in candidates_alignments]

    assignments = []
    covered: list[tuple[int, int]] = []

    page_ranges: dict[int, Sequence[tuple[int, int]]] = {}

    for candidate, alignment in zip(candidates, alignments, strict=True):
        if alignment.stats.num_exact_matches < len(candidate) * match_fraction:
            continue
        r = _aligned_range(alignment)
        if not _is_consistent_with_page_ranges(r, candidate.anchor.page, page_ranges):
            continue
        r_uncovered = range_ops.subtract_ranges([r], covered)
        uncovered_chars = sum(r[1] - r[0] for r in r_uncovered)
        if uncovered_chars < len(candidate) * new_coverage_fraction:
            continue

        page_ranges[candidate.anchor.page] = range_ops.union_ranges(
            r_uncovered,
            page_ranges.get(candidate.anchor.page, []),
        )

        covered = range_ops.union_ranges(r_uncovered, covered)

        doc_start = document_fragment.normalized_to_source[r_uncovered[0][0]]
        doc_end = document_fragment.normalized_to_source[r_uncovered[-1][1]]

        assignments.append((candidate, (doc_start, doc_end)))

    page_spans = {page: (r[0][0], r[-1][1]) for page, r in sorted(page_ranges.items())}

    new_spans = []
    uncovered = range_ops.subtract_ranges([(0, len(document_fragment.normalized))], covered)

    for start, end in uncovered:
        page_range = _page_range_for_range(document_fragment.page_range, (start, end), page_spans)
        frag = _slice_document_fragment(document_fragment, start, end, page_range)
        if len(frag):
            new_spans.append(frag)

    return assignments, new_spans


def _process_alignment_iteration(
    iteration_num: int,
    spans: list[_DocumentFragment],
    anchor_spans: set[_AnchorFragment],
    uniqueness_threshold: float,
    min_overlap: float = 0.9,
) -> tuple[list[_DocumentFragment], list[tuple[_AnchorFragment, tuple[int, int]]]]:
    logging.debug("--- Iteration %d (Threshold: %f, Overlap: %f) ---", iteration_num, uniqueness_threshold, min_overlap)
    logging.debug("%d spans, %d anchor spans.", len(spans), len(anchor_spans))

    candidates = list(
        _assign_high_confidence_spans(
            spans,
            sorted(anchor_spans, key=lambda x: (x.anchor.page, x.anchor.box.top, x.anchor.box.left, x.anchor.text)),
            uniqueness_threshold=uniqueness_threshold,
            min_overlap=min_overlap,
            with_ungapped=iteration_num == 1,
        ),
    )

    if not candidates:
        return spans, []

    new_spans = []
    all_assigned_ranges_in_iteration = []
    matched_span_ids = set()

    for doc_span, anchor_candidates in candidates:
        matched_span_ids.add(id(doc_span))
        assignments, holes = _assign_spans(doc_span, anchor_candidates)

        if assignments:
            all_assigned_ranges_in_iteration.extend(assignments)

        new_spans.extend(holes)

    # Keep spans that weren't involved in any match
    for s in spans:
        if id(s) not in matched_span_ids:
            new_spans.append(s)

    logging.debug("Assigned in this iteration: %s", bool(all_assigned_ranges_in_iteration))
    logging.debug(
        "Remaining anchors: %d",
        len(anchor_spans) - len({anchor for anchor, _ in all_assigned_ranges_in_iteration}),
    )
    logging.debug("New span count (holes + unvisited): %d", len(new_spans))
    return new_spans, all_assigned_ranges_in_iteration


def align_anchors(
    markdown_content: str,
    anchor_seq: Sequence[anchors.Anchor],
    uniqueness_threshold: float = _UNIQUENESS_THRESHOLD,
    min_overlap: float = _MIN_OVERLAP,
) -> dict[anchors.Anchor, tuple[int, int]]:
    """Align a sequence of anchors to their positions in a Markdown string.

    Runs the iterative Smith-Waterman alignment loop: ungapped on the first pass,
    gapped on subsequent passes. Each pass filters candidates by uniqueness and
    overlap, assigns them to character ranges within document fragments, then
    splits unmatched fragment regions for the next iteration.

    Args:
        markdown_content: The generated Markdown text to align against.
        anchor_seq: OCR-derived anchors, each carrying text, page number, and bounding box.
        uniqueness_threshold: An anchor is accepted only when its best-match score
            exceeds this fraction of its second-best score (prevents ambiguous matches).
        min_overlap: Minimum fraction of the anchor's normalised length that must
            be covered by the alignment.

    Returns:
        Mapping of Anchor -> (start, end) character offsets in markdown_content.
        Anchors that could not be matched with sufficient confidence are omitted.
    """

    # Create initial spans (just the full content)
    anchor_spans: set[_AnchorFragment] = {span for span in [_make_anchor_fragment(a) for a in anchor_seq] if len(span)}
    if not anchor_spans:
        return {}

    max_page = max(a.page for a in anchor_seq)
    spans = list(_make_document_fragments(markdown_content, (0, max_page + 1)))

    logging.debug("initial span count %d; initial anchor count %d", len(spans), len(anchor_spans))

    iteration = 0
    all_assigned_ranges: list[tuple[int, tuple[_AnchorFragment, tuple[int, int]]]] = []

    while True:
        iteration += 1
        spans, assigned_ranges = _process_alignment_iteration(
            iteration,
            spans,
            anchor_spans,
            uniqueness_threshold,
            min_overlap,
        )
        if assigned_ranges:
            all_assigned_ranges.extend((iteration, s) for s in assigned_ranges)

            # Update anchor_spans to remove assigned ones for next iteration
            assigned_set = {anchor for anchor, _ in assigned_ranges}
            anchor_spans = {b for b in anchor_spans if b not in assigned_set}

        if not anchor_spans or (iteration > 1 and not assigned_ranges):
            break

    logging.debug("Final remaining anchor count %d assigned count %d", len(anchor_spans), len(all_assigned_ranges))

    # Apply replacements for debugging
    # Sort ranges by start index descending to apply safely
    all_assigned_ranges.sort(key=lambda x: x[1][1][0], reverse=True)

    return {anchor_span.anchor: (start, end) for iteration, (anchor_span, (start, end)) in all_assigned_ranges}
