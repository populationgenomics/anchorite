"""Derive Anchors by aligning Markdown segments to PDF characters.

Given a Markdown document and the corresponding PDF, this module:

1. Parses the Markdown into fine-grained segments — headings, individual
   sentences, list items, blockquote lines, affiliation entries.  When the
   Markdown carries ``<!--page-->`` page-break markers (the typical chunked-OCR
   shape), they seed each segment's page hint.  When no markers are present
   (e.g. JATS-derived Markdown, where the source has no notion of pages), the
   page is left to fall out of the alignment.
2. Extracts per-character bounding boxes from the PDF using pypdfium2.
3. Aligns each segment's normalised text against the flat character text of its
   candidate page(s) using Smith-Waterman local alignment.  With a page hint,
   the search is restricted to a window around it; without one, phase 1
   searches every page and relies on its uniqueness ratio to discriminate.
4. Unions the bounding boxes of the matched characters to produce an ``Anchor``
   for each segment.

This inverts the existing flow (OCR anchors → align to markdown) so that the
richer semantic structure of the Markdown drives anchor granularity rather than
the accidents of PDF typesetting.
"""

import dataclasses
import logging
import pathlib
from collections.abc import Callable
from typing import Literal, overload

import pypdfium2 as pdfium
import seq_smith

from .anchors import Anchor
from .md_segments import MarkdownSegment, parse_markdown_segments
from .normalize import (
    SCORE_MATRIX_LOOSE,
    SCORE_MATRIX_STRICT,
    normalize_loose,
    normalize_strict,
)
from .pdf_atoms import (
    PageData,
    extract_page_data,
    line_bboxes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alignment configuration
# ---------------------------------------------------------------------------

_GAP_OPEN, _GAP_EXTEND = -2, -2
_MIN_SCORE = 10

_NormFn = Callable[..., tuple[bytes, tuple[int, ...]]]

# Phase 1 (conservative HSP-based): pages to search around the page marker.
_PHASE1_PAGE_SLACK = 10
# Phase 1: best score must be >= this multiple of the second-best (cross-page AND within-page).
_PHASE1_UNIQUENESS_RATIO = 2.0
# Phase 1: fraction of the normalised segment that must be covered by the best HSP.
_PHASE1_MIN_COVERAGE = 0.9
# Phase 1: segments with fewer alphanum chars than this are skipped (too short to be unique).
_PHASE1_MIN_LEN = 10


# ---------------------------------------------------------------------------
# Range / residual helpers
# ---------------------------------------------------------------------------


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent integer ranges into a sorted, disjoint list."""
    merged: list[list[int]] = []
    for s, e in sorted(ranges):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _residual_string(
    flat_str: str,
    covered: list[tuple[int, int]],
) -> tuple[str, list[int]]:
    """Return the uncovered portions of *flat_str* concatenated, plus a position map.

    ``pos_map[i]`` is the index of ``result[i]`` in the original *flat_str*.
    A sentinel ``pos_map[-1] == len(flat_str)`` is appended so that exclusive
    end indices can be looked up safely.
    """
    parts: list[str] = []
    pos_map: list[int] = []
    prev = 0
    for s, e in covered:
        if s > prev:
            parts.append(flat_str[prev:s])
            pos_map.extend(range(prev, s))
        prev = e
    if prev < len(flat_str):
        parts.append(flat_str[prev:])
        pos_map.extend(range(prev, len(flat_str)))
    pos_map.append(len(flat_str))  # sentinel
    return "".join(parts), pos_map


def _aln_to_flat_ranges(
    aln: seq_smith.Alignment,
    ref_to_flat: tuple[int, ...],
) -> list[tuple[int, int]]:
    flat_ranges: list[tuple[int, int]] = []
    for frag in aln.fragments:
        if frag.fragment_type != seq_smith.FragmentType.Match:
            continue
        flat_ranges.append(
            (
                ref_to_flat[frag.sa_start],
                ref_to_flat[frag.sa_start + frag.len],
            ),
        )
    return flat_ranges


def _align_against(
    reference: bytes,
    ref_to_flat: tuple[int, ...],
    norm_seg: bytes,
    score_matrix: object,
    min_score: int,
) -> tuple[int, list[tuple[int, int]]] | None:
    """Run Smith-Waterman and return (score, flat_ranges) or None if below threshold.

    ``seq_smith`` returns the *last* maximum-scoring alignment when multiple
    positions tie.  To get the *earliest* (reading-order) match, we re-run
    on progressively shorter prefixes of the reference until no earlier
    match at the same score exists.
    """
    if not norm_seg:
        return None
    aln = seq_smith.local_align(reference, norm_seg, score_matrix, _GAP_OPEN, _GAP_EXTEND)
    if aln.score < min_score:
        return None
    # Reject weak partial hits: require at least half the segment to be covered.
    # This catches cases like matching only "conflicting" from "Conflicting
    # interpretations" when the heading doesn't appear in the PDF.
    seg_covered = sum(f.len for f in aln.fragments if f.fragment_type == seq_smith.FragmentType.Match)
    if seg_covered * 2 < len(norm_seg):
        return None
    best_score = aln.score

    # Iteratively search for an earlier match with the same score.
    current_aln = aln
    while True:
        match_starts = [f.sa_start for f in current_aln.fragments if f.fragment_type == seq_smith.FragmentType.Match]
        if not match_starts:
            break
        cutoff = min(match_starts)
        if cutoff == 0:
            break  # already at the start
        earlier_aln = seq_smith.local_align(
            reference[:cutoff],
            norm_seg,
            score_matrix,
            _GAP_OPEN,
            _GAP_EXTEND,
        )
        if earlier_aln.score < best_score:
            break  # no earlier match reaches the same score
        current_aln = earlier_aln

    return best_score, _aln_to_flat_ranges(current_aln, ref_to_flat)


@dataclasses.dataclass(frozen=True)
class _AlignmentOutcome:
    """Per-segment alignment result, parallel to ``parse_markdown_segments(md)``.

    ``anchors`` and ``passes`` are the public ``associate()`` outputs.
    """

    anchors: list[Anchor]
    """Matched anchors in markdown order (unmatched segments omitted)."""
    passes: list[int]
    """Parallel to ``anchors``: 1 = phase 1, 2 = phase 2."""


def _align_markdown_to_pages(  # noqa: C901, PLR0912, PLR0915
    page_data: list[PageData],
    markdown: str,
    min_score: int = _MIN_SCORE,
) -> _AlignmentOutcome:
    """Two-phase alignment of markdown segments against pre-extracted page data.

    Behaviour and tuning are identical to the previous in-line implementation
    inside ``associate()``; the body has only been parameterised on
    ``page_data`` so a second consumer (``pdf_index.PdfIndex``) can reuse the
    alignment without re-reading the PDF.
    """
    segments = parse_markdown_segments(markdown)
    if not segments:
        return _AlignmentOutcome(anchors=[], passes=[])

    num_pages = len(page_data)

    # results[i]: Anchor for segments[i], or None if unmatched.
    results: list[Anchor | None] = [None] * len(segments)
    # confidence[i]: 1 = phase 1 (conservative), 2 = phase 2 (page-constrained).
    confidence: list[int] = [0] * len(segments)
    # Consumed flat-string ranges per page (raw; merged on demand).
    page_matched_ranges: dict[int, list[tuple[int, int]]] = {}

    def _atoms_from_flat_ranges(
        flat_to_atom: list[int],
        flat_ranges: list[tuple[int, int]],
    ) -> list[int]:
        indices: set[int] = set()
        for fs, fe in flat_ranges:
            indices.update(flat_to_atom[j] for j in range(fs, min(fe, len(flat_to_atom))))
        return sorted(indices)

    def _try_page_residual(  # noqa: C901
        page_idx: int,
        seg: MarkdownSegment,
        threshold: int,
    ) -> tuple[int, list[tuple[int, int]]] | None:
        """Align *seg* against the residual of *page_idx*.

        Returns ``(score, flat_ranges)`` on success, else ``None``.
        """
        if page_idx < 0 or page_idx >= num_pages:
            return None
        pd = page_data[page_idx]
        if not pd.atoms:
            return None

        covered = _merge_ranges(page_matched_ranges.get(page_idx, []))
        residual, pos_map = _residual_string(pd.atom_index.flat_str, covered)
        if not residual:
            return None

        def _align(
            norm_fn: _NormFn,
            score_matrix: object,
        ) -> tuple[int, list[tuple[int, int]]] | None:
            res_norm, res_to_res = norm_fn(residual)  # PDF
            seg_norm, _ = norm_fn(seg.text, strip_html=True)  # markdown
            if not seg_norm:
                return None
            hit = _align_against(res_norm, res_to_res, seg_norm, score_matrix, threshold)
            if hit is None:
                return None
            # ``hit[1]`` ranges are in *residual* coordinates; map back through
            # ``pos_map`` to original flat positions.  ``pos_map`` is
            # non-contiguous when the residual was stitched from multiple
            # uncovered slices — a single residual range may correspond to
            # several disjoint flat ranges, with previously-matched chars
            # masked between them.  Naively taking
            # ``(pos_map[rs], pos_map[re])`` would re-include those masked
            # chars and inflate the matched bbox set across content the
            # alignment never actually claimed (e.g. into a neighbouring
            # sentence whose lines fell between two matched chunks).
            flat_ranges: list[tuple[int, int]] = []
            for rs, rend in hit[1]:
                if rs >= rend:
                    continue
                run_start = pos_map[rs]
                prev = run_start
                for k in range(rs + 1, rend):
                    p = pos_map[k]
                    if p != prev + 1:
                        flat_ranges.append((run_start, prev + 1))
                        run_start = p
                    prev = p
                flat_ranges.append((run_start, prev + 1))
            return hit[0], flat_ranges

        result = _align(normalize_strict, SCORE_MATRIX_STRICT)
        if result is None:
            result = _align(normalize_loose, SCORE_MATRIX_LOOSE)
        return result

    def _accept_match(
        seg: MarkdownSegment,
        i: int,
        flat_ranges: list,
        matched_page: int,
        conf: int,
    ) -> None:
        pd = page_data[matched_page]
        atom_indices = _atoms_from_flat_ranges(pd.atom_index.flat_to_atom, flat_ranges)
        if not atom_indices:
            return
        matched_atoms = [pd.atoms[j] for j in atom_indices]
        boxes = tuple(
            line_bboxes(
                matched_atoms,
                pd.width,
                pd.height,
                pd.origin_x,
                pd.origin_y,
            ),
        )
        if boxes:
            results[i] = Anchor(text=seg.text, page=matched_page, boxes=boxes)
            confidence[i] = conf
            page_matched_ranges.setdefault(matched_page, []).extend(flat_ranges)

    # ── Phase 1: conservative HSP-based page assignment ──────────────────────
    # Normalise segment and page to alphanumeric only (no spaces).  Collect
    # the top-2 ungapped HSPs per candidate page, pool them globally, then
    # accept the best one only when (a) it covers ≥ _PHASE1_MIN_COVERAGE of
    # the segment and (b) it scores ≥ _PHASE1_UNIQUENESS_RATIO × the second-
    # best HSP *anywhere* (same page or a different page — the location of
    # the runner-up is irrelevant; only the score gap matters for whether
    # the best hit is unambiguous).

    # Lazy cache: alphanum-only bytes per page.
    page_alphanum_bytes: dict[int, bytes] = {}

    def _get_alphanum_page(page_idx: int) -> bytes:
        if page_idx not in page_alphanum_bytes:
            ci = page_data[page_idx].atom_index
            norm_bytes, _ = normalize_loose(ci.flat_str)  # PDF: don't strip HTML
            page_alphanum_bytes[page_idx] = norm_bytes
        return page_alphanum_bytes[page_idx]

    # seg_idx → PDF page index assigned by phase 1.
    phase1_page: dict[int, int] = {}

    for i, seg in enumerate(segments):
        if seg.page is not None and seg.page >= num_pages:
            continue

        norm_seg, _ = normalize_loose(seg.text, strip_html=True)
        if len(norm_seg) < _PHASE1_MIN_LEN:
            continue  # too short to identify uniquely

        # Without a page hint, search every page; the uniqueness check below
        # still suppresses ambiguous matches.  With a hint, restrict to a
        # window around it for cost.
        if seg.page is None:
            candidate_pages = list(range(num_pages))
        else:
            p_lo = max(0, seg.page - _PHASE1_PAGE_SLACK)
            p_hi = min(num_pages - 1, seg.page + _PHASE1_PAGE_SLACK)
            candidate_pages = list(range(p_lo, p_hi + 1))
        page_norms = [_get_alphanum_page(p) for p in candidate_pages]

        # Top-2 ungapped HSPs of segment vs each candidate page.
        top2_per_page = seq_smith.top_k_ungapped_local_align_many(
            norm_seg,
            page_norms,
            SCORE_MATRIX_LOOSE,
            k=2,
            filter_overlap_a=False,
            filter_overlap_b=False,
        )

        # Pool every HSP across every candidate page; pick the global best
        # and runner-up regardless of which page each lives on.
        pooled: list[tuple[int, int, int]] = []  # (score, len, page_idx)
        for page_idx, hsps in zip(candidate_pages, top2_per_page, strict=True):
            for hsp in hsps:
                pooled.append((hsp.score, hsp.stats.len, page_idx))
        if not pooled:
            continue
        pooled.sort(reverse=True)
        best_score, best_len, best_page = pooled[0]

        # Coverage: best HSP must span ≥ _PHASE1_MIN_COVERAGE of the segment.
        if best_len < len(norm_seg) * _PHASE1_MIN_COVERAGE:
            continue

        # Uniqueness: best score must beat second-best by the configured
        # ratio.  Score-only comparison; the runner-up's page doesn't matter.
        if len(pooled) >= 2 and pooled[1][0] * _PHASE1_UNIQUENESS_RATIO > best_score:  # noqa: PLR2004
            continue

        phase1_page[i] = best_page

    # ── Phase 1 refinement: full SW alignment on the assigned page ────────────
    # Process in document order so residuals accumulate correctly across segments
    # on the same page.
    for i in sorted(phase1_page.keys()):
        seg = segments[i]
        matched_page = phase1_page[i]
        norm_len = len(normalize_strict(seg.text, strip_html=True)[0]) or len(
            normalize_loose(seg.text, strip_html=True)[0],
        )
        threshold = max(5, min(min_score, norm_len))
        result = _try_page_residual(matched_page, seg, threshold)
        if result is not None:
            _, flat_ranges = result
            _accept_match(seg, i, flat_ranges, matched_page, 1)

    phase1_count = sum(1 for r in results if r is not None)
    logger.info(
        "Phase 1 (conservative HSP): %d/%d segments matched (%d%%)",
        phase1_count,
        len(segments),
        100 * phase1_count // max(len(segments), 1),
    )

    # ── Phase 2: page-constrained matching ────────────────────────────────────
    # For each segment not matched in phase 1, the document-order constraint
    # limits it to pages in [prev_matched_page, next_matched_page].  Take the
    # highest-scoring hit in that interval; no uniqueness requirement (the
    # narrow window suppresses false positives).
    for i, seg in enumerate(segments):
        if results[i] is not None:
            continue
        if seg.page is not None and seg.page >= num_pages:
            continue

        norm_len = len(normalize_strict(seg.text, strip_html=True)[0]) or len(
            normalize_loose(seg.text, strip_html=True)[0],
        )
        threshold = max(5, min(min_score, norm_len))

        prev_page: int | None = None
        for j in range(i - 1, -1, -1):
            if results[j] is not None:
                prev_page = results[j].page
                break
        next_page: int | None = None
        for j in range(i + 1, len(results)):
            if results[j] is not None:
                next_page = results[j].page
                break
        p2_lo = prev_page if prev_page is not None else 0
        p2_hi = next_page if next_page is not None else num_pages - 1

        best: tuple[int, list, int] | None = None
        for page in range(p2_lo, p2_hi + 1):
            candidate = _try_page_residual(page, seg, threshold)
            if candidate is not None and (best is None or candidate[0] > best[0]):
                best = (candidate[0], candidate[1], page)

        if best is not None:
            _, flat_ranges, matched_page = best
            _accept_match(seg, i, flat_ranges, matched_page, 2)

    anchors = [a for a in results if a is not None]
    passes = [c for a, c in zip(results, confidence, strict=True) if a is not None]
    return _AlignmentOutcome(anchors=anchors, passes=passes)


@overload
def associate(
    pdf_path: pathlib.Path,
    markdown: str,
    min_score: int = ...,
    return_pass_info: Literal[False] = ...,
) -> list[Anchor]: ...


@overload
def associate(
    pdf_path: pathlib.Path,
    markdown: str,
    min_score: int = ...,
    *,
    return_pass_info: Literal[True],
) -> tuple[list[Anchor], list[int]]: ...


def associate(
    pdf_path: pathlib.Path,
    markdown: str,
    min_score: int = _MIN_SCORE,
    return_pass_info: bool = False,
) -> list[Anchor] | tuple[list[Anchor], list[int]]:
    """Align each Markdown segment to the PDF and return one Anchor per segment.

    Uses a two-phase approach:

    **Phase 1 (conservative):** Normalise both segment and page text to
    alphanumeric characters only (no spaces), then run ungapped local alignment
    (HSPs) with k=2.  A segment is assigned to a page only when:

    * The best HSP covers ≥ ``_PHASE1_MIN_COVERAGE`` of the segment.
    * The best HSP is ≥ ``_PHASE1_UNIQUENESS_RATIO`` × the second-best, both
      *within* the winning page and *across* all candidate pages.

    Accepted segments are then precisely aligned (with spaces, gapped SW) to
    the *residual* of their assigned page to obtain bounding boxes.

    **Phase 2 (page-constrained):** Segments not matched in phase 1 are
    re-attempted using the document-order constraint: since the Markdown is in
    reading order, any unmatched segment must lie between the pages of its
    nearest matched neighbours.  The search range ``[prev_page, next_page]``
    is derived from the phase-1 results; no uniqueness requirement applies.

    Args:
        pdf_path: Path to the PDF file.
        markdown: The Markdown to align.  May contain ``<!--page-->`` page-
            break markers (used as phase-1 search-window hints); if omitted,
            phase 1 searches every page.
        min_score: Score cap for the adaptive alignment threshold.
        return_pass_info: If True, return ``(anchors, passes)`` where *passes*
            is a parallel list of ints: 1 = phase 1, 2 = phase 2.

    Returns:
        One ``Anchor`` per successfully matched segment, in Markdown order.
        Segments that cannot be matched are omitted.
        When *return_pass_info* is True, returns ``(anchors, passes)``.
    """
    doc = pdfium.PdfDocument(pdf_path)
    page_data = extract_page_data(doc)
    outcome = _align_markdown_to_pages(page_data, markdown, min_score)
    if return_pass_info:
        return outcome.anchors, outcome.passes
    return outcome.anchors
