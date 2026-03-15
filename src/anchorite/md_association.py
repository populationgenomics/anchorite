"""Prototype: derive Anchors by aligning Markdown segments to PDF characters.

Given a cleaned Markdown document (with ``<!--page-->`` page-break markers) and
the corresponding PDF, this module:

1. Parses the Markdown into fine-grained segments — headings, individual
   sentences, list items, blockquote lines, affiliation entries — using
   ``<!--page-->`` markers to infer which PDF page each segment lives on.
2. Extracts per-character bounding boxes from the PDF using pypdfium2.
3. Aligns each segment's normalised text against the flat character text of its
   candidate page using Smith-Waterman local alignment.
4. Unions the bounding boxes of the matched characters to produce an ``Anchor``
   for each segment.

This inverts the existing flow (OCR anchors → align to markdown) so that the
richer semantic structure of the Markdown drives anchor granularity rather than
the accidents of PDF typesetting.
"""

import dataclasses
import math
import pathlib
import re
import string
from typing import NamedTuple

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import seq_smith

from .anchors import Anchor, BBox

# ---------------------------------------------------------------------------
# Character extraction
# ---------------------------------------------------------------------------

_CHAR_NORM: dict[str, str] = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u201a": ",",
    "\u2013": "-", "\u2014": "--", "\u2212": "-", "\u2010": "-", "\u2011": "-",
    "\u00ad": "", "\u00a0": " ", "\ufffe": "",
}


@dataclasses.dataclass(frozen=True)
class _Char:
    text: str
    x0: float
    y0: float  # bottom in PDF coords (pts, origin bottom-left)
    x1: float
    y1: float  # top in PDF coords
    font_size: float


def _extract_page_chars(page: pdfium.PdfPage) -> list[_Char]:
    """Extract non-whitespace chars with bboxes from a single page."""
    textpage = page.get_textpage()
    total_chars = textpage.count_chars()
    chars: list[_Char] = []
    char_index = 0

    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_TEXT]):
        buf_size = pdfium_c.FPDFTextObj_GetText(obj, textpage, None, 0)
        buf = (pdfium_c.FPDF_WCHAR * buf_size)()
        pdfium_c.FPDFTextObj_GetText(obj, textpage, buf, buf_size)
        obj_text = bytes(buf).decode("utf-16-le").rstrip("\x00")

        m = obj.get_matrix()
        font_size = obj.get_font_size() * math.sqrt(m.a ** 2 + m.b ** 2)

        obj_pos = 0
        while obj_pos < len(obj_text) and char_index < total_chars:
            text = textpage.get_text_range(char_index, 1)
            if text in ("\r", "\n"):
                char_index += 1
                continue
            if not text.isspace():
                normalized = _CHAR_NORM.get(text, text)
                if normalized:
                    left, bottom, right, top = textpage.get_charbox(char_index, loose=False)
                    if right > left and top > bottom:
                        chars.append(_Char(normalized, left, bottom, right, top, font_size))
            char_index += 1
            obj_pos += 1

    return chars


# ---------------------------------------------------------------------------
# Flat char string with position index
# ---------------------------------------------------------------------------

class _CharIndex(NamedTuple):
    flat_str: str
    """The flat text string built from the page's chars."""
    flat_to_char: list[int]
    """flat_to_char[i] = index into chars for flat_str[i]."""


def _build_char_index(chars: list[_Char]) -> _CharIndex:
    """Build a flat string and a per-character index mapping back to chars."""
    parts: list[str] = []
    flat_to_char: list[int] = []

    for i, ch in enumerate(chars):
        for c in ch.text:
            parts.append(c)
            flat_to_char.append(i)
        if i + 1 < len(chars):
            gap = chars[i + 1].x0 - ch.x1
            if gap > ch.font_size * 0.2:
                parts.append(" ")
                flat_to_char.append(i)

    return _CharIndex("".join(parts), flat_to_char)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_ALIGN_ALPHABET_STRICT = string.ascii_lowercase + string.digits + " "
_SCORE_MATRIX_STRICT = seq_smith.make_score_matrix(_ALIGN_ALPHABET_STRICT, +1, -1)
_ALIGN_ALPHABET_LOOSE = string.ascii_lowercase + string.digits
_SCORE_MATRIX_LOOSE = seq_smith.make_score_matrix(_ALIGN_ALPHABET_LOOSE, +1, -1)
_GAP_OPEN, _GAP_EXTEND = -2, -2
_MIN_SCORE = 10


def _normalize_strict(text: str) -> tuple[bytes, tuple[int, ...]]:
    """Lowercase + collapse non-alphanumeric runs to a single space."""
    normalized: list[str] = []
    idx_map: list[int] = []
    for i, c in enumerate(text):
        lc = c.lower()
        if lc in string.ascii_letters + string.digits:
            normalized.append(lc)
            idx_map.append(i)
        else:
            if normalized and normalized[-1] != " ":
                normalized.append(" ")
                idx_map.append(i)
    idx_map.append(len(text))
    return seq_smith.encode("".join(normalized), _ALIGN_ALPHABET_STRICT), tuple(idx_map)


def _normalize_loose(text: str) -> tuple[bytes, tuple[int, ...]]:
    """Keep only lowercase letters and digits; strip everything else.

    Used as a fallback for segments that fail the strict pass.  Discarding
    spaces means that letter-spaced display headings like
    ``C A S E  R E P O R T`` normalise to the same sequence as
    ``CASE REPORT``, at the cost of losing word-boundary information.
    """
    normalized: list[str] = []
    idx_map: list[int] = []
    for i, c in enumerate(text):
        lc = c.lower()
        if lc in string.ascii_letters + string.digits:
            normalized.append(lc)
            idx_map.append(i)
    idx_map.append(len(text))
    return seq_smith.encode("".join(normalized), _ALIGN_ALPHABET_LOOSE), tuple(idx_map)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"

_ABBREVIATIONS: frozenset[str] = frozenset({
    "al", "fig", "figs", "eq", "eqs", "vs", "etc",
    "dr", "mr", "mrs", "ms", "prof", "inc", "ltd", "co", "jr", "sr",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "vol", "no", "pp", "p", "ed", "eds", "ref", "refs",
    "approx", "dept", "est", "max", "min", "cf", "viz",
})

# Sentence boundary: terminal punctuation, optional reference markers
# (superscripts or a space-separated digit run), then whitespace, then uppercase.
_SENT_END_RE = re.compile(
    r"[.!?]"
    r"[" + _SUPERSCRIPT_DIGITS + r"]*"  # optional superscript refs directly after punct
    r"(?:\s+\d[\d,\-]*)?"               # optional space + numeric refs (e.g. ". 1,2")
    r"\s+"                               # required whitespace before next sentence
    r"(?=[A-Z])",                        # lookahead: next char is uppercase
)


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into individual sentences.

    Handles trailing reference markers (superscripts and numeric citations) and
    skips common abbreviations and single-letter initials.
    """
    sentences: list[str] = []
    prev = 0
    for m in _SENT_END_RE.finditer(text):
        # Find the word immediately before the terminal punctuation.
        before = text[prev : m.start()]
        word_m = re.search(r"([a-zA-Z]+)[" + _SUPERSCRIPT_DIGITS + r"0-9,\-]*$", before)
        if word_m:
            word = word_m.group(1).lower()
            if len(word) == 1 or word in _ABBREVIATIONS:
                continue  # abbreviation or initial — not a real boundary
        sent = text[prev : m.end()].rstrip()
        if sent:
            sentences.append(sent)
        prev = m.end()
    remaining = text[prev:].strip()
    if remaining:
        sentences.append(remaining)
    return sentences if sentences else [text]


# ---------------------------------------------------------------------------
# Markdown segment parsing
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MarkdownSegment:
    """One fine-grained semantic unit from the Markdown."""

    text: str
    """Segment text (Markdown syntax preserved, HTML comments stripped)."""
    page: int
    """PDF page index (0-based) inferred from surrounding ``<!--page-->`` markers."""
    md_start: int
    """Start character offset of the enclosing block in the original Markdown."""
    md_end: int
    """End character offset of the enclosing block in the original Markdown."""


_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_PAGE_MARKER_RE = re.compile(r"<!--page-->")

# Ordered and unordered list item prefixes.
_LIST_ITEM_RE = re.compile(r"^(\s{0,3}(?:[-*+]|\d+[.)]) )")
# Lines starting with superscript digits → affiliation / footnote entries.
_SUPER_PREFIX_RE = re.compile(r"^[" + _SUPERSCRIPT_DIGITS + r"]")


def _segments_from_block(
    block_text: str,
    page: int,
    md_start: int,
    md_end: int,
) -> list[MarkdownSegment]:
    """Convert a single blank-line-delimited block into fine-grained segments."""
    text = block_text.strip()
    if not text:
        return []

    def seg(t: str) -> MarkdownSegment:
        return MarkdownSegment(t.strip(), page, md_start, md_end)

    # ── Heading ──────────────────────────────────────────────────────────────
    if re.match(r"^#{1,6}\s", text):
        lines = text.splitlines()
        heading_seg = seg(lines[0])
        rest = "\n".join(lines[1:]).strip()
        if not rest:
            return [heading_seg]
        return [heading_seg] + _segments_from_block(rest, page, md_start, md_end)

    # ── Blockquote ───────────────────────────────────────────────────────────
    if text.startswith(">"):
        results: list[MarkdownSegment] = []
        for line in text.splitlines():
            line = re.sub(r"^>\s?", "", line).strip()
            if not line:
                continue
            # A line that is itself a list item (e.g. "> * item").
            if _LIST_ITEM_RE.match(line):
                results.append(seg(line))
            else:
                results.extend(seg(s) for s in _split_sentences(line))
        return results

    # ── List ─────────────────────────────────────────────────────────────────
    lines = text.splitlines()
    if _LIST_ITEM_RE.match(lines[0]):
        items: list[str] = []
        current: list[str] = []
        for line in lines:
            if _LIST_ITEM_RE.match(line):
                if current:
                    items.append(" ".join(current))
                current = [line]
            elif line.strip():
                current.append(line.strip())
        if current:
            items.append(" ".join(current))
        return [seg(item) for item in items if item.strip()]

    # ── Affiliation / footnote block (lines with superscript prefix) ─────────
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) > 1 and sum(
        1 for l in non_empty if _SUPER_PREFIX_RE.match(l.strip())
    ) >= len(non_empty) * 0.5:
        return [seg(line) for line in non_empty]

    # ── Table (GFM pipe syntax) ───────────────────────────────────────────────
    if lines and "|" in lines[0]:
        results = []
        for line in lines:
            if re.match(r"^\s*\|[-:\s|]+\|\s*$", line):
                continue  # separator row
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            results.extend(seg(c) for c in cells if c)
        return results

    # ── Regular paragraph → sentence splitting ────────────────────────────────
    return [seg(s) for s in _split_sentences(text)]


def parse_markdown_segments(markdown: str) -> list[MarkdownSegment]:
    """Parse Markdown into fine-grained segments with page hints.

    Produces one segment per heading, sentence, list item, blockquote line,
    affiliation entry, or table cell.  ``<!--page-->`` comments advance the page
    counter; all other HTML comments are stripped from segment text.
    """
    segments: list[MarkdownSegment] = []
    current_page = -1

    block_start = 0
    for m in re.finditer(r"\n{2,}|\Z", markdown, re.MULTILINE):
        block_raw = markdown[block_start : m.start()]
        md_start = block_start
        md_end = m.start()
        block_start = m.end()

        for _ in _PAGE_MARKER_RE.finditer(block_raw):
            current_page += 1

        if current_page < 0:
            continue

        text = _COMMENT_RE.sub("", block_raw).strip()
        segments.extend(_segments_from_block(text, current_page, md_start, md_end))

    return segments


# ---------------------------------------------------------------------------
# Association: markdown segment → PDF bbox
# ---------------------------------------------------------------------------

def _chars_in_range(
    chars: list[_Char],
    flat_to_char: list[int],
    flat_start: int,
    flat_end: int,
) -> list[_Char]:
    indices = set(flat_to_char[i] for i in range(flat_start, min(flat_end, len(flat_to_char))))
    return [chars[i] for i in sorted(indices)]


def _bbox_from_chars(chars: list[_Char], page_width: float, page_height: float) -> BBox:
    x0 = min(c.x0 for c in chars)
    y0 = min(c.y0 for c in chars)
    x1 = max(c.x1 for c in chars)
    y1 = max(c.y1 for c in chars)
    top = round((1.0 - y1 / page_height) * 1000)
    left = round(x0 / page_width * 1000)
    bottom = round((1.0 - y0 / page_height) * 1000)
    right = round(x1 / page_width * 1000)
    return BBox(top=top, left=left, bottom=bottom, right=right)


def _line_bboxes(
    chars: list[_Char],
    page_width: float,
    page_height: float,
) -> list[BBox]:
    """Return one BBox per line of matched chars.

    Chars are sorted top-to-bottom by y-midpoint and grouped into lines by
    y-overlap: a char joins the current line when its y-range overlaps the
    accumulated line band; otherwise it starts a new line.  This gives one
    tight box per text line regardless of how many columns the anchor spans.
    """
    if not chars:
        return []

    # Sort top-to-bottom (descending y in PDF coords where y increases upward).
    by_y = sorted(chars, key=lambda c: -(c.y0 + c.y1) / 2)

    clusters: list[list[_Char]] = [[by_y[0]]]
    band_y0 = by_y[0].y0
    band_y1 = by_y[0].y1

    for ch in by_y[1:]:
        overlap = min(ch.y1, band_y1) - max(ch.y0, band_y0)
        if overlap > 0:
            clusters[-1].append(ch)
            band_y0 = min(band_y0, ch.y0)
            band_y1 = max(band_y1, ch.y1)
        else:
            clusters.append([ch])
            band_y0 = ch.y0
            band_y1 = ch.y1

    return [_bbox_from_chars(cluster, page_width, page_height) for cluster in clusters]


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


def _infer_page(
    seg_idx: int,
    results: list[Anchor | None],
    segments: list[MarkdownSegment],
    default: int,
) -> int:
    """Infer the PDF page for an unmatched segment from its neighbours.

    Scans backwards for the last matched anchor and forwards for the first.
    If both agree on a page, that page is returned.  If they bracket the
    segment across a page boundary the ``<!--page-->``-derived *default* is
    used as a tie-breaker.
    """
    prev_page: int | None = None
    for j in range(seg_idx - 1, -1, -1):
        if results[j] is not None:
            prev_page = results[j].page
            break

    next_page: int | None = None
    for j in range(seg_idx + 1, len(results)):
        if results[j] is not None:
            next_page = results[j].page
            break

    if prev_page is not None and next_page is not None:
        if prev_page == next_page:
            return prev_page
        # Segment sits between two pages; trust the page-marker default.
        if prev_page <= default <= next_page:
            return default
        return prev_page
    if prev_page is not None:
        return max(prev_page, default)
    if next_page is not None:
        return min(next_page, default)
    return default


def _aln_to_flat_ranges(
    aln: object,
    ref_to_flat: tuple[int, ...],
) -> list[tuple[int, int]]:
    flat_ranges: list[tuple[int, int]] = []
    for frag in aln.fragments:
        if frag.fragment_type != seq_smith.FragmentType.Match:
            continue
        flat_ranges.append((
            ref_to_flat[frag.sa_start],
            ref_to_flat[frag.sa_start + frag.len],
        ))
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
    best_score = aln.score

    # Iteratively search for an earlier match with the same score.
    current_aln = aln
    while True:
        match_starts = [
            f.sa_start
            for f in current_aln.fragments
            if f.fragment_type == seq_smith.FragmentType.Match
        ]
        if not match_starts:
            break
        cutoff = min(match_starts)
        if cutoff == 0:
            break  # already at the start
        earlier_aln = seq_smith.local_align(
            reference[:cutoff], norm_seg, score_matrix, _GAP_OPEN, _GAP_EXTEND
        )
        if earlier_aln.score < best_score:
            break  # no earlier match reaches the same score
        current_aln = earlier_aln

    return best_score, _aln_to_flat_ranges(current_aln, ref_to_flat)


def associate(
    pdf_path: pathlib.Path,
    markdown: str,
    min_score: int = _MIN_SCORE,
    return_pass_info: bool = False,
) -> list[Anchor] | tuple[list[Anchor], list[int]]:
    """Align each Markdown segment to the PDF and return one Anchor per segment.

    Processes segments in Markdown order.  Each segment is aligned against the
    *residual* of its candidate page — the flat-string text not yet claimed by
    any earlier segment.  This prevents a later occurrence of a short phrase
    (e.g. a heading like "Protein-protein interactions" buried in a body
    sentence) from being stolen by the heading segment, because the body
    sentence is processed first and consumes that region.

    The candidate page is the primary page from the ``<!--page-->`` marker plus
    its immediate neighbours (±1), to tolerate off-by-one marker errors.  The
    page with the highest alignment score is chosen.

    The score threshold scales with segment length:
    ``max(5, min(min_score, len(norm_seg)))``.  This lets short segments (e.g.
    section labels) match when they are the only remaining text, while still
    requiring a minimum quality score for all matches.

    Args:
        pdf_path: Path to the PDF file.
        markdown: Cleaned Markdown with ``<!--page-->`` page-break markers.
        min_score: Score cap for the adaptive threshold.
        return_pass_info: If True, return ``(anchors, confidences)`` where
            *confidences* is a parallel list: 1 = score ≥ *min_score*
            (confident), 2 = score < *min_score* (short-segment marginal).

    Returns:
        One ``Anchor`` per successfully matched segment, in Markdown order.
        Segments that cannot be matched with sufficient confidence are omitted.
        When *return_pass_info* is True, returns ``(anchors, confidences)``.
    """
    segments = parse_markdown_segments(markdown)
    if not segments:
        return ([], []) if return_pass_info else []

    doc = pdfium.PdfDocument(pdf_path)
    num_pages = len(doc)

    page_chars: dict[int, list[_Char]] = {}
    page_char_index: dict[int, _CharIndex] = {}

    def _get_page_data(page_idx: int) -> tuple[list[_Char], _CharIndex]:
        if page_idx not in page_chars:
            page = doc[page_idx]
            chars = _extract_page_chars(page)
            ci = _build_char_index(chars)
            page_chars[page_idx] = chars
            page_char_index[page_idx] = ci
        return page_chars[page_idx], page_char_index[page_idx]

    # results[i] is the Anchor for segments[i], or None if unmatched.
    results: list[Anchor | None] = [None] * len(segments)
    # confidence[i]: 1 = score >= min_score, 2 = marginal short-segment match.
    confidence: list[int] = [0] * len(segments)
    # Consumed flat-string ranges per page (raw; merged on demand).
    page_matched_ranges: dict[int, list[tuple[int, int]]] = {}

    def _chars_from_flat_ranges(
        chars: list[_Char],
        flat_to_char: list[int],
        flat_ranges: list[tuple[int, int]],
    ) -> list[_Char]:
        indices: set[int] = set()
        for fs, fe in flat_ranges:
            indices.update(
                flat_to_char[j]
                for j in range(fs, min(fe, len(flat_to_char)))
            )
        return [chars[i] for i in sorted(indices)]

    def _try_page_residual(
        page_idx: int,
        seg: MarkdownSegment,
        threshold: int,
    ) -> tuple[int, list[tuple[int, int]]] | None:
        """Align *seg* against the residual of *page_idx*.

        Returns ``(score, flat_ranges)`` on success, else ``None``.
        """
        if page_idx < 0 or page_idx >= num_pages:
            return None
        chars, ci = _get_page_data(page_idx)
        if not chars:
            return None

        covered = _merge_ranges(page_matched_ranges.get(page_idx, []))
        residual, pos_map = _residual_string(ci.flat_str, covered)
        if not residual:
            return None

        def _align(norm_fn, score_matrix):
            res_norm, res_to_res = norm_fn(residual)
            seg_norm, _ = norm_fn(seg.text)
            if not seg_norm:
                return None
            hit = _align_against(res_norm, res_to_res, seg_norm, score_matrix, threshold)
            if hit is None:
                return None
            flat_ranges = [
                (pos_map[rs], pos_map[min(re, len(pos_map) - 1)])
                for rs, re in hit[1]
            ]
            return hit[0], flat_ranges

        result = _align(_normalize_strict, _SCORE_MATRIX_STRICT)
        if result is None:
            result = _align(_normalize_loose, _SCORE_MATRIX_LOOSE)
        return result

    # ── Single sequential pass ────────────────────────────────────────────────
    # Segments are processed in Markdown order.  Each successful match consumes
    # flat-string ranges so subsequent segments see only the remaining text.
    for i, seg in enumerate(segments):
        if seg.page >= num_pages:
            continue

        # Adaptive threshold: cap at min_score for long segments; scale down
        # for short ones (but floor at 5 to avoid noise).
        norm_len = len(_normalize_strict(seg.text)[0]) or len(_normalize_loose(seg.text)[0])
        threshold = max(5, min(min_score, norm_len))

        # Try primary page and ±1; take highest score.
        best: tuple[int, list[tuple[int, int]], int] | None = None
        for offset in (0, -1, +1):
            candidate = _try_page_residual(seg.page + offset, seg, threshold)
            if candidate is not None:
                score, flat_ranges = candidate
                if best is None or score > best[0]:
                    best = (score, flat_ranges, seg.page + offset)

        if best is None:
            continue

        score, flat_ranges, matched_page = best
        chars, ci = _get_page_data(matched_page)
        matched_chars = _chars_from_flat_ranges(chars, ci.flat_to_char, flat_ranges)
        if not matched_chars:
            continue

        page_obj = doc[matched_page]
        boxes = tuple(_line_bboxes(matched_chars, page_obj.get_width(), page_obj.get_height()))
        if boxes:
            results[i] = Anchor(text=seg.text, page=matched_page, boxes=boxes)
            confidence[i] = 1 if score >= min_score else 2
            page_matched_ranges.setdefault(matched_page, []).extend(flat_ranges)

    anchors = [a for a, c in zip(results, confidence) if a is not None]
    if return_pass_info:
        passes = [c for a, c in zip(results, confidence) if a is not None]
        return anchors, passes
    return anchors
