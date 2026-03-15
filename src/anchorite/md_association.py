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
import unicodedata
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
            cp = pdfium_c.FPDFText_GetUnicode(textpage, char_index)
            # PDFium counts non-BMP characters (e.g. Mathematical Italic symbols,
            # U+1D400–U+1D7FF) as two UTF-16 surrogate-pair indices.  Detect a
            # high surrogate and reassemble the full code point from the pair.
            if 0xD800 <= cp <= 0xDBFF:
                if char_index + 1 < total_chars:
                    cp_low = pdfium_c.FPDFText_GetUnicode(textpage, char_index + 1)
                    if 0xDC00 <= cp_low <= 0xDFFF:
                        cp = 0x10000 + (cp - 0xD800) * 0x400 + (cp_low - 0xDC00)
                        ci_for_box = char_index
                        char_index += 2  # consume both surrogate indices
                        obj_pos += 1     # but only one code point in obj_text
                    else:
                        char_index += 1
                        obj_pos += 1
                        continue
                else:
                    char_index += 1
                    obj_pos += 1
                    continue
            elif 0xDC00 <= cp <= 0xDFFF:
                # Orphaned low surrogate — should not occur; skip.
                char_index += 1
                obj_pos += 1
                continue
            else:
                ci_for_box = char_index
                char_index += 1

            text = chr(cp)
            if text in ("\r", "\n"):
                # Line-break markers inserted by PDFium are absent from obj_text.
                continue  # char_index already advanced; do NOT advance obj_pos
            obj_pos += 1

            if not text.isspace():
                normalized = _CHAR_NORM.get(text, text)
                # Map Mathematical Alphanumeric Symbols and other compatibility
                # characters to ASCII equivalents (e.g. 𝑆𝑒𝑛𝑠𝑖𝑡𝑖𝑣𝑖𝑡𝑦 → Sensitivity).
                normalized = unicodedata.normalize("NFKC", normalized)
                if normalized:
                    left, bottom, right, top = textpage.get_charbox(ci_for_box, loose=False)
                    if right > left and top > bottom:
                        chars.append(_Char(normalized, left, bottom, right, top, font_size))

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
# How many PDF pages above max(page_lo, seg.page) to include in the pass-1 search window.
_PAGE_SLACK = 5
# Pass-1 uniqueness: best-page score must be >= this multiple of the second-best.
_UNIQUENESS_RATIO = 2.0


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
    # Reject weak partial hits: require at least half the segment to be covered.
    # This catches cases like matching only "conflicting" from "Conflicting
    # interpretations" when the heading doesn't appear in the PDF.
    seg_covered = sum(
        f.len for f in aln.fragments
        if f.fragment_type == seq_smith.FragmentType.Match
    )
    if seg_covered * 2 < len(norm_seg):
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

    Additionally, the alignment must cover at least half of the normalised
    segment characters.  This rejects weak partial hits (e.g. matching only
    "conflicting" from a heading "Conflicting interpretations" that does not
    appear in the PDF) even when the score threshold is met.

    Args:
        pdf_path: Path to the PDF file.
        markdown: Cleaned Markdown with ``<!--page-->`` page-break markers.
        min_score: Score cap for the adaptive threshold.
        return_pass_info: If True, return ``(anchors, confidences)`` where
            *confidences* is a parallel list of integers (currently always 1).

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

    def _accept_match(
        seg: MarkdownSegment,
        i: int,
        score: int,
        flat_ranges: list,
        matched_page: int,
        conf: int,
    ) -> None:
        chars, ci = _get_page_data(matched_page)
        matched_chars = _chars_from_flat_ranges(chars, ci.flat_to_char, flat_ranges)
        if not matched_chars:
            return
        page_obj = doc[matched_page]
        boxes = tuple(_line_bboxes(matched_chars, page_obj.get_width(), page_obj.get_height()))
        if boxes:
            results[i] = Anchor(text=seg.text, page=matched_page, boxes=boxes)
            confidence[i] = conf
            page_matched_ranges.setdefault(matched_page, []).extend(flat_ranges)

    def _best_across_pages(
        seg: MarkdownSegment,
        page_range: range,
        threshold: int,
    ) -> list[tuple[int, list, int]]:
        """Return all above-threshold matches sorted best-first."""
        hits: list[tuple[int, list, int]] = []
        for page in page_range:
            candidate = _try_page_residual(page, seg, threshold)
            if candidate is not None:
                hits.append((candidate[0], candidate[1], page))
        hits.sort(key=lambda x: x[0], reverse=True)
        return hits

    # ── Pass 1: unique matches ────────────────────────────────────────────────
    # Search a symmetric window of _PAGE_SLACK pages around the page marker.
    # A match is accepted only when it is *unique*: the best-scoring page
    # beats all others by _UNIQUENESS_RATIO.  Ambiguous segments (e.g.
    # per-page running headers) are deferred to pass 2.
    # No global ordering state is maintained here — the uniqueness check
    # prevents wrong matches, so early errors cannot cascade.

    for i, seg in enumerate(segments):
        if seg.page >= num_pages:
            continue

        norm_len = len(_normalize_strict(seg.text)[0]) or len(_normalize_loose(seg.text)[0])
        threshold = max(5, min(min_score, norm_len))

        p1_lo = max(0, seg.page - _PAGE_SLACK)
        p1_hi = min(num_pages - 1, seg.page + _PAGE_SLACK)
        hits = _best_across_pages(seg, range(p1_lo, p1_hi + 1), threshold)

        if not hits:
            continue

        # Uniqueness check: best score must exceed second-best by _UNIQUENESS_RATIO.
        if len(hits) >= 2 and hits[1][0] * _UNIQUENESS_RATIO > hits[0][0]:
            continue  # ambiguous — defer to pass 2

        score, flat_ranges, matched_page = hits[0]
        _accept_match(seg, i, score, flat_ranges, matched_page, 1 if score >= min_score else 2)

    # ── Pass 2: fill in deferred segments using neighbour-inferred pages ──────
    # Segments skipped in pass 1 (ambiguous or search-window miss) are retried
    # here.  For each unmatched segment, _infer_page() estimates its PDF page
    # from the nearest pass-1 anchors on either side, narrowing the search to
    # [inferred - 1, inferred + 1].  No uniqueness requirement.
    for i, seg in enumerate(segments):
        if results[i] is not None:
            continue
        if seg.page >= num_pages:
            continue

        norm_len = len(_normalize_strict(seg.text)[0]) or len(_normalize_loose(seg.text)[0])
        threshold = max(5, min(min_score, norm_len))

        # Search the full interval between the nearest pass-1 anchors on either
        # side — this is what the document-order constraint actually tells us.
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

        hits = _best_across_pages(seg, range(p2_lo, p2_hi + 1), threshold)
        if not hits:
            continue

        score, flat_ranges, matched_page = hits[0]
        _accept_match(seg, i, score, flat_ranges, matched_page, 1 if score >= min_score else 2)

    anchors = [a for a, c in zip(results, confidence) if a is not None]
    if return_pass_info:
        passes = [c for a, c in zip(results, confidence) if a is not None]
        return anchors, passes
    return anchors
