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

_ALIGN_ALPHABET = string.ascii_lowercase + string.digits + " "
_SCORE_MATRIX = seq_smith.make_score_matrix(_ALIGN_ALPHABET, +1, -1)
_GAP_OPEN, _GAP_EXTEND = -2, -2
_MIN_SCORE = 10


def _normalize(text: str) -> tuple[bytes, tuple[int, ...]]:
    """Lowercase + collapse non-alphanumeric to single spaces."""
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
    return seq_smith.encode("".join(normalized), _ALIGN_ALPHABET), tuple(idx_map)


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
        return [seg(text)]

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


def associate(
    pdf_path: pathlib.Path,
    markdown: str,
    min_score: int = _MIN_SCORE,
) -> list[Anchor]:
    """Align each Markdown segment to the PDF and return one Anchor per segment.

    Args:
        pdf_path: Path to the PDF file.
        markdown: Cleaned Markdown with ``<!--page-->`` page-break markers.
        min_score: Minimum Smith-Waterman score to accept a match.

    Returns:
        One ``Anchor`` per successfully matched segment, in Markdown order.
        Segments that cannot be matched with sufficient confidence are omitted.
    """
    segments = parse_markdown_segments(markdown)
    if not segments:
        return []

    doc = pdfium.PdfDocument(pdf_path)
    num_pages = len(doc)

    page_chars: dict[int, list[_Char]] = {}
    page_char_index: dict[int, _CharIndex] = {}
    page_norm: dict[int, tuple[bytes, tuple[int, ...]]] = {}

    def _get_page_data(page_idx: int) -> tuple[list[_Char], _CharIndex, bytes, tuple[int, ...]]:
        if page_idx not in page_chars:
            page = doc[page_idx]
            chars = _extract_page_chars(page)
            ci = _build_char_index(chars)
            norm_bytes, norm_to_flat = _normalize(ci.flat_str)
            page_chars[page_idx] = chars
            page_char_index[page_idx] = ci
            page_norm[page_idx] = (norm_bytes, norm_to_flat)
        return (
            page_chars[page_idx],
            page_char_index[page_idx],
            *page_norm[page_idx],
        )

    anchors: list[Anchor] = []

    for seg in segments:
        if seg.page >= num_pages:
            continue

        chars, ci, norm_page, norm_to_flat = _get_page_data(seg.page)
        if not chars:
            continue

        norm_seg, _ = _normalize(seg.text)
        if not norm_seg:
            continue

        alignment = seq_smith.local_align(
            norm_page, norm_seg, _SCORE_MATRIX, _GAP_OPEN, _GAP_EXTEND
        )
        if alignment.score < min_score:
            continue

        matched_chars: list[_Char] = []
        for frag in alignment.fragments:
            if frag.fragment_type != seq_smith.FragmentType.Match:
                continue
            flat_start = norm_to_flat[frag.sa_start]
            flat_end = norm_to_flat[frag.sa_start + frag.len]
            matched_chars.extend(_chars_in_range(chars, ci.flat_to_char, flat_start, flat_end))

        if not matched_chars:
            continue

        page = doc[seg.page]
        boxes = tuple(_line_bboxes(matched_chars, page.get_width(), page.get_height()))
        if boxes:
            anchors.append(Anchor(text=seg.text, page=seg.page, boxes=boxes))

    return anchors
