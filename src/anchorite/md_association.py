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
import math
import pathlib
import re
import string
import unicodedata
from collections.abc import Callable
from typing import Literal, NamedTuple, overload

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import seq_smith

from .anchors import Anchor, BBox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character extraction
# ---------------------------------------------------------------------------

_CHAR_NORM: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201a": ",",
    "\u2013": "-",
    "\u2014": "--",
    "\u2212": "-",
    "\u2010": "-",
    "\u2011": "-",
    "\u00ad": "",
    "\u00a0": " ",
    "\ufffe": "",
}


@dataclasses.dataclass(frozen=True)
class _Char:
    text: str
    x0: float
    y0: float  # bottom in PDF coords (pts, origin bottom-left)
    x1: float
    y1: float  # top in PDF coords
    font_size: float


def _extract_page_chars(page: pdfium.PdfPage) -> list[_Char]:  # noqa: C901, PLR0912
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
        font_size = obj.get_font_size() * math.sqrt(m.a**2 + m.b**2)

        obj_pos = 0
        while obj_pos < len(obj_text) and char_index < total_chars:
            cp = pdfium_c.FPDFText_GetUnicode(textpage, char_index)
            # PDFium counts non-BMP characters (e.g. Mathematical Italic symbols,
            # U+1D400–U+1D7FF) as two UTF-16 surrogate-pair indices.  Detect a
            # high surrogate and reassemble the full code point from the pair.
            if _HIGH_SURROGATE_LO <= cp <= _HIGH_SURROGATE_HI:
                if char_index + 1 < total_chars:
                    cp_low = pdfium_c.FPDFText_GetUnicode(textpage, char_index + 1)
                    if _LOW_SURROGATE_LO <= cp_low <= _LOW_SURROGATE_HI:
                        cp = _NON_BMP_BASE + (cp - _HIGH_SURROGATE_LO) * 0x400 + (cp_low - _LOW_SURROGATE_LO)
                        ci_for_box = char_index
                        char_index += 2  # consume both surrogate indices
                        obj_pos += 1  # but only one code point in obj_text
                    else:
                        char_index += 1
                        obj_pos += 1
                        continue
                else:
                    char_index += 1
                    obj_pos += 1
                    continue
            elif _LOW_SURROGATE_LO <= cp <= _LOW_SURROGATE_HI:
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
    """Build a flat string and a per-character index mapping back to chars.

    Inserts a space between successive chars when either:

    * the horizontal gap to the next char exceeds 20 % of font size (an
      intra-line word break), or
    * the next char drops to a different visual line — its baseline is
      shifted vertically by more than 50 % of font size, or sits to the
      *left* of the current char.  Without this, end-of-line + start-of-
      next-line concatenates ("``we``" + "``identified``" → "``weidentified``")
      because PDFium's coordinate stream emits no whitespace at line
      breaks, and the alignment string drifts out of sync with the
      Markdown.

    End-of-line soft hyphens are reconnected: when the line-break-trailing
    char is ``-`` between two alphabetic glyphs, both the hyphen and the
    inserted space are dropped, so the typeset ``induc-`` + ``tion``
    reconnects to ``induction`` (matching the Markdown's un-hyphenated
    form).  Numeric ranges like ``2009-`` + ``2010`` keep the hyphen
    because the surrounding glyphs aren't alphabetic.  Dash variants
    (en-dash, em-dash, hyphen-minus) have already been normalised to
    ``-`` during char extraction, so a single literal check suffices.
    """
    parts: list[str] = []
    flat_to_char: list[int] = []

    for i, ch in enumerate(chars):
        for c in ch.text:
            parts.append(c)
            flat_to_char.append(i)
        if i + 1 < len(chars):
            nxt = chars[i + 1]
            x_gap = nxt.x0 - ch.x1
            y_drop = abs(nxt.y0 - ch.y0)
            line_break = nxt.x0 < ch.x0 or y_drop > ch.font_size * 0.5
            if (
                line_break
                and len(parts) >= 2  # noqa: PLR2004
                and parts[-1] == "-"
                and parts[-2].isalpha()
                and nxt.text
                and nxt.text[0].isalpha()
            ):
                # Soft hyphen between two letters at a line break: drop
                # the hyphen and insert no space.  The next iteration
                # appends the next char's letters directly.
                parts.pop()
                flat_to_char.pop()
                continue
            if x_gap > ch.font_size * 0.2 or line_break:
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

_NormFn = Callable[..., tuple[bytes, tuple[int, ...]]]

# UTF-16 surrogate ranges for non-BMP code points returned by PDFium.
_HIGH_SURROGATE_LO, _HIGH_SURROGATE_HI = 0xD800, 0xDBFF
_LOW_SURROGATE_LO, _LOW_SURROGATE_HI = 0xDC00, 0xDFFF
_NON_BMP_BASE = 0x10000
# Phase 1 (conservative HSP-based): pages to search around the page marker.
_PHASE1_PAGE_SLACK = 10
# Phase 1: best score must be >= this multiple of the second-best (cross-page AND within-page).
_PHASE1_UNIQUENESS_RATIO = 2.0
# Phase 1: fraction of the normalised segment that must be covered by the best HSP.
_PHASE1_MIN_COVERAGE = 0.9
# Phase 1: segments with fewer alphanum chars than this are skipped (too short to be unique).
_PHASE1_MIN_LEN = 10


# HTML tags that survive into Markdown (e.g. ``<sup>`` from JATS-derived
# articles, ``<a id="...">`` anchors) must not contribute alphanum bytes to
# the alignment string — otherwise ``<sup>1</sup>`` becomes the letters
# ``sup1sup`` and the matching PDF text ``1`` aligns nowhere near it.  Both
# normalisers detect tag spans in advance and skip past them, leaving the
# index map pointing at the original text offsets.
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Inline Markdown links ``[text](url)``.  Renders as just ``text`` in the PDF,
# but a naïve normalisation contributes both the visible text *and* the URL
# target — so an autolink ``[https://x.org](https://x.org/path)`` doubles its
# alphanum footprint.  When the segment is significantly longer than its PDF
# counterpart the alignment's coverage gates reject the match.  We treat the
# wrapper (``[`` and ``](url)``) as zero-width, leaving the inner link text
# to align like ordinary prose.
#
# The regex is deliberately conservative: link text and URL must each be
# single-line and contain no nested ``]`` / ``)``.  Edge cases (URLs with
# balanced parens, nested brackets) fall through to the existing behaviour.
_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\n]*)\)")


def _strip_spans(text: str) -> list[tuple[int, int]]:
    """Return sorted, merged character spans whose content is zero-width for
    alignment: HTML tags and the wrapper portions of inline Markdown links.
    """
    spans: list[tuple[int, int]] = [(m.start(), m.end()) for m in _HTML_TAG_RE.finditer(text)]
    for m in _MD_LINK_RE.finditer(text):
        spans.append((m.start(), m.start(1)))  # leading '['
        spans.append((m.end(1), m.end()))  # trailing '](url)'
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and merged[-1][1] >= s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


_ASCII_ALNUM = frozenset(string.ascii_lowercase + string.digits)


def _nfkd_alnum(c: str) -> str:
    """Return the lowercase-ASCII-alphanum letters NFKD-decomposed from *c*.

    NFKD applies compatibility decompositions, which lets us recover the base
    letter from accented characters (``ö`` → ``o`` + combining diaeresis),
    expand ligatures (``ﬁ`` → ``fi``), turn superscript and subscript digits
    into plain digits (``²`` → ``2``), and map Mathematical Alphanumeric
    Symbols and other compatibility characters to their ASCII equivalents.
    Combining marks and other non-ASCII components of the decomposition are
    discarded.  The empty string is returned when nothing alphanumeric remains.
    """
    out = []
    for d in unicodedata.normalize("NFKD", c):
        ld = d.lower()
        if ld in _ASCII_ALNUM:
            out.append(ld)
    return "".join(out)


def _normalize_strict(text: str, *, strip_html: bool = False) -> tuple[bytes, tuple[int, ...]]:
    """Lowercase + collapse non-alphanumeric runs to a single space.

    Each input character is NFKD-decomposed before classification (see
    ``_nfkd_alnum``), so accented letters and ligatures contribute their
    base letters rather than dropping out as non-ASCII.

    Combining marks (Unicode general category ``M*``) are zero-width: they
    don't emit alphanum bytes and don't trigger the punctuation-collapses-to-
    space branch.  This keeps decomposed input (``o`` + ``U+0308``)
    indistinguishable from precomposed input (``ö``) in the alignment string.

    Args:
        text: The text to normalise.
        strip_html: When True, ``<...>``-style tags are treated as zero-width
            (their tag-name letters don't contribute alignment bytes).  Only
            safe for Markdown input — pdfium-extracted PDF text contains
            literal ``<`` / ``>`` characters when those glyphs appear in the
            document (e.g. ``p < 0.05``), and stripping them silently drops
            real content.  Defaults to False (PDF-safe).
    """
    skip_spans = _strip_spans(text) if strip_html else []
    normalized: list[str] = []
    idx_map: list[int] = []
    span_iter = iter(skip_spans)
    next_span = next(span_iter, None)
    i = 0
    while i < len(text):
        if next_span is not None and i == next_span[0]:
            i = next_span[1]
            next_span = next(span_iter, None)
            continue
        c = text[i]
        emitted = _nfkd_alnum(c)
        if emitted:
            for d in emitted:
                normalized.append(d)
                idx_map.append(i)
        elif unicodedata.category(c).startswith("M"):
            pass  # combining mark — zero-width, neither letter nor separator
        elif normalized and normalized[-1] != " ":
            normalized.append(" ")
            idx_map.append(i)
        i += 1
    idx_map.append(len(text))
    return seq_smith.encode("".join(normalized), _ALIGN_ALPHABET_STRICT), tuple(idx_map)


def _normalize_loose(text: str, *, strip_html: bool = False) -> tuple[bytes, tuple[int, ...]]:
    """Keep only lowercase letters and digits; strip everything else.

    Used as a fallback for segments that fail the strict pass.  Discarding
    spaces means that letter-spaced display headings like
    ``C A S E  R E P O R T`` normalise to the same sequence as
    ``CASE REPORT``, at the cost of losing word-boundary information.

    Each input character is NFKD-decomposed before classification (see
    ``_nfkd_alnum``).

    See ``_normalize_strict`` for the meaning of ``strip_html``.
    """
    skip_spans = _strip_spans(text) if strip_html else []
    normalized: list[str] = []
    idx_map: list[int] = []
    span_iter = iter(skip_spans)
    next_span = next(span_iter, None)
    i = 0
    while i < len(text):
        if next_span is not None and i == next_span[0]:
            i = next_span[1]
            next_span = next(span_iter, None)
            continue
        for d in _nfkd_alnum(text[i]):
            normalized.append(d)
            idx_map.append(i)
        i += 1
    idx_map.append(len(text))
    return seq_smith.encode("".join(normalized), _ALIGN_ALPHABET_LOOSE), tuple(idx_map)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"

_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "al",
        "fig",
        "figs",
        "eq",
        "eqs",
        "vs",
        "etc",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "inc",
        "ltd",
        "co",
        "jr",
        "sr",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "vol",
        "no",
        "pp",
        "p",
        "ed",
        "eds",
        "ref",
        "refs",
        "approx",
        "dept",
        "est",
        "max",
        "min",
        "cf",
        "viz",
    },
)

# Sentence boundary: terminal punctuation, optional reference markers
# (superscripts or a space-separated digit run), then whitespace, then uppercase.
_SENT_END_RE = re.compile(
    r"[.!?]"
    r"[" + _SUPERSCRIPT_DIGITS + r"]*"  # optional superscript refs directly after punct
    r"(?:\s+\d[\d,\-]*)?"  # optional space + numeric refs (e.g. ". 1,2")
    r"\s+"  # required whitespace before next sentence
    r"(?=[A-Z])",  # lookahead: next char is uppercase
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
    page: int | None
    """PDF page index (0-based) inferred from surrounding ``<!--page-->`` markers,
    or ``None`` when the source Markdown carries no page markers (in which case
    the page is determined by the alignment itself)."""
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


def _segments_from_block(  # noqa: C901, PLR0911, PLR0912
    block_text: str,
    page: int | None,
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
        return [heading_seg, *_segments_from_block(rest, page, md_start, md_end)]

    # ── Blockquote ───────────────────────────────────────────────────────────
    if text.startswith(">"):
        results: list[MarkdownSegment] = []
        for raw_line in text.splitlines():
            line = re.sub(r"^>\s?", "", raw_line).strip()
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
    non_empty = [ln for ln in lines if ln.strip()]
    if len(non_empty) > 1 and sum(1 for ln in non_empty if _SUPER_PREFIX_RE.match(ln.strip())) >= len(non_empty) * 0.5:
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
    """Parse Markdown into fine-grained segments, optionally with page hints.

    Produces one segment per heading, sentence, list item, blockquote line,
    affiliation entry, or table cell.  All HTML comments are stripped from
    segment text.

    Page-hint behaviour:

    * If the Markdown contains one or more ``<!--page-->`` markers, the markers
      seed each segment's ``page`` field; content preceding the first marker
      is dropped (it isn't pinned to any page).  This is the typical input
      shape produced by chunked OCR pipelines, where a marker is emitted at
      every page break.
    * If the Markdown contains no markers at all (e.g. JATS XML rendered to
      Markdown, where page boundaries aren't carried by the source), every
      segment's ``page`` is ``None`` and the page is determined later by the
      alignment.
    """
    # Ensure every <!--page--> marker sits in its own blank-line-delimited block.
    # Without this, a marker that immediately follows a paragraph (no blank line)
    # ends up in the same block as that paragraph; after comment-stripping the
    # subsequent content (tables, etc.) is concatenated onto the last sentence.
    markdown = re.sub(r"(?<!\n\n)(<!--page-->)", r"\n\n\1", markdown)
    markdown = re.sub(r"(<!--page-->)(?!\n)", r"\1\n\n", markdown)

    has_markers = _PAGE_MARKER_RE.search(markdown) is not None

    segments: list[MarkdownSegment] = []
    # marker_page tracks the page counter when the source uses markers.  In
    # the no-marker case it's irrelevant — every segment carries ``page=None``.
    marker_page = -1

    block_start = 0
    for m in re.finditer(r"\n{2,}|\Z", markdown, re.MULTILINE):
        block_raw = markdown[block_start : m.start()]
        md_start = block_start
        md_end = m.start()
        block_start = m.end()

        for _ in _PAGE_MARKER_RE.finditer(block_raw):
            marker_page += 1

        page: int | None
        if has_markers:
            if marker_page < 0:
                continue  # Pre-marker content is unpinned; drop it.
            page = marker_page
        else:
            page = None

        text = _COMMENT_RE.sub("", block_raw).strip()
        segments.extend(_segments_from_block(text, page, md_start, md_end))

    return segments


# ---------------------------------------------------------------------------
# Association: markdown segment → PDF bbox
# ---------------------------------------------------------------------------


def _bbox_from_chars(
    chars: list[_Char],
    page_width: float,
    page_height: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> BBox:
    # Subtract the page's mediabox origin so coordinates are page-relative.
    # PDFs whose mediabox is offset from (0, 0) would otherwise produce
    # bboxes shifted by that offset.
    x0 = min(c.x0 for c in chars) - origin_x
    y0 = min(c.y0 for c in chars) - origin_y
    x1 = max(c.x1 for c in chars) - origin_x
    y1 = max(c.y1 for c in chars) - origin_y
    top = round((1.0 - y1 / page_height) * 1000)
    left = round(x0 / page_width * 1000)
    bottom = round((1.0 - y0 / page_height) * 1000)
    right = round(x1 / page_width * 1000)
    return BBox(top=top, left=left, bottom=bottom, right=right)


def _line_bboxes(
    chars: list[_Char],
    page_width: float,
    page_height: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
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

    return [_bbox_from_chars(cluster, page_width, page_height, origin_x, origin_y) for cluster in clusters]


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


@dataclasses.dataclass
class _PageData:
    """Per-page PDF data needed for alignment and bbox derivation.

    Single source of truth for "PDF page → cached char data" across
    ``associate()`` and ``pdf_index.PdfIndex``.  Both call
    ``_extract_page_data`` to populate; alignment never re-reads the PDF.
    """

    chars: list[_Char]
    """Per-character extraction output for this page."""
    char_index: _CharIndex
    """Flat string + per-position char index for this page."""
    width: float
    """Page width in PDF points."""
    height: float
    """Page height in PDF points."""
    origin_x: float
    """Mediabox left origin in PDF points."""
    origin_y: float
    """Mediabox bottom origin in PDF points."""


def _extract_page_data(pdf: pdfium.PdfDocument) -> list[_PageData]:
    """Extract per-page chars, dimensions, and origin from an open PDF document.

    Eager: every page is read up front.  Both ``associate()`` and
    ``pdf_index.PdfIndex`` need most or all pages anyway, and a single read
    pass keeps both consumers off divergent PDFium code paths.
    """
    page_data: list[_PageData] = []
    for page_idx in range(len(pdf)):
        page = pdf[page_idx]
        chars = _extract_page_chars(page)
        ci = _build_char_index(chars)
        mb = page.get_mediabox()
        page_data.append(
            _PageData(
                chars=chars,
                char_index=ci,
                width=page.get_width(),
                height=page.get_height(),
                origin_x=mb[0],
                origin_y=mb[1],
            ),
        )
    return page_data


@dataclasses.dataclass(frozen=True)
class _AlignmentOutcome:
    """Per-segment alignment result, parallel to ``parse_markdown_segments(md)``.

    ``anchors`` and ``passes`` carry the public ``associate()`` outputs;
    ``matched_chars_per_segment`` is the extra detail ``pdf_index.PdfIndex``
    needs to rebuild its flat string in markdown order using only the chars
    each segment actually claimed.
    """

    anchors: list[Anchor]
    """Matched anchors in markdown order (unmatched segments omitted)."""
    passes: list[int]
    """Parallel to ``anchors``: 1 = phase 1, 2 = phase 2."""
    matched_chars_per_segment: list[tuple[int, list[int]] | None]
    """Per *parsed* segment in markdown order (length =
    ``len(parse_markdown_segments(markdown))``): ``(page_idx, sorted_char_indices)``
    when matched, or ``None``.  ``char_indices`` are sorted indices into
    ``page_data[page_idx].chars``.
    """


def _align_markdown_to_pages(  # noqa: C901, PLR0912, PLR0915
    page_data: list[_PageData],
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
        return _AlignmentOutcome(anchors=[], passes=[], matched_chars_per_segment=[])

    num_pages = len(page_data)

    # results[i]: Anchor for segments[i], or None if unmatched.
    results: list[Anchor | None] = [None] * len(segments)
    # confidence[i]: 1 = phase 1 (conservative), 2 = phase 2 (page-constrained).
    confidence: list[int] = [0] * len(segments)
    # Consumed flat-string ranges per page (raw; merged on demand).
    page_matched_ranges: dict[int, list[tuple[int, int]]] = {}
    # Per-segment matched (page_idx, sorted_char_indices) for downstream
    # consumers (PdfIndex cleanup); None for unmatched segments.
    matched_chars_per_segment: list[tuple[int, list[int]] | None] = [None] * len(segments)

    def _chars_from_flat_ranges(
        flat_to_char: list[int],
        flat_ranges: list[tuple[int, int]],
    ) -> list[int]:
        indices: set[int] = set()
        for fs, fe in flat_ranges:
            indices.update(flat_to_char[j] for j in range(fs, min(fe, len(flat_to_char))))
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
        if not pd.chars:
            return None

        covered = _merge_ranges(page_matched_ranges.get(page_idx, []))
        residual, pos_map = _residual_string(pd.char_index.flat_str, covered)
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

        result = _align(_normalize_strict, _SCORE_MATRIX_STRICT)
        if result is None:
            result = _align(_normalize_loose, _SCORE_MATRIX_LOOSE)
        return result

    def _accept_match(
        seg: MarkdownSegment,
        i: int,
        flat_ranges: list,
        matched_page: int,
        conf: int,
    ) -> None:
        pd = page_data[matched_page]
        char_indices = _chars_from_flat_ranges(pd.char_index.flat_to_char, flat_ranges)
        if not char_indices:
            return
        matched_chars = [pd.chars[j] for j in char_indices]
        boxes = tuple(
            _line_bboxes(
                matched_chars,
                pd.width,
                pd.height,
                pd.origin_x,
                pd.origin_y,
            ),
        )
        if boxes:
            results[i] = Anchor(text=seg.text, page=matched_page, boxes=boxes)
            confidence[i] = conf
            matched_chars_per_segment[i] = (matched_page, char_indices)
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
            ci = page_data[page_idx].char_index
            norm_bytes, _ = _normalize_loose(ci.flat_str)  # PDF: don't strip HTML
            page_alphanum_bytes[page_idx] = norm_bytes
        return page_alphanum_bytes[page_idx]

    # seg_idx → PDF page index assigned by phase 1.
    phase1_page: dict[int, int] = {}

    for i, seg in enumerate(segments):
        if seg.page is not None and seg.page >= num_pages:
            continue

        norm_seg, _ = _normalize_loose(seg.text, strip_html=True)
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
            _SCORE_MATRIX_LOOSE,
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
        norm_len = len(_normalize_strict(seg.text, strip_html=True)[0]) or len(
            _normalize_loose(seg.text, strip_html=True)[0],
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

        norm_len = len(_normalize_strict(seg.text, strip_html=True)[0]) or len(
            _normalize_loose(seg.text, strip_html=True)[0],
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
    return _AlignmentOutcome(
        anchors=anchors,
        passes=passes,
        matched_chars_per_segment=matched_chars_per_segment,
    )


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
    page_data = _extract_page_data(doc)
    outcome = _align_markdown_to_pages(page_data, markdown, min_score)
    if return_pass_info:
        return outcome.anchors, outcome.passes
    return outcome.anchors
