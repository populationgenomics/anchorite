"""Shared text normalisation for fuzzy alignment.

Exposes the strict and loose normalisers used across the package — the
bbox-generation side (``md_association``, ``bbox_alignment``) and the
quote-resolution side (``resolve``, ``resolve_quote``, ``is_quote_grounded``, ``locate_quote_span``,
``PdfIndex.resolve``) all funnel through the same code, so a quote
produced from a piece of Markdown is guaranteed to align against the
same Markdown its bboxes were derived from.

Each normaliser returns ``(normalized_bytes, idx_map)``:

* ``normalized_bytes`` is encoded against the appropriate
  ``ALIGN_ALPHABET`` for direct use with ``seq_smith``.
* ``idx_map`` has length ``len(normalized_bytes) + 1``: entry ``i`` is
  the source-text offset of the character that contributed
  ``normalized_bytes[i]``, plus a sentinel ``len(text)`` at the end so
  exclusive-end indices look up safely.

Each input character is NFKD-decomposed before classification, so
accented letters (``ö`` → ``o`` + combining diaeresis), ligatures
(``ﬁ`` → ``fi``), superscript/subscript digits (``²`` → ``2``), and
Mathematical Alphanumeric Symbols collapse to their ASCII bases.
Combining marks (Unicode general category ``M*``) are zero-width: they
emit no alphanum byte and don't trigger the punctuation-collapses-to-
space branch, so decomposed input is indistinguishable from precomposed
input in the alignment string.
"""

import re
import string
import unicodedata

import seq_smith

ALIGN_ALPHABET_STRICT = string.ascii_lowercase + string.digits + " "
SCORE_MATRIX_STRICT = seq_smith.make_score_matrix(ALIGN_ALPHABET_STRICT, +1, -1)
ALIGN_ALPHABET_LOOSE = string.ascii_lowercase + string.digits
SCORE_MATRIX_LOOSE = seq_smith.make_score_matrix(ALIGN_ALPHABET_LOOSE, +1, -1)


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


def strip_spans(text: str) -> list[tuple[int, int]]:
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


def nfkd_alnum(c: str) -> str:
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


def normalize_strict(text: str, *, strip_html: bool = False) -> tuple[bytes, tuple[int, ...]]:
    """Lowercase + collapse non-alphanumeric runs to a single space.

    Each input character is NFKD-decomposed before classification (see
    ``nfkd_alnum``), so accented letters and ligatures contribute their
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
    skip_spans = strip_spans(text) if strip_html else []
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
        emitted = nfkd_alnum(c)
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
    return seq_smith.encode("".join(normalized), ALIGN_ALPHABET_STRICT), tuple(idx_map)


def normalize_loose(text: str, *, strip_html: bool = False) -> tuple[bytes, tuple[int, ...]]:
    """Keep only lowercase letters and digits; strip everything else.

    Used as a fallback for segments that fail the strict pass.  Discarding
    spaces means that letter-spaced display headings like
    ``C A S E  R E P O R T`` normalise to the same sequence as
    ``CASE REPORT``, at the cost of losing word-boundary information.

    Each input character is NFKD-decomposed before classification (see
    ``nfkd_alnum``).

    See ``normalize_strict`` for the meaning of ``strip_html``.
    """
    skip_spans = strip_spans(text) if strip_html else []
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
        for d in nfkd_alnum(text[i]):
            normalized.append(d)
            idx_map.append(i)
        i += 1
    idx_map.append(len(text))
    return seq_smith.encode("".join(normalized), ALIGN_ALPHABET_LOOSE), tuple(idx_map)
