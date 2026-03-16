"""Parse Markdown into fine-grained segments for anchor alignment."""

import dataclasses
import re
import string


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
        before = text[prev : m.start()]
        word_m = re.search(r"([a-zA-Z]+)[" + _SUPERSCRIPT_DIGITS + r"0-9,\-]*$", before)
        if word_m:
            word = word_m.group(1).lower()
            if len(word) == 1 or word in _ABBREVIATIONS:
                continue
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
    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) > 1 and sum(
        1 for line in non_empty if _SUPER_PREFIX_RE.match(line.strip())
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

    Args:
        markdown: The complete assembled Markdown with ``<!--page-->``
            page-break markers.

    Returns:
        List of segments in document order.  Each segment carries the PDF page
        index (0-based) inferred from the nearest preceding ``<!--page-->``
        marker.  Segments before the first marker are excluded.
    """
    # Ensure every <!--page--> marker sits in its own blank-line-delimited block.
    # Without this, a marker that immediately follows a paragraph (no blank line)
    # ends up in the same block as that paragraph; after comment-stripping the
    # subsequent content (tables, etc.) is concatenated onto the last sentence.
    markdown = re.sub(r"(?<!\n\n)(<!--page-->)", r"\n\n\1", markdown)
    markdown = re.sub(r"(<!--page-->)(?!\n)", r"\1\n\n", markdown)

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
