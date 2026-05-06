"""Tests for markdown segment parsing and PDF association.

The parser tests cover the API contract that ``parse_markdown_segments``
accepts both the page-marked Markdown shape (chunked OCR pipelines emit
``<!--page-->`` between page-level chunks) and the no-marker shape (e.g.
JATS XML rendered to Markdown, where the source has no notion of pages).
"""

import pathlib

import pytest

from anchorite import md_association
from anchorite.md_association import (
    MarkdownSegment,
    associate,
    parse_markdown_segments,
)

# ---------------------------------------------------------------------------
# Parser: page hint behaviour
# ---------------------------------------------------------------------------

class TestParseMarkdownSegments:

    def test_with_markers_assigns_integer_pages(self) -> None:
        md = (
            "<!--page-->\n\n"
            "First page sentence.\n\n"
            "<!--page-->\n\n"
            "Second page sentence.\n"
        )
        segs = parse_markdown_segments(md)
        pages = [s.page for s in segs]
        assert pages == [0, 1]
        assert all(isinstance(s.page, int) for s in segs)

    def test_without_markers_yields_none_pages(self) -> None:
        md = (
            "First sentence.\n\n"
            "Second sentence.\n\n"
            "Third sentence.\n"
        )
        segs = parse_markdown_segments(md)
        assert len(segs) == 3
        assert all(s.page is None for s in segs)

    def test_without_markers_preserves_pre_marker_content(self) -> None:
        # The marker'd path drops content before the first marker (it isn't
        # pinned to any page).  The no-marker path must keep it.
        md = "Leading sentence.\n\nFollowing sentence.\n"
        segs = parse_markdown_segments(md)
        assert [s.text for s in segs] == ["Leading sentence.", "Following sentence."]

    def test_with_markers_drops_pre_marker_content(self) -> None:
        # Content before the first marker has no page assignment, so it's
        # excluded.  This is existing behaviour; the test pins the contract.
        md = "Stray pre-marker text.\n\n<!--page-->\n\nReal page content.\n"
        segs = parse_markdown_segments(md)
        assert [s.text for s in segs] == ["Real page content."]

    def test_no_markers_segment_structure_matches_marked_structure(self) -> None:
        # Stripping markers must not change segment granularity (heading,
        # sentences, list items) — only the page assignment changes.
        body = (
            "# Heading\n\n"
            "First sentence. Second sentence.\n\n"
            "- item one\n"
            "- item two\n"
        )
        with_markers = "<!--page-->\n\n" + body
        without_markers = body

        marked_segs = parse_markdown_segments(with_markers)
        bare_segs = parse_markdown_segments(without_markers)

        assert [s.text for s in marked_segs] == [s.text for s in bare_segs]
        assert all(s.page == 0 for s in marked_segs)
        assert all(s.page is None for s in bare_segs)

    def test_marker_immediately_after_paragraph(self) -> None:
        # Existing behaviour: a marker that follows a paragraph without a
        # blank line is moved into its own block.  Pin it so the no-marker
        # change doesn't regress it.
        md = "Paragraph one.<!--page-->\n\nPage two content.\n"
        segs = parse_markdown_segments(md)
        # "Paragraph one." is pre-marker content, dropped.  "Page two content."
        # is on page 0.
        assert len(segs) == 1
        assert segs[0].text == "Page two content."
        assert segs[0].page == 0


# ---------------------------------------------------------------------------
# associate(): contract for None-page segments
# ---------------------------------------------------------------------------

# Locate a fixture PDF/markdown pair.  The repo doesn't ship one yet (PDFs are
# heavyweight); the test is skipped if not present.  When a fixture is added
# under ``tests/fixtures/<stem>.{pdf,md}``, this asserts the no-marker path
# produces a page assignment substantively equivalent to the marked path.

_FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"


def _pairs() -> list[tuple[pathlib.Path, pathlib.Path]]:
    pairs = []
    for pdf in sorted(_FIXTURE_DIR.glob("*.pdf")):
        md = pdf.with_suffix(".md")
        if md.exists():
            pairs.append((pdf, md))
    return pairs


@pytest.mark.parametrize(("pdf_path", "md_path"), _pairs())
def test_no_markers_matches_marker_assignments(
    pdf_path: pathlib.Path, md_path: pathlib.Path,
) -> None:
    """No-marker associate() should assign each segment to the same page as
    the marker'd associate() does.  Bbox shapes need not be byte-identical
    (masking order can differ when the search window changes), but the page
    each segment lands on must agree.
    """
    marked_md = md_path.read_text()
    # Strip markers to simulate JATS-derived markdown.
    bare_md = md_association._PAGE_MARKER_RE.sub("", marked_md)

    marked_anchors = associate(pdf_path, marked_md)
    bare_anchors = associate(pdf_path, bare_md)

    # Build {text -> page} maps.  Texts are unique within a paper at the
    # sentence/heading granularity in normal cases.
    assert isinstance(marked_anchors, list)
    assert isinstance(bare_anchors, list)
    marked_pages = {a.text: a.page for a in marked_anchors}
    bare_pages = {a.text: a.page for a in bare_anchors}

    common = set(marked_pages) & set(bare_pages)
    # Coverage must be reasonably high — the no-marker path shouldn't lose
    # most segments to false-uniqueness rejections.
    assert len(common) >= 0.8 * len(marked_pages), (
        f"no-marker path matched {len(common)}/{len(marked_pages)} of marker'd "
        f"segments; expected ≥80%"
    )

    # Page assignments for shared segments must agree.
    disagreements = [t for t in common if marked_pages[t] != bare_pages[t]]
    assert not disagreements, (
        f"page assignments disagree for {len(disagreements)} segments; "
        f"first: {disagreements[0]!r} marker={marked_pages[disagreements[0]]} "
        f"bare={bare_pages[disagreements[0]]}"
    )


# ---------------------------------------------------------------------------
# Type contract
# ---------------------------------------------------------------------------

def test_segment_page_can_be_none() -> None:
    """``MarkdownSegment.page`` accepts ``None`` (frozen-dataclass type-check)."""
    seg = MarkdownSegment(text="hello", page=None, md_start=0, md_end=5)
    assert seg.page is None


# ---------------------------------------------------------------------------
# Normalisation: HTML tags must be zero-width
# ---------------------------------------------------------------------------

class TestHtmlTagsInNormalisation:
    """HTML tags surviving into Markdown (`<sup>`, `<a id="...">`, etc.) must
    not contribute alphanum bytes to the alignment string when *strip_html*
    is True — otherwise their tag names line up with letters in the PDF and
    trash the alignment.  *strip_html* defaults to False because pdfium-
    extracted PDF text contains literal ``<`` / ``>`` characters when those
    glyphs appear in the document (e.g. ``p < 0.05``); stripping them would
    silently drop real content.
    """

    def test_strict_strips_when_enabled(self) -> None:
        text = "Author<sup>1</sup>"
        plain = "Author1"
        assert (
            md_association._normalize_strict(text, strip_html=True)[0]
            == md_association._normalize_strict(plain, strip_html=True)[0]
        )

    def test_loose_strips_when_enabled(self) -> None:
        text = "Author<sup>1</sup>"
        plain = "Author1"
        assert (
            md_association._normalize_loose(text, strip_html=True)[0]
            == md_association._normalize_loose(plain, strip_html=True)[0]
        )

    def test_default_does_not_strip(self) -> None:
        # PDF-side default: `<` / `>` are literal characters and must contribute
        # like any other punctuation.  ``a<b`` (PDF rendering of ``a < b``)
        # normalises like ``a b``.
        norm, _ = md_association._normalize_strict("a<b")
        assert norm == md_association._normalize_strict("a b")[0]

    def test_idx_map_points_into_original_text(self) -> None:
        text = "Author<sup>1</sup>"
        norm_bytes, idx_map = md_association._normalize_loose(text, strip_html=True)
        # 7 alphanum bytes (a,u,t,h,o,r,1) plus one trailing sentinel.
        assert len(norm_bytes) == 7
        assert list(idx_map) == [0, 1, 2, 3, 4, 5, 11, len(text)]

    def test_anchor_tag_with_id_zero_width(self) -> None:
        text = '<a id="R1"></a>\n## References'
        plain = "## References"
        assert (
            md_association._normalize_strict(text, strip_html=True)[0]
            == md_association._normalize_strict(plain, strip_html=True)[0]
        )

    def test_lone_lt_gt_treated_as_punctuation(self) -> None:
        # A bare ``<`` or ``>`` (no closing ``>``) is left alone even when
        # strip_html is True — the regex requires a complete tag.
        norm_bytes, _ = md_association._normalize_strict("a < b", strip_html=True)
        assert norm_bytes == md_association._normalize_strict("a   b", strip_html=True)[0]


# ---------------------------------------------------------------------------
# Normalisation: NFKD compatibility decomposition
# ---------------------------------------------------------------------------

class TestNfkdNormalisation:
    """Each input character is NFKD-decomposed before classification, so
    accented letters keep their base form, ligatures expand, superscript
    digits become plain digits, and Mathematical Alphanumeric Symbols map
    back to ASCII.  Without NFKD these all dropped out as non-ASCII and
    shrank the alignable sequence — short segments hit the phase-1
    minimum-length cutoff and never matched.
    """

    def test_accented_letters_keep_base(self) -> None:
        # Töpf used to normalise to 'tpf' (ö dropped); now keeps 'topf'.
        assert md_association._normalize_loose("Töpf")[0] == md_association._normalize_loose("Topf")[0]
        assert md_association._normalize_loose("Müller")[0] == md_association._normalize_loose("Muller")[0]
        assert md_association._normalize_loose("naïve")[0] == md_association._normalize_loose("naive")[0]
        assert md_association._normalize_loose("Göngör")[0] == md_association._normalize_loose("Gongor")[0]

    def test_ligatures_expand(self) -> None:
        # ﬁnal (single ligature glyph) used to drop entirely; now becomes 'final'.
        assert md_association._normalize_loose("ﬁnal")[0] == md_association._normalize_loose("final")[0]
        assert md_association._normalize_loose("eﬃcient")[0] == md_association._normalize_loose("efficient")[0]

    def test_superscript_digits_become_plain(self) -> None:
        # ⁶¹RNRKRKAEPY⁷⁰ used to lose the superscripts entirely.
        assert md_association._normalize_loose("⁶¹RNRKRKAEPY⁷⁰")[0] == md_association._normalize_loose("61RNRKRKAEPY70")[0]
        assert md_association._normalize_loose("H²O")[0] == md_association._normalize_loose("H2O")[0]

    def test_math_alphanumeric_symbols(self) -> None:
        # Mathematical Italic capital S (𝑆) should normalise to 's', etc.
        assert md_association._normalize_loose("𝑆ensitivity")[0] == md_association._normalize_loose("Sensitivity")[0]

    def test_idx_map_for_one_to_many_decomposition(self) -> None:
        # 'ﬁnal' has 4 chars; under NFKD ``ﬁ`` decomposes to ``fi`` (two
        # bytes), so the normalised output has 5 bytes — but both
        # decomposed letters of ``ﬁ`` map back to its single original
        # offset 0.
        norm, idx_map = md_association._normalize_loose("ﬁnal")
        assert len(norm) == 5
        # Original offsets for f, i, n, a, l: ﬁ is at 0; n, a, l at 1, 2, 3.
        assert list(idx_map) == [0, 0, 1, 2, 3, 4]

    def test_combining_marks_dropped(self) -> None:
        # NFKD decomposes 'ö' to 'o' + U+0308 combining diaeresis.  The
        # combining mark must not contribute an output byte.
        norm, idx_map = md_association._normalize_loose("ö")
        assert len(norm) == 1   # just 'o', no combining mark
        assert list(idx_map) == [0, 1]

    def test_strict_mode_emits_space_for_pure_punctuation(self) -> None:
        # NFKD doesn't decompose dashes/quotes; they still collapse to a
        # space in strict mode — preserve the existing behaviour.
        norm, _ = md_association._normalize_strict("a — b")
        assert norm == md_association._normalize_strict("a   b")[0]


# ---------------------------------------------------------------------------
# _build_char_index: line-break and soft-hyphen handling
# ---------------------------------------------------------------------------

def _line(text: str, *, baseline: float, x0: float = 0.0, font_size: float = 10.0) -> list:
    """Build a sequence of ``_Char`` records on one visual line."""
    chars = []
    cursor = x0
    for c in text:
        chars.append(md_association._Char(
            text=c, x0=cursor, y0=baseline, x1=cursor + font_size * 0.6,
            y1=baseline + font_size, font_size=font_size,
        ))
        cursor += font_size * 0.6
    return chars


class TestBuildCharIndex:

    def test_line_break_inserts_space(self) -> None:
        # ``we`` at the end of one line, ``identified`` at the start of the
        # next.  Without the line-break space the flat string would read
        # ``weidentified``.
        chars = _line("we", baseline=100.0) + _line("identified", baseline=80.0)
        ci = md_association._build_char_index(chars)
        assert ci.flat_str == "we identified"

    def test_horizontal_word_gap_inserts_space(self) -> None:
        # Two words on the same line with a horizontal gap > 20 % of font size.
        chars_a = _line("hello", baseline=100.0)
        chars_b = _line("world", baseline=100.0, x0=chars_a[-1].x1 + 5.0)
        ci = md_association._build_char_index(chars_a + chars_b)
        assert ci.flat_str == "hello world"

    def test_soft_hyphen_at_line_break_reconnects(self) -> None:
        # Typeset ``induc-`` at the end of one line, ``tion`` at the start of
        # the next — this is a soft-hyphenated word that should reconnect to
        # ``induction`` (matching the Markdown's un-hyphenated form).
        chars = _line("induc-", baseline=100.0) + _line("tion", baseline=80.0)
        ci = md_association._build_char_index(chars)
        assert ci.flat_str == "induction"

    def test_hyphen_at_line_break_after_digit_keeps_hyphen(self) -> None:
        # Numeric range ``2009-`` followed by ``2010`` on the next line.  The
        # surrounding chars aren't alphabetic, so the hyphen-suppression
        # heuristic must NOT fire — a space is inserted as for any line break.
        chars = _line("2009-", baseline=100.0) + _line("2010", baseline=80.0)
        ci = md_association._build_char_index(chars)
        assert ci.flat_str == "2009- 2010"

    def test_hyphen_at_line_break_before_digit_keeps_hyphen(self) -> None:
        # Hyphenated identifier ``cohort-`` followed by ``38`` on the next
        # line.  The next char isn't alphabetic, so we keep the hyphen.
        chars = _line("cohort-", baseline=100.0) + _line("38", baseline=80.0)
        ci = md_association._build_char_index(chars)
        assert ci.flat_str == "cohort- 38"

    def test_mid_line_hyphen_unaffected(self) -> None:
        # ``e-mail`` on a single line: the hyphen stays even though it sits
        # between two letters, because it isn't at a line break.
        chars = _line("e-mail", baseline=100.0)
        ci = md_association._build_char_index(chars)
        assert ci.flat_str == "e-mail"
