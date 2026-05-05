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
    not contribute alphanum bytes to the alignment string — otherwise their
    tag names line up with letters in the PDF and trash the alignment.
    """

    def test_strict_skips_html_tag_letters(self) -> None:
        text = "Author<sup>1</sup>"
        plain = "Author1"
        # Both should normalise to the same byte sequence.
        assert md_association._normalize_strict(text)[0] == md_association._normalize_strict(plain)[0]

    def test_loose_skips_html_tag_letters(self) -> None:
        text = "Author<sup>1</sup>"
        plain = "Author1"
        assert md_association._normalize_loose(text)[0] == md_association._normalize_loose(plain)[0]

    def test_idx_map_points_into_original_text(self) -> None:
        # After stripping ``<sup>``, the alphanum bytes in normalised text
        # must map back to the *original* offsets ("A","u","t","h","o","r" at
        # 0..5; "1" at 11 — the digit between the two tags).
        text = "Author<sup>1</sup>"
        norm_bytes, idx_map = md_association._normalize_loose(text)
        # idx_map covers the 7 emitted alphanum bytes plus one trailing sentinel.
        assert len(norm_bytes) == 7
        assert list(idx_map) == [0, 1, 2, 3, 4, 5, 11, len(text)]

    def test_anchor_tag_with_id_zero_width(self) -> None:
        text = '<a id="R1"></a>\n## References'
        plain = "## References"
        # The leading <a id="R1"></a> contributes nothing; everything after it
        # normalises identically to the bare heading.
        assert md_association._normalize_strict(text)[0] == md_association._normalize_strict(plain)[0]

    def test_lone_lt_gt_treated_as_punctuation(self) -> None:
        # A bare ``<`` or ``>`` (not part of a complete ``<...>`` tag) shouldn't
        # be silently skipped — the regex requires a closing ``>`` to match.
        # Verify a stray ``<`` collapses to a space in strict mode like any
        # other punctuation.
        norm_bytes, _ = md_association._normalize_strict("a < b")
        assert norm_bytes == md_association._normalize_strict("a   b")[0]


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
