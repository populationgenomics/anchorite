"""Tests for ``anchorite.PdfIndex``.

Each test synthesises a tiny PDF in-memory via ``pypdfium2``'s text-creation
API so the suite stays self-contained — the repo doesn't ship any PDF
fixtures.
"""

from __future__ import annotations

import ctypes
import io

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from anchorite import BBox, PdfIndex


def _make_pdf(pages: list[str]) -> bytes:
    """Build a multi-page PDF; each page carries one text run at (100, 700)."""
    doc = pdfium.PdfDocument.new()
    for text in pages:
        page = doc.new_page(612, 792)
        text_obj = pdfium_c.FPDFPageObj_NewTextObj(doc.raw, b"Helvetica", 12.0)
        buf = (ctypes.c_ushort * (len(text) + 1))()
        for i, ch in enumerate(text):
            buf[i] = ord(ch)
        buf[len(text)] = 0
        pdfium_c.FPDFText_SetText(text_obj, buf)
        matrix = pdfium_c.FS_MATRIX(1.0, 0.0, 0.0, 1.0, 100.0, 700.0)
        pdfium_c.FPDFPageObj_SetMatrix(text_obj, ctypes.byref(matrix))
        pdfium_c.FPDFPage_InsertObject(page, text_obj)
        pdfium_c.FPDFPage_GenerateContent(page)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


# Reusable single-page corpus.
_PAGE0 = "Hello world the quick brown fox jumps over the lazy dog"
_PAGE1 = "A second page with completely different content for resolution tests"


def test_construction_no_markdown_populates_cache() -> None:
    pdf = _make_pdf([_PAGE0])
    index = PdfIndex(pdf)

    assert index._flat_str
    assert len(index._flat_to_page) == len(index._flat_str)
    assert len(index._flat_to_page_atom) == len(index._flat_str)
    # Single page → no inter-page separators with atom_idx == -1.
    assert all(c >= 0 for c in index._flat_to_page_atom)
    # Norm-to-flat sentinel.
    assert index._norm_to_flat[-1] == len(index._flat_str)
    assert len(index._norm_to_flat) == len(index._flat_norm) + 1


def test_construction_inserts_separator_between_pages() -> None:
    pdf = _make_pdf([_PAGE0, _PAGE1])
    index = PdfIndex(pdf)

    # Exactly one inter-page separator (-1 atom-idx) between two pages.
    seps = [i for i, ci in enumerate(index._flat_to_page_atom) if ci < 0]
    assert len(seps) == 1


def test_resolve_known_quote_returns_first_page() -> None:
    pdf = _make_pdf([_PAGE0, _PAGE1])
    index = PdfIndex(pdf)

    result = index.resolve(["the quick brown fox"])
    boxes = result["the quick brown fox"]

    assert boxes, "expected at least one (page, BBox) tuple"
    pages = {p for p, _ in boxes}
    assert pages == {0}
    for _, b in boxes:
        assert isinstance(b, BBox)


def test_resolve_known_quote_returns_second_page() -> None:
    pdf = _make_pdf([_PAGE0, _PAGE1])
    index = PdfIndex(pdf)

    result = index.resolve(["completely different content"])
    boxes = result["completely different content"]

    assert boxes
    assert {p for p, _ in boxes} == {1}


def test_resolve_pages_are_zero_indexed() -> None:
    pdf = _make_pdf([_PAGE0])
    index = PdfIndex(pdf)
    boxes = index.resolve(["quick brown fox"])["quick brown fox"]
    assert boxes
    assert {p for p, _ in boxes} == {0}


def test_resolve_empty_and_whitespace_quotes_return_empty_list() -> None:
    pdf = _make_pdf([_PAGE0])
    index = PdfIndex(pdf)

    result = index.resolve(["", "   ", "\n\t"])
    assert result == {"": [], "   ": [], "\n\t": []}


def test_resolve_unmatchable_quote_returns_empty_list() -> None:
    pdf = _make_pdf([_PAGE0])
    index = PdfIndex(pdf)

    result = index.resolve(["xyzzy nothing of the sort exists here zwzwzw"])
    assert result["xyzzy nothing of the sort exists here zwzwzw"] == []


def test_resolve_batches_many_quotes() -> None:
    pdf = _make_pdf([_PAGE0, _PAGE1])
    index = PdfIndex(pdf)

    quotes = [
        "the quick brown fox",
        "completely different content",
        "garbage that does not appear",
        "",
    ]
    result = index.resolve(quotes)

    assert {p for p, _ in result["the quick brown fox"]} == {0}
    assert {p for p, _ in result["completely different content"]} == {1}
    assert result["garbage that does not appear"] == []
    assert result[""] == []


def test_resolve_every_input_quote_is_a_dict_key() -> None:
    pdf = _make_pdf([_PAGE0])
    index = PdfIndex(pdf)

    quotes = ["the quick brown fox", "garbage", ""]
    result = index.resolve(quotes)

    assert set(result.keys()) == set(quotes)


def test_resolve_deduplicates_identical_inputs() -> None:
    pdf = _make_pdf([_PAGE0])
    index = PdfIndex(pdf)

    quote = "the quick brown fox"
    result = index.resolve([quote, quote, quote])

    # Identical inputs collapse onto a single key with the expected match.
    assert list(result.keys()) == [quote]
    assert result[quote]


def test_resolve_deduplicates_normalisation_equivalents() -> None:
    pdf = _make_pdf([_PAGE0])
    index = PdfIndex(pdf)

    # Whitespace and case variants normalise to the same bytes; each input
    # string remains a distinct dict key but all share the same bbox list.
    variants = [
        "the quick brown fox",
        "THE QUICK BROWN FOX",
        "the   quick\tbrown  fox",
    ]
    result = index.resolve(variants)

    assert set(result.keys()) == set(variants)
    first = result[variants[0]]
    assert first
    for v in variants[1:]:
        assert result[v] == first


def test_resolve_num_threads_pass_through() -> None:
    pdf = _make_pdf([_PAGE0, _PAGE1])
    index = PdfIndex(pdf)
    quote = "the quick brown fox"

    default = index.resolve([quote])
    explicit = index.resolve([quote], num_threads=1)

    # Same quote, same index → identical (page, BBox) sets.
    assert {(p, (b.top, b.left, b.bottom, b.right)) for p, b in default[quote]} == {
        (p, (b.top, b.left, b.bottom, b.right)) for p, b in explicit[quote]
    }


def test_resolve_against_empty_pdf_text_returns_empty_lists() -> None:
    # A PDF with one page containing only whitespace has no extractable chars.
    pdf = _make_pdf([" "])
    index = PdfIndex(pdf)
    result = index.resolve(["anything"])
    assert result == {"anything": []}


def test_construction_with_markdown_resolves_known_quote() -> None:
    # Markdown matches the PDF text closely; cleanup path should still
    # produce a working cache for resolve.
    pdf = _make_pdf([_PAGE0, _PAGE1])
    md = "<!--page-->\n\n" + _PAGE0 + "\n\n<!--page-->\n\n" + _PAGE1 + "\n"
    index = PdfIndex(pdf, markdown=md)

    result = index.resolve(["the quick brown fox"])
    boxes = result["the quick brown fox"]
    assert boxes
    assert {p for p, _ in boxes} == {0}


def test_construction_with_markdown_drops_unmatched_atoms() -> None:
    # Page 0 carries text the markdown describes; page 1's text is *not*
    # in the markdown, so the cleanup path drops it.  A quote drawn from
    # page 1 should then fail to resolve.
    pdf = _make_pdf([_PAGE0, _PAGE1])
    md = "<!--page-->\n\n" + _PAGE0 + "\n"  # page 1 omitted on purpose

    index = PdfIndex(pdf, markdown=md)

    # Page-0 quote still works.
    assert index.resolve(["the quick brown fox"])["the quick brown fox"]
    # Page-1 quote was dropped by the cleanup pass — no chars to align against.
    assert index.resolve(["completely different content"])["completely different content"] == []


# ---------------------------------------------------------------------------
# Regression: short table cells survive the cleanup
#
# Per-segment alignment dropped any markdown segment shorter than the
# score-threshold floor — i.e. most cells in a typical data table — so
# the resulting cleaned cache contained none of those chars and a quote
# spanning the row failed to resolve.  Character-level chained
# alignment anchors short fragments via their neighbours' context;
# these tests pin that behaviour.
# ---------------------------------------------------------------------------


_NARRATIVE_BEFORE = (
    "introduction paragraph that explains the study cohort sample selection "
    "and methodology used to gather the case data described in detail below"
)
_NARRATIVE_AFTER = (
    "discussion paragraph summarising the findings clinical significance and "
    "implications for future research directions in this disease area"
)


def test_table_row_quote_resolves_with_markdown_denoising() -> None:
    """A verbatim markdown table row resolves to a bbox after denoising.

    Reproduces the original bug pattern: paragraphs around a row of
    short cells (digits, single letters, short symbol strings).  In
    the per-segment design, the short cells failed the score-threshold
    floor and were dropped from the cleaned cache.  Under chained
    alignment they survive by sequence-level coherence with the
    flanking paragraphs.
    """
    row_text = "31 m 28 10 syncope 14174 y4725c ct trio de novo none none"
    pdf_page = f"{_NARRATIVE_BEFORE} {row_text} {_NARRATIVE_AFTER}"
    pdf = _make_pdf([pdf_page])

    # Markdown wraps the same row in GFM table syntax — pipes collapse
    # to spaces during normalisation.
    md = (
        f"{_NARRATIVE_BEFORE}.\n\n"
        "| id | sex | age | onset | sym | mut | aa | type | inh | seg | f1 | f2 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| 31 | M | 28 | 10 | syncope | 14174 | Y4725C | CT | Trio | de novo | none | none |\n\n"
        f"{_NARRATIVE_AFTER}.\n"
    )

    index = PdfIndex(pdf, markdown=md)
    quote = "31 | M | 28 | 10 | syncope | 14174 | Y4725C | CT | Trio | de novo | none | none"
    boxes = index.resolve([quote])[quote]
    assert boxes, "row quote should resolve to at least one bbox"
    assert {p for p, _ in boxes} == {0}


def test_all_short_cells_row_survives_cleanup() -> None:
    """Stress: a row of *only* short cells, none individually anchorable.

    The chain has no entries inside the row — every cell is too short
    to clear the seed-score floor.  Surrounding paragraphs anchor the
    chain; the gap-fill local SW between them recovers the row.

    Asserts on the cleaned cache directly, not via ``resolve``, because
    synthetic PDFs introduce kerning artifacts (digits like ``10`` may
    render with a font-internal gap that ``build_atom_index`` parses
    as a word break — ``1 0``) that affect short-quote scoring even
    when the alignment correctly claimed every atom.  The algorithm
    itself is covered by ``test_chained_alignment``'s
    all-short-cells case at the byte level, free of those artifacts.
    """
    row_text = "31 m 28 10 ct 19 f 22 8 pa 14 m 33 9 vt"
    pdf_page = f"{_NARRATIVE_BEFORE} {row_text} {_NARRATIVE_AFTER}"
    pdf = _make_pdf([pdf_page])

    md = (
        f"{_NARRATIVE_BEFORE}.\n\n"
        "| a | b | c | d | e |\n|---|---|---|---|---|\n"
        "| 31 | M | 28 | 10 | CT |\n"
        "| 19 | F | 22 | 8 | PA |\n"
        "| 14 | M | 33 | 9 | VT |\n\n"
        f"{_NARRATIVE_AFTER}.\n"
    )

    index_with_md = PdfIndex(pdf, markdown=md)
    index_without_md = PdfIndex(pdf)

    # The denoising should preserve nearly all atoms — the markdown
    # covers the PDF's content (modulo formatting characters).  Any
    # large drop here would indicate the cleanup is incorrectly
    # discarding row content.
    coverage = len(index_with_md._flat_str) / max(len(index_without_md._flat_str), 1)
    assert coverage >= 0.9, f"unexpected coverage {coverage:.2%}; row content likely dropped"

    # Spot-check: digits from the row appear in the cleaned cache.
    # Using normalised bytes is robust to kerning-induced inserted
    # spaces, since normalize_strict collapses any whitespace run.
    cache_no_space = index_with_md._flat_str.replace(" ", "")
    for token in ("31", "28", "10", "ct", "19", "22", "pa", "14", "33", "vt"):
        assert token in cache_no_space, f"row token {token!r} missing from cleaned cache"


def test_jats_style_markdown_without_page_markers_still_denoises() -> None:
    """JATS-derived markdown carries no ``<!--page-->`` markers.

    The chained alignment doesn't depend on markers; it operates on the
    full-document byte stream.  Denoising should still work end-to-end.
    """
    pdf = _make_pdf([_PAGE0, _PAGE1])
    # No <!--page--> markers at all; two paragraphs separated by a blank line.
    md = f"{_PAGE0}\n\n{_PAGE1}\n"

    index = PdfIndex(pdf, markdown=md)

    # Both pages' content survived the cleanup.
    assert index.resolve(["the quick brown fox"])["the quick brown fox"]
    assert index.resolve(["completely different content"])["completely different content"]


def test_hallucinated_markdown_does_not_break_real_content() -> None:
    """Markdown contains a sentence with no PDF counterpart (hallucination).

    The fake sentence becomes a B-gap in the alignment and contributes
    no claimed atoms.  Real content around it should still align and
    resolve normally.
    """
    pdf = _make_pdf([_PAGE0, _PAGE1])
    md = (
        f"<!--page-->\n\n{_PAGE0}\n\n"
        "this entire sentence is a hallucination that does not exist in the pdf at all\n\n"
        f"<!--page-->\n\n{_PAGE1}\n"
    )

    index = PdfIndex(pdf, markdown=md)
    assert index.resolve(["the quick brown fox"])["the quick brown fox"]
    assert index.resolve(["completely different content"])["completely different content"]
    # The hallucinated sentence should NOT resolve — its chars are not
    # in the PDF, so they're not in the cleaned cache either.
    assert index.resolve(["hallucination that does not exist"])["hallucination that does not exist"] == []
