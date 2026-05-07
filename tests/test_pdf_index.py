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
    assert len(index._flat_to_page_char) == len(index._flat_str)
    # Single page → no inter-page separators with char_idx == -1.
    assert all(c >= 0 for c in index._flat_to_page_char)
    # Norm-to-flat sentinel.
    assert index._norm_to_flat[-1] == len(index._flat_str)
    assert len(index._norm_to_flat) == len(index._flat_norm) + 1


def test_construction_inserts_separator_between_pages() -> None:
    pdf = _make_pdf([_PAGE0, _PAGE1])
    index = PdfIndex(pdf)

    # Exactly one inter-page separator (-1 char-idx) between two pages.
    seps = [i for i, ci in enumerate(index._flat_to_page_char) if ci < 0]
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


def test_construction_with_markdown_drops_unmatched_chars() -> None:
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
