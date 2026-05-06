import anchorite


def test_anchorite_align_and_annotate() -> None:
    markdown = "The quick brown fox jumps over the lazy dog."
    anchors = [
        anchorite.Anchor("quick brown fox", 0, (anchorite.BBox(10, 10, 20, 100),)),
        anchorite.Anchor("lazy dog", 0, (anchorite.BBox(50, 50, 60, 150),)),
    ]

    alignment = anchorite.align(anchors, markdown)
    assert len(alignment) == 2

    annotated = anchorite.annotate(markdown, alignment)
    assert '<span data-bbox="10,10,20,100" data-page="0">quick brown fox</span>' in annotated
    assert '<span data-bbox="50,50,60,150" data-page="0">lazy dog</span>' in annotated


def test_anchorite_math_snapping() -> None:
    markdown = "The formula is $E=mc^2$ and it is famous."
    anchors = [
        # Anchor points inside the math
        anchorite.Anchor("mc^2", 0, (anchorite.BBox(100, 100, 110, 200),)),
    ]

    alignment = anchorite.align(anchors, markdown)
    annotated = anchorite.annotate(markdown, alignment)

    # Should snap to the $...$ boundaries
    assert 'is <span data-bbox="100,100,110,200" data-page="0">$E=mc^2$</span> and' in annotated


def test_anchorite_resolve() -> None:
    annotated = (
        'The <span data-bbox="10,10,20,20" data-page="0">quick brown fox jumps over</span> the '
        '<span data-bbox="30,30,40,40" data-page="1">lazy dog that slept all day</span>.'
    )
    quotes = ["quick brown fox jumps over", "lazy dog that slept all day"]

    results = anchorite.resolve(annotated, quotes)

    assert results["quick brown fox jumps over"] == [(0, anchorite.BBox(10, 10, 20, 20))]
    assert results["lazy dog that slept all day"] == [(1, anchorite.BBox(30, 30, 40, 40))]


def test_anchorite_strip_and_nested_resolve() -> None:
    # Nested spans: inner is inside outer
    annotated = (
        '<span data-bbox="0,0,100,100" data-page="0">The quick brown fox '
        '<span data-bbox="10,10,20,20" data-page="0">jumps over the lazy</span> dog</span>'
    )

    stripped = anchorite.strip(annotated)
    assert stripped.plain_text == "The quick brown fox jumps over the lazy dog"

    results = anchorite.resolve(annotated, ["jumps over the lazy"])
    # "jumps over the lazy" should be mapped to both bboxes
    assert len(results["jumps over the lazy"]) == 2
    assert (0, anchorite.BBox(0, 0, 100, 100)) in results["jumps over the lazy"]
    assert (0, anchorite.BBox(10, 10, 20, 20)) in results["jumps over the lazy"]


def test_anchorite_resolve_partial_quote() -> None:
    # A quote that spans across anchors
    annotated = (
        '<span data-bbox="1,1,1,1" data-page="0">The quick brown fox jumps</span> '
        'over the <span data-bbox="2,2,2,2" data-page="0">lazy dog that slept</span>'
    )
    # plain text is "The quick brown fox jumps over the lazy dog that slept"

    quote = "fox jumps over the lazy dog"
    results = anchorite.resolve(annotated, [quote])
    # Should find both bboxes because both contribute to the quote
    assert len(results[quote]) == 2
    assert (0, anchorite.BBox(1, 1, 1, 1)) in results[quote]
    assert (0, anchorite.BBox(2, 2, 2, 2)) in results[quote]


# ---------------------------------------------------------------------------
# resolve_quote: lit-manager-style API (markdown + (span, page, box) records)
# ---------------------------------------------------------------------------

def test_resolve_quote_basic() -> None:
    markdown = "The quick brown fox jumps over the lazy dog."
    spans = [
        anchorite.SpanAnchor(span=(0, 25), page=0, box=anchorite.BBox(10, 10, 20, 20)),
        anchorite.SpanAnchor(span=(25, 44), page=1, box=anchorite.BBox(30, 30, 40, 40)),
    ]
    out = anchorite.resolve_quote(markdown, spans, "quick brown fox jumps")
    assert out == [(0, anchorite.BBox(10, 10, 20, 20))]


def test_resolve_quote_overlapping_spans() -> None:
    # A quote that crosses span boundaries returns boxes from both spans.
    markdown = "The quick brown fox jumps over the lazy dog that slept."
    spans = [
        anchorite.SpanAnchor(span=(0, 25), page=0, box=anchorite.BBox(1, 1, 1, 1)),
        anchorite.SpanAnchor(span=(34, 55), page=0, box=anchorite.BBox(2, 2, 2, 2)),
    ]
    quote = "fox jumps over the lazy dog"
    out = anchorite.resolve_quote(markdown, spans, quote)
    assert (0, anchorite.BBox(1, 1, 1, 1)) in out
    assert (0, anchorite.BBox(2, 2, 2, 2)) in out


def test_resolve_quote_returns_all_boxes_for_duplicate_starts() -> None:
    # A multi-line anchor (e.g. a wrapped sentence) emits one SpanAnchor per
    # visual line, all sharing the same span start/end.  Every box must be
    # returned — earlier implementations relied on ``bisect_right - 1`` and
    # silently dropped all but the last record at a given start position.
    markdown = "A very long sentence that wraps across three visual lines on the page."
    line1 = anchorite.BBox(10, 10, 20, 100)
    line2 = anchorite.BBox(22, 10, 32, 100)
    line3 = anchorite.BBox(34, 10, 44, 200)
    spans = [
        anchorite.SpanAnchor(span=(0, len(markdown)), page=0, box=line1),
        anchorite.SpanAnchor(span=(0, len(markdown)), page=0, box=line2),
        anchorite.SpanAnchor(span=(0, len(markdown)), page=0, box=line3),
    ]
    out = anchorite.resolve_quote(markdown, spans, markdown)
    pages = sorted({(p, b) for p, b in out})
    assert (0, line1) in pages
    assert (0, line2) in pages
    assert (0, line3) in pages


def test_resolve_quote_uses_html_aware_normalisation() -> None:
    # The Markdown carries ``<sup>1</sup>`` markup; the LLM-extracted quote
    # comes from rendered text and has plain digits.  They must align via
    # the shared normalisation, so the bbox is returned.
    markdown = "Author<sup>1</sup> reported the variant."
    spans = [
        anchorite.SpanAnchor(span=(0, 39), page=0, box=anchorite.BBox(50, 100, 60, 400)),
    ]
    out = anchorite.resolve_quote(markdown, spans, "Author1 reported the variant")
    assert out == [(0, anchorite.BBox(50, 100, 60, 400))]


def test_resolve_quote_uses_nfkd_normalisation() -> None:
    # Markdown has the precomposed accent; the LLM-extracted quote has the
    # accent stripped.  NFKD decomposition + ASCII filter aligns them.
    markdown = "Töpf et al. described the cohort."
    spans = [
        anchorite.SpanAnchor(span=(0, 33), page=0, box=anchorite.BBox(50, 100, 60, 400)),
    ]
    out = anchorite.resolve_quote(markdown, spans, "Topf et al. described the cohort")
    assert out == [(0, anchorite.BBox(50, 100, 60, 400))]


def test_resolve_quote_low_coverage_returns_empty() -> None:
    markdown = "The quick brown fox jumps over the lazy dog."
    spans = [
        anchorite.SpanAnchor(span=(0, 44), page=0, box=anchorite.BBox(1, 1, 1, 1)),
    ]
    # Mostly content that doesn't appear in the markdown.
    out = anchorite.resolve_quote(markdown, spans, "vienna sausage tetragrammaton mendelevium recombinant")
    assert out == []


def test_resolve_quote_empty_inputs() -> None:
    spans = [anchorite.SpanAnchor(span=(0, 5), page=0, box=anchorite.BBox(0, 0, 0, 0))]
    assert anchorite.resolve_quote("hello", spans, "") == []
    assert anchorite.resolve_quote("hello", spans, "   ") == []
    assert anchorite.resolve_quote("hello", [], "hello") == []
