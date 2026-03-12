import anchorite


def test_anchorite_align_and_annotate() -> None:
    markdown = "The quick brown fox jumps over the lazy dog."
    anchors = [
        anchorite.Anchor("quick brown fox", 0, anchorite.BBox(10, 10, 20, 100)),
        anchorite.Anchor("lazy dog", 0, anchorite.BBox(50, 50, 60, 150)),
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
        anchorite.Anchor("mc^2", 0, anchorite.BBox(100, 100, 110, 200)),
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
