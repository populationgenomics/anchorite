import anchorite


def test_nested_zero_length_at_start() -> None:
    # Zero-length span bbox_b is skipped; only bbox_a wraps content.
    content = "Hello"
    bbox_a = anchorite.Anchor(text="Hello", page=1, boxes=(anchorite.BBox(0, 0, 5, 1),))
    bbox_b = anchorite.Anchor(text="", page=1, boxes=(anchorite.BBox(0, 0, 0, 0),))

    annotated = anchorite.annotate(content, {bbox_a: (0, 5), bbox_b: (0, 0)})

    tag_a = '<span data-bbox="0,0,5,1" data-page="1">'
    assert f"{tag_a}Hello</span>" == annotated


def test_nested_zero_length_at_end() -> None:
    # Zero-length span bbox_c is skipped; only bbox_a wraps content.
    content = "Hello"
    bbox_a = anchorite.Anchor(text="Hello", page=1, boxes=(anchorite.BBox(0, 0, 5, 1),))
    bbox_c = anchorite.Anchor(text="", page=1, boxes=(anchorite.BBox(5, 5, 5, 5),))

    annotated = anchorite.annotate(content, {bbox_a: (0, 5), bbox_c: (5, 5)})

    tag_a = '<span data-bbox="0,0,5,1" data-page="1">'
    assert f"{tag_a}Hello</span>" == annotated


def test_identical_range_spans_are_properly_nested() -> None:
    # Two spans with identical (start, end) must be properly nested, not crossed.
    # The first span in the alignment dict is treated as outer.
    content = "Hello"
    bbox_a = anchorite.Anchor(text="Hello", page=1, boxes=(anchorite.BBox(0, 0, 5, 1),))
    bbox_b = anchorite.Anchor(text="Hello", page=1, boxes=(anchorite.BBox(0, 0, 5, 2),))

    annotated = anchorite.annotate(content, {bbox_a: (0, 5), bbox_b: (0, 5)})

    tag_a = '<span data-bbox="0,0,5,1" data-page="1">'
    tag_b = '<span data-bbox="0,0,5,2" data-page="1">'
    end = "</span>"

    assert annotated == f"{tag_a}{tag_b}Hello{end}{end}"


def test_co_terminal_closing_spans() -> None:
    # Two spans that share the same end position but different starts: the inner
    # (shorter) span must close before the outer (longer) span.
    content = "abcde"
    bbox_outer = anchorite.Anchor(text="", page=1, boxes=(anchorite.BBox(0, 0, 1, 1),))
    bbox_inner = anchorite.Anchor(text="", page=1, boxes=(anchorite.BBox(0, 0, 2, 2),))

    # outer = [0,5), inner = [2,5): both end at 5
    annotated = anchorite.annotate(content, {bbox_outer: (0, 5), bbox_inner: (2, 5)})

    tag_outer = '<span data-bbox="0,0,1,1" data-page="1">'
    tag_inner = '<span data-bbox="0,0,2,2" data-page="1">'
    end = "</span>"

    assert annotated == f"{tag_outer}ab{tag_inner}cde{end}{end}"


def test_zero_length_span_at_abutting_boundary() -> None:
    # A zero-length span sitting exactly at the boundary between two abutting
    # spans must be skipped without disturbing the boundary tag ordering.
    content = "underdiagnosed"
    bbox_a = anchorite.Anchor(text="under-", page=0, boxes=(anchorite.BBox(0, 0, 10, 100),))
    bbox_zero = anchorite.Anchor(text="", page=0, boxes=(anchorite.BBox(10, 0, 10, 100),))
    bbox_b = anchorite.Anchor(text="diagnosed", page=0, boxes=(anchorite.BBox(10, 0, 20, 100),))

    annotated = anchorite.annotate(content, {bbox_a: (0, 5), bbox_zero: (5, 5), bbox_b: (5, 14)})

    tag_a = '<span data-bbox="0,0,10,100" data-page="0">'
    tag_b = '<span data-bbox="10,0,20,100" data-page="0">'
    end = "</span>"

    assert annotated == f"{tag_a}under{end}{tag_b}diagnosed{end}"
