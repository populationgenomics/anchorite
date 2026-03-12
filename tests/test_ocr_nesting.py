import anchorite


def test_nested_zero_length_at_start() -> None:
    content = "Hello"
    bbox_a = anchorite.Anchor(text="Hello", page=1, box=anchorite.BBox(0, 0, 5, 1))
    bbox_b = anchorite.Anchor(text="", page=1, box=anchorite.BBox(0, 0, 0, 0))

    result = anchorite.AlignmentResult(content, {bbox_a: (0, 5), bbox_b: (0, 0)}, coverage_percent=0.0)
    annotated = result.annotate()

    tag_a = '<span data-bbox="0,0,5,1" data-page="1">'
    tag_b = '<span data-bbox="0,0,0,0" data-page="1">'

    assert f"{tag_a}{tag_b}</span>Hello</span>" == annotated


def test_nested_zero_length_at_end() -> None:
    content = "Hello"
    bbox_a = anchorite.Anchor(text="Hello", page=1, box=anchorite.BBox(0, 0, 5, 1))
    bbox_c = anchorite.Anchor(text="", page=1, box=anchorite.BBox(5, 5, 5, 5))

    result = anchorite.AlignmentResult(content, {bbox_a: (0, 5), bbox_c: (5, 5)}, coverage_percent=0.0)
    annotated = result.annotate()

    tag_a = '<span data-bbox="0,0,5,1" data-page="1">'
    tag_c = '<span data-bbox="5,5,5,5" data-page="1">'

    assert f"{tag_a}Hello{tag_c}</span></span>" == annotated
