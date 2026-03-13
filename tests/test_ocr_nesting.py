import anchorite


def test_nested_zero_length_at_start() -> None:
    # Zero-length span bbox_b is skipped; only bbox_a wraps content.
    content = "Hello"
    bbox_a = anchorite.Anchor(text="Hello", page=1, box=anchorite.BBox(0, 0, 5, 1))
    bbox_b = anchorite.Anchor(text="", page=1, box=anchorite.BBox(0, 0, 0, 0))

    annotated = anchorite.annotate(content, {bbox_a: (0, 5), bbox_b: (0, 0)})

    tag_a = '<span data-bbox="0,0,5,1" data-page="1">'
    assert f"{tag_a}Hello</span>" == annotated


def test_nested_zero_length_at_end() -> None:
    # Zero-length span bbox_c is skipped; only bbox_a wraps content.
    content = "Hello"
    bbox_a = anchorite.Anchor(text="Hello", page=1, box=anchorite.BBox(0, 0, 5, 1))
    bbox_c = anchorite.Anchor(text="", page=1, box=anchorite.BBox(5, 5, 5, 5))

    annotated = anchorite.annotate(content, {bbox_a: (0, 5), bbox_c: (5, 5)})

    tag_a = '<span data-bbox="0,0,5,1" data-page="1">'
    assert f"{tag_a}Hello</span>" == annotated
