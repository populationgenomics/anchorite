import anchorite


def test_ocr_result_annotate() -> None:
    markdown_content = "Hello World"
    # span "Hello" is [0, 5)
    # span "World" is [6, 11)

    bbox1 = anchorite.Anchor(text="Hello", page=1, box=anchorite.BBox(0, 0, 5, 1))
    span1 = (0, 5)

    bbox2 = anchorite.Anchor(text="World", page=1, box=anchorite.BBox(6, 6, 11, 1))
    span2 = (6, 11)

    result = anchorite.AlignmentResult(
        markdown_content=markdown_content,
        anchor_spans={bbox1: span1, bbox2: span2},
        coverage_percent=1.0,
    )

    annotated = result.annotate()

    # Expected format: <span data-bbox="{top},{left},{bottom},{right}" data-page="{page}">{text}</span>
    # bbox1: 0,0,5,1 -> "0,0,5,1"
    tag1_start = '<span data-bbox="0,0,5,1" data-page="1">'
    tag_end = "</span>"

    tag2_start = '<span data-bbox="6,6,11,1" data-page="1">'

    expected = f"{tag1_start}Hello{tag_end} {tag2_start}World{tag_end}"

    assert annotated == expected


def test_ocr_result_annotate_overlap() -> None:
    # Test overlapping spans (nested)
    content = "Hello"
    bbox1 = anchorite.Anchor(text="Hello", page=1, box=anchorite.BBox(0, 0, 5, 1))
    span1 = (0, 5)

    bbox2 = anchorite.Anchor(text="He", page=1, box=anchorite.BBox(0, 0, 1, 1))
    span2 = (0, 2)

    result = anchorite.AlignmentResult(content, {bbox1: span1, bbox2: span2}, coverage_percent=1.0)

    annotated = result.annotate()

    tag1_start = '<span data-bbox="0,0,5,1" data-page="1">'
    tag2_start = '<span data-bbox="0,0,1,1" data-page="1">'
    tag_end = "</span>"

    expected = f"{tag1_start}{tag2_start}He{tag_end}llo{tag_end}"
    assert annotated == expected


def test_ocr_result_annotate_zero_length() -> None:
    # Test zero-length span
    # Text content: "Hello"
    # bbox1: "" [2, 2) (Insertion point at index 2)

    content = "Hello"
    bbox1 = anchorite.Anchor(text="", page=1, box=anchorite.BBox(0, 0, 0, 0))
    span1 = (2, 2)

    result = anchorite.AlignmentResult(content, {bbox1: span1}, coverage_percent=0.0)

    annotated = result.annotate()

    tag_start = '<span data-bbox="0,0,0,0" data-page="1">'
    tag_end = "</span>"

    expected = f"He{tag_start}{tag_end}llo"
    assert annotated == expected
