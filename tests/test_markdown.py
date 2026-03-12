from anchorite.markdown import renumber_markers


def test_renumber_markers() -> None:
    chunks = [
        "Part 1: <!--table--> Content <!--table--> <!--figure-->",
        "Part 2: <!--table--> Content <!--figure--> <!--figure-->",
    ]
    renumbered = renumber_markers(chunks)

    assert renumbered[0] == "Part 1: <!--table: 1--> Content <!--table: 2--> <!--figure: 1-->"
    assert renumbered[1] == "Part 2: <!--table: 3--> Content <!--figure: 2--> <!--figure: 3-->"
