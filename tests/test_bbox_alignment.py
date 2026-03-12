import anchorite
from anchorite import bbox_alignment


def test_missing_assignment_hyphenation_and_hole_filling() -> None:
    markdown_content = """\
we enter an era of precision cancer medicine, where many drugs are
active in small molecularly defined subgroups of patients (e.g., only
3%-7% of lung cancer patients harbor the drug sensi- tizing EML4-ALK
gene fusion (Soda et al., 2007)), the scarcity of models for many
cancer genotypes and tissues is a limitation. New cell culturing tech-
nologies enable derivation of patient cell lines with high efficiency
and thus make derivation of a larger set of cell lines encompassing
the molecular diversity of cancer a realistic possibility (Liu et al.,
2012; Sato et al., 2011).
"""

    bbox_texts = [
        "set of cell lines encompassing the molecular diversity of cancer",
        "lines with high efficiency and thus make derivation of a larger",
        "are active in small molecularly defined subgroups of patients",
        "we enter an era of precision cancer medicine, where many drugs",
        "models for many cancer genotypes and tissues is a limitation.",
        "tizing EML4-ALK gene fusion [Soda et al., 2007]), the scarcity of",
        "(e.g., only 3%-7% of lung cancer patients harbor the drug sensi-",
        "a realistic possibility (Liu et al., 2012; Sato et al., 2011).",
        "New cell culturing technologies enable derivation of patient cell",
    ]

    dummy_rect = anchorite.BBox(0, 0, 0, 0)
    anchors = [anchorite.Anchor(text=t, page=0, box=dummy_rect) for t in bbox_texts]

    # Run alignment
    assignments = bbox_alignment.align_anchors(markdown_content, anchors)

    # Verify all anchors are assigned
    assert len(assignments) == len(anchors)

    # Ensure every original anchor is in the assignment dictionary
    for anchor in anchors:
        assert anchor in assignments


def test_hyphen_match_simple() -> None:
    # A simpler unit test for the hyphen logic specifically via Gapped Alignment
    markdown = "hyphen- ated"
    bbox_text = "hyphenated"

    anchors = [anchorite.Anchor(text=bbox_text, page=1, box=anchorite.BBox(0, 0, 0, 0))]

    # Run
    assignments = bbox_alignment.align_anchors(markdown, anchors)

    # Verify assignment
    assert anchors[0] in assignments
