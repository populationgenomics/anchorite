"""Annotate Markdown with anchor spans, and strip them back to plain text + validation map."""

import dataclasses
import re
from collections.abc import Mapping

from .anchors import Anchor, BBox


def annotate(
    markdown: str,
    alignment: Mapping[Anchor, tuple[int, int]],
) -> str:
    """Inject coordinate ``<anchorite-span>`` tags into Markdown at aligned positions.

    Produces ``<anchorite-span data-bbox="t,l,b,r" data-page="N">text</anchorite-span>``
    for each anchor in ``alignment``. The custom element name distinguishes
    anchorite-inserted tags from any user-authored ``<span>`` HTML in the
    Markdown, so ``strip`` can round-trip without colliding with other inline
    HTML. Span boundaries are snapped outward to the edges of any enclosing
    math block (``$...$`` or ``$$...$$``) to avoid splitting LaTeX. Overlapping
    and nested spans are handled by inserting tags in sorted order.

    Args:
        markdown: The plain Markdown string to annotate.
        alignment: Mapping of Anchor -> ``(start, end)`` character offsets,
            as returned by ``align``.

    Returns:
        Annotated Markdown string with embedded ``<anchorite-span>`` tags.
    """
    math_ranges = []
    # Pattern matches $$...$$ (DOTALL) or $...$ (inline, allowing newlines for wrapped text)
    pattern = re.compile(r"(\$\$[\s\S]+?\$\$|\$[^$]+?\$)")
    for m in pattern.finditer(markdown):
        math_ranges.append((m.start(), m.end()))

    insertions = []
    for i, (anchor, (span_start, span_end)) in enumerate(alignment.items()):
        if span_start == span_end:
            continue
        start, end = span_start, span_end
        # Check for overlap with math ranges
        for m_start, m_end in math_ranges:
            if max(start, m_start) < min(end, m_end):
                # Snap to the math range
                start = min(start, m_start)
                end = max(end, m_end)
                break

        length = end - start
        box_str = ";".join(f"{b.top},{b.left},{b.bottom},{b.right}" for b in anchor.boxes)
        start_tag = f'<anchorite-span data-bbox="{box_str}" data-page="{anchor.page}">'
        end_tag = "</anchorite-span>"

        insertions.append((start, False, length, i, start_tag))
        insertions.append((end, True, length, i, end_tag))

    # Sorting rules:
    # * we wish to process in descending position order
    # * if there are position ties (n.b. later insertions end up to the left of earlier):
    #   * we want to insert closing spans to the left of opening spans, so we want to process opening spans first.
    #   * if there are opening span ties:
    #     * we want to insert longer spans to the left of shorter (so we want to process shorter spans first).
    #   * if there are closing span ties:
    #     * we want to insert longer spans to the right of shorter (so we want to process longer spans first).
    #   * if position, kind, and length are all equal (identical ranges):
    #     * the first-seen span (lower i) is treated as outer.

    def _key(x: tuple[int, bool, int, int, str]) -> tuple[int, bool, int, int]:
        position, is_closing, length, span_index, _tag = x
        return (
            -position,
            is_closing,  # opening spans sort before closing.
            -length if is_closing else +length,  # longer closing spans and shorter opening spans first.
            +span_index if is_closing else -span_index,  # first-seen span is outer.
        )

    insertions.sort(key=_key)

    chars = list(markdown)
    for index, _, _, _, text in insertions:
        chars.insert(index, text)

    return "".join(chars)


@dataclasses.dataclass(frozen=True)
class StrippedMarkdown:
    """Markdown content with tags stripped and a validation map."""

    plain_text: str
    """The plain text with all anchor spans removed."""
    validation_map: list[tuple[int, int, Anchor]]
    """A list of (start, end, Anchor) ranges in plain_text."""


def strip(annotated_md: str) -> StrippedMarkdown:
    """Remove ``<anchorite-span>`` annotation tags and build a validation map.

    Returns a ``StrippedMarkdown`` with two fields:

    - ``plain_text``: the Markdown with all anchor spans removed.
    - ``validation_map``: sorted list of ``(start, end, Anchor)`` tuples giving
      each anchor's character range in ``plain_text``.

    Only ``<anchorite-span>`` tags produced by ``annotate`` are removed; any
    other inline HTML (including user-authored ``<span>`` tags) passes through
    into ``plain_text`` unchanged.

    The validation map can be used to verify that a generated quote is grounded
    in the source document — see ``resolve`` for the higher-level interface.
    """
    # Regex to find <anchorite-span data-bbox="..." data-page="...">...</anchorite-span>
    token_pattern = re.compile(
        r'(?P<start><anchorite-span data-bbox="(?P<bbox>-?\d+,-?\d+,-?\d+,-?\d+(?:;-?\d+,-?\d+,-?\d+,-?\d+)*)"'
        r' data-page="(?P<page>\d+)">)|(?P<end></anchorite-span>)',
    )

    plain_chars = []
    validation_map = []
    # Stack stores (start_index_in_plain_text, anchor_object)
    stack = []
    last_pos = 0
    current_plain_pos = 0

    for match in token_pattern.finditer(annotated_md):
        # Text before the tag
        before = annotated_md[last_pos : match.start()]
        plain_chars.append(before)
        current_plain_pos += len(before)

        if match.group("start"):
            bbox_str = match.group("bbox")
            page = int(match.group("page"))
            boxes = tuple(BBox(*[int(x) for x in group.split(",")]) for group in bbox_str.split(";"))
            anchor = Anchor(text="", page=page, boxes=boxes)
            stack.append((current_plain_pos, anchor))
        elif stack:
            start_plain_pos, anchor = stack.pop()
            validation_map.append((start_plain_pos, current_plain_pos, anchor))

        last_pos = match.end()

    plain_chars.append(annotated_md[last_pos:])
    return StrippedMarkdown(
        plain_text="".join(plain_chars),
        validation_map=sorted(validation_map),
    )
