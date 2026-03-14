"""Core data types: BBox and Anchor."""

import dataclasses


@dataclasses.dataclass(frozen=True, order=True)
class BBox:
    """
    A bounding box tuple (top, left, bottom, right).
    Coordinates are typically 0-1000 normalized.
    """

    top: int
    """Top coordinate (y-min: [0-1000])."""
    left: int
    """Left coordinate (x-min: [0-1000])."""
    bottom: int
    """Bottom coordinate (y-max: [0-1000])."""
    right: int
    """Right coordinate (x-max: [0-1000])."""


@dataclasses.dataclass(frozen=True, order=True)
class Anchor:
    """An 'Anchor' links a segment of text to one or more physical locations on a page.

    Multi-box anchors arise when a single semantic unit (e.g. a sentence) spans
    several lines, each with its own bounding box.
    """

    text: str
    """The text content."""
    page: int
    """Page number (0-indexed)."""
    boxes: tuple[BBox, ...]
    """Bounding boxes — one per visual line covered by this anchor."""
