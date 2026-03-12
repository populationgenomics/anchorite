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
    """An 'Anchor' links a segment of text to a physical location (BBox) on a page."""

    text: str
    """The text content."""
    page: int
    """Page number (0-indexed)."""
    box: BBox
    """The bounding box coordinates."""
