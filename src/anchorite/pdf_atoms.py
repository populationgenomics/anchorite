"""Per-atom PDF text extraction.

An *atom* is the indivisible text-bearing unit produced by an extractor: a
glyph, a word, or a line, depending on the producer.  This module ships the
pypdfium2-backed extractor, which produces glyph-level atoms.  When a
second backend (e.g. Document AI's line-level OCR) is added, it is
expected to produce atoms of a different granularity through a sibling
module — the consumer-facing shape (``flat_str`` + ``flat_to_atom`` +
``Atom`` bbox lookup) is the same regardless of granularity, which is why
the type is named ``Atom`` rather than ``Char``.

For now every ``Atom`` is a single glyph (its ``text`` may still be
multi-character via NFKC decomposition or ligature expansion — ``ﬃ`` →
``"ffi"``).  Both ``md_association.associate`` and ``pdf_index.PdfIndex``
consume what this module produces; PDFium quirks (UTF-16 surrogate-pair
reassembly, soft-hyphen reconnection at line breaks, line-break space
insertion, mediabox origin offsets) live here and nowhere else.
"""

from __future__ import annotations

import dataclasses
import math
import unicodedata
from typing import TYPE_CHECKING, NamedTuple

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from .anchors import BBox

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Per-glyph normalisation table
# ---------------------------------------------------------------------------
# Residual mappings PDFium emits that ``unicodedata.normalize("NFKC", ...)``
# does not fold (NFKC handles ﬁ/ﬂ/ﬃ/ﬄ but leaves ﬅ/ﬆ; smart quotes and
# dashes are "real" Unicode and stay).  Applied per-glyph during extraction
# before NFKC.  Implementation detail of ``extract_page_atoms``; not part of
# the public surface.
_CHAR_NORM: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201a": ",",
    "\u2013": "-",
    "\u2014": "--",
    "\u2212": "-",
    "\u2010": "-",
    "\u2011": "-",
    "\u00ad": "",
    "\u00a0": " ",
    "\ufffe": "",
}

# UTF-16 surrogate ranges for non-BMP code points returned by PDFium.
_HIGH_SURROGATE_LO, _HIGH_SURROGATE_HI = 0xD800, 0xDBFF
_LOW_SURROGATE_LO, _LOW_SURROGATE_HI = 0xDC00, 0xDFFF
_NON_BMP_BASE = 0x10000


# ---------------------------------------------------------------------------
# Atom + flat-string index
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Atom:
    """An indivisible text-bearing unit with a bounding box.

    The current pypdfium2 producer emits one ``Atom`` per glyph.  ``text``
    is the glyph's normalised form, which may be multi-character: ``ﬃ`` →
    ``"ffi"`` after NFKC decomposition.

    Coordinates are in PDF points expressed in the page's **displayed**
    frame: origin at the page bbox's bottom-left, x increases to the right,
    y increases upward.  ``extract_page_atoms`` applies the page's
    ``/Rotate`` value (mirroring PDFium's internal ``page_matrix_``) so
    ``x`` is always screen-horizontal and ``y`` is always screen-vertical
    regardless of how the source PDF was authored.  Bboxes remain
    ``(bottom-left, top-right)`` pairs.  On a ``/Rotate=90 CCW`` page that
    means glyphs along one visible line have ascending ``y``.  Downstream
    consumers (``bbox_from_atoms``, ``line_bboxes``, ``build_atom_index``)
    need no ``/Rotate`` awareness.
    """

    text: str
    x0: float
    y0: float  # bottom in displayed PDF coords (pts, origin = page bbox bottom-left)
    x1: float
    y1: float  # top in displayed PDF coords
    font_size: float


class AtomIndex(NamedTuple):
    """Per-page flat string with a position-to-atom map.

    ``flat_str`` is the page's text built from a list of atoms by
    ``build_atom_index``: each atom's ``text`` is concatenated, with spaces
    inserted at intra-line gaps and at line breaks (and soft hyphens
    reconnected).  ``flat_to_atom[i]`` is the index into the original atoms
    list for ``flat_str[i]`` — for an inserted space, it is the index of
    the atom *before* the gap.
    """

    flat_str: str
    flat_to_atom: list[int]


@dataclasses.dataclass
class PageData:
    """Per-page extraction output cached by ``extract_page_data``.

    Single source of truth for "PDF page → cached atom data" across
    ``md_association.associate`` and ``pdf_index.PdfIndex``.  Both call
    ``extract_page_data`` to populate; downstream alignment never re-reads
    the PDF.
    """

    atoms: list[Atom]
    """Per-glyph extraction output for this page (displayed frame)."""
    atom_index: AtomIndex
    """Flat string + per-position atom index for this page."""
    width: float
    """Page width in PDF points (displayed dimension)."""
    height: float
    """Page height in PDF points (displayed dimension)."""
    rotation: int
    """Clockwise page rotation in degrees (0, 90, 180, or 270) as read from
    ``/Rotate``.  Recorded for debugging; not load-bearing downstream because
    atom coords are already in the displayed frame."""


# ---------------------------------------------------------------------------
# pypdfium2 extraction
# ---------------------------------------------------------------------------


def _rotate_charbox(
    left: float,
    bottom: float,
    right: float,
    top: float,
    rotation: int,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Map a PDFium charbox into the page's **displayed** frame.

    PDFium's ``get_charbox`` returns coords in raw PDF user space regardless
    of ``/Rotate``.  This applies the same transform PDFium uses internally
    (``CPDF_Page::UpdateDimensions``'s ``page_matrix_``): translate so the
    page bbox's bottom-left sits at ``(0, 0)``, then rotate clockwise by the
    page's ``/Rotate``.  Result is in
    ``[0, page.get_width()] × [0, page.get_height()]`` (displayed dims),
    with y increasing upward — the bottom-left-origin convention every
    downstream consumer already assumes.
    """
    bl, bb, br, bt = bbox
    if rotation == 90:  # noqa: PLR2004
        return (bottom - bb, br - right, top - bb, br - left)
    if rotation == 180:  # noqa: PLR2004
        return (br - right, bt - top, br - left, bt - bottom)
    if rotation == 270:  # noqa: PLR2004
        return (bt - top, left - bl, bt - bottom, right - bl)
    return (left - bl, bottom - bb, right - bl, top - bb)


def extract_page_atoms(page: pdfium.PdfPage) -> list[Atom]:  # noqa: C901, PLR0912, PLR0915
    """Extract non-whitespace glyphs with bboxes from a single PDF page.

    Atom coords come out in the page's **displayed** frame (origin at the
    page bbox's bottom-left).  The transform applied to every charbox is
    documented on ``_rotate_charbox``.
    """
    rotation = page.get_rotation()
    page_bbox = page.get_bbox()
    textpage = page.get_textpage()
    total_chars = textpage.count_chars()
    atoms: list[Atom] = []
    char_index = 0

    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_TEXT]):
        buf_size = pdfium_c.FPDFTextObj_GetText(obj, textpage, None, 0)
        buf = (pdfium_c.FPDF_WCHAR * buf_size)()
        pdfium_c.FPDFTextObj_GetText(obj, textpage, buf, buf_size)
        obj_text = bytes(buf).decode("utf-16-le").rstrip("\x00")

        m = obj.get_matrix()
        font_size = obj.get_font_size() * math.sqrt(m.a**2 + m.b**2)

        obj_pos = 0
        while obj_pos < len(obj_text) and char_index < total_chars:
            cp = pdfium_c.FPDFText_GetUnicode(textpage, char_index)
            # PDFium counts non-BMP characters (e.g. Mathematical Italic symbols,
            # U+1D400–U+1D7FF) as two UTF-16 surrogate-pair indices.  Detect a
            # high surrogate and reassemble the full code point from the pair.
            if _HIGH_SURROGATE_LO <= cp <= _HIGH_SURROGATE_HI:
                if char_index + 1 < total_chars:
                    cp_low = pdfium_c.FPDFText_GetUnicode(textpage, char_index + 1)
                    if _LOW_SURROGATE_LO <= cp_low <= _LOW_SURROGATE_HI:
                        cp = _NON_BMP_BASE + (cp - _HIGH_SURROGATE_LO) * 0x400 + (cp_low - _LOW_SURROGATE_LO)
                        ci_for_box = char_index
                        char_index += 2  # consume both surrogate indices
                        obj_pos += 1  # but only one code point in obj_text
                    else:
                        char_index += 1
                        obj_pos += 1
                        continue
                else:
                    char_index += 1
                    obj_pos += 1
                    continue
            elif _LOW_SURROGATE_LO <= cp <= _LOW_SURROGATE_HI:
                # Orphaned low surrogate — should not occur; skip.
                char_index += 1
                obj_pos += 1
                continue
            else:
                ci_for_box = char_index
                char_index += 1

            text = chr(cp)
            if text in ("\r", "\n"):
                # Line-break markers inserted by PDFium are absent from obj_text.
                continue  # char_index already advanced; do NOT advance obj_pos
            obj_pos += 1

            if not text.isspace():
                normalized = _CHAR_NORM.get(text, text)
                # Map Mathematical Alphanumeric Symbols and other compatibility
                # characters to ASCII equivalents (e.g. 𝑆𝑒𝑛𝑠𝑖𝑡𝑖𝑣𝑖𝑡𝑦 → Sensitivity).
                normalized = unicodedata.normalize("NFKC", normalized)
                if normalized:
                    left, bottom, right, top = textpage.get_charbox(ci_for_box, loose=False)
                    if right > left and top > bottom:
                        rl, rb, rr, rt = _rotate_charbox(
                            left,
                            bottom,
                            right,
                            top,
                            rotation,
                            page_bbox,
                        )
                        atoms.append(Atom(normalized, rl, rb, rr, rt, font_size))

    return atoms


# ---------------------------------------------------------------------------
# Flat-string assembly
# ---------------------------------------------------------------------------


def build_atom_index(atoms: Sequence[Atom]) -> AtomIndex:
    """Build a flat string and a per-position atom-index map.

    Inserts a space between successive atoms when either:

    * the horizontal gap to the next atom exceeds 20 % of font size (an
      intra-line word break), or
    * the next atom drops to a different visual line — its baseline is
      shifted vertically by more than 50 % of font size, or sits to the
      *left* of the current atom.  Without this, end-of-line + start-of-
      next-line concatenates ("``we``" + "``identified``" → "``weidentified``")
      because PDFium's coordinate stream emits no whitespace at line
      breaks, and the alignment string drifts out of sync with the
      Markdown.

    End-of-line soft hyphens are reconnected: when the line-break-trailing
    atom is ``-`` between two alphabetic glyphs, both the hyphen and the
    inserted space are dropped, so the typeset ``induc-`` + ``tion``
    reconnects to ``induction`` (matching the Markdown's un-hyphenated
    form).  Numeric ranges like ``2009-`` + ``2010`` keep the hyphen
    because the surrounding glyphs aren't alphabetic.  Dash variants
    (en-dash, em-dash, hyphen-minus) have already been normalised to
    ``-`` during atom extraction, so a single literal check suffices.
    """
    parts: list[str] = []
    flat_to_atom: list[int] = []

    for i, ch in enumerate(atoms):
        for c in ch.text:
            parts.append(c)
            flat_to_atom.append(i)
        if i + 1 < len(atoms):
            nxt = atoms[i + 1]
            x_gap = nxt.x0 - ch.x1
            y_drop = abs(nxt.y0 - ch.y0)
            line_break = nxt.x0 < ch.x0 or y_drop > ch.font_size * 0.5
            if (
                line_break
                and len(parts) >= 2  # noqa: PLR2004
                and parts[-1] == "-"
                and parts[-2].isalpha()
                and nxt.text
                and nxt.text[0].isalpha()
            ):
                # Soft hyphen between two letters at a line break: drop
                # the hyphen and insert no space.  The next iteration
                # appends the next atom's letters directly.
                parts.pop()
                flat_to_atom.pop()
                continue
            if x_gap > ch.font_size * 0.2 or line_break:
                parts.append(" ")
                flat_to_atom.append(i)

    return AtomIndex("".join(parts), flat_to_atom)


# ---------------------------------------------------------------------------
# Whole-document extraction
# ---------------------------------------------------------------------------


def extract_page_data(pdf: pdfium.PdfDocument) -> list[PageData]:
    """Extract per-page atoms, dimensions, and rotation from an open PDF document.

    Eager: every page is read up front.  Both ``md_association.associate``
    and ``pdf_index.PdfIndex`` need most or all pages anyway, and a single
    read pass keeps both consumers off divergent PDFium code paths.

    ``width``/``height`` are pypdfium2's displayed dimensions
    (``page.get_width()``/``get_height()``), pairing correctly with
    ``extract_page_atoms``'s already-displayed-frame atom coords.
    """
    page_data: list[PageData] = []
    for page_idx in range(len(pdf)):
        page = pdf[page_idx]
        atoms = extract_page_atoms(page)
        ai = build_atom_index(atoms)
        page_data.append(
            PageData(
                atoms=atoms,
                atom_index=ai,
                width=page.get_width(),
                height=page.get_height(),
                rotation=page.get_rotation(),
            ),
        )
    return page_data


# ---------------------------------------------------------------------------
# BBox helpers
# ---------------------------------------------------------------------------


def bbox_from_atoms(
    atoms: Sequence[Atom],
    page_width: float,
    page_height: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> BBox:
    """Convert an atom span to a normalised-coordinate ``BBox``.

    Coordinates are in 0-1000 normalised page space, with the mediabox
    origin subtracted so PDFs whose mediabox is offset from (0, 0) do not
    produce shifted bboxes.

    ``atoms`` must be non-empty.  Passing an empty sequence currently
    raises a bare ``ValueError`` from the internal ``min()`` / ``max()``;
    a typed error is tracked in RD-1043.
    """
    x0 = min(c.x0 for c in atoms) - origin_x
    y0 = min(c.y0 for c in atoms) - origin_y
    x1 = max(c.x1 for c in atoms) - origin_x
    y1 = max(c.y1 for c in atoms) - origin_y
    top = round((1.0 - y1 / page_height) * 1000)
    left = round(x0 / page_width * 1000)
    bottom = round((1.0 - y0 / page_height) * 1000)
    right = round(x1 / page_width * 1000)
    return BBox(top=top, left=left, bottom=bottom, right=right)


def line_bboxes(
    atoms: Sequence[Atom],
    page_width: float,
    page_height: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> list[BBox]:
    """Return one ``BBox`` per line of matched atoms.

    Atoms are sorted top-to-bottom by y-midpoint and grouped into lines by
    y-overlap: an atom joins the current line when its y-range overlaps the
    accumulated line band; otherwise it starts a new line.  This gives one
    tight box per text line regardless of how many columns the anchor spans.
    """
    if not atoms:
        return []

    # Sort top-to-bottom (descending y in PDF coords where y increases upward).
    by_y = sorted(atoms, key=lambda c: -(c.y0 + c.y1) / 2)

    clusters: list[list[Atom]] = [[by_y[0]]]
    band_y0 = by_y[0].y0
    band_y1 = by_y[0].y1

    for ch in by_y[1:]:
        overlap = min(ch.y1, band_y1) - max(ch.y0, band_y0)
        if overlap > 0:
            clusters[-1].append(ch)
            band_y0 = min(band_y0, ch.y0)
            band_y1 = max(band_y1, ch.y1)
        else:
            clusters.append([ch])
            band_y0 = ch.y0
            band_y1 = ch.y1

    return [bbox_from_atoms(cluster, page_width, page_height, origin_x, origin_y) for cluster in clusters]
