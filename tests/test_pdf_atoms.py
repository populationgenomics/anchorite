"""Tests for the per-atom PDF extractor.

Covers the flat-string assembly rules (intra-line gap, line-break space
insertion, soft-hyphen reconnection), bbox helpers that convert atom spans
to normalised page coordinates, and the page-rotation transform applied at
extraction time.  Live PDF extraction (``extract_page_atoms``,
``extract_page_data``) is also exercised end-to-end via
``test_md_association`` and ``test_pdf_index``.
"""

from __future__ import annotations

import ctypes
import io

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from anchorite import pdf_atoms
from anchorite.pdf_atoms import Atom, bbox_from_atoms, build_atom_index, extract_page_data, line_bboxes


def _line(text: str, *, baseline: float, x0: float = 0.0, font_size: float = 10.0) -> list[Atom]:
    """Build a sequence of ``Atom`` records on one visual line."""
    atoms = []
    cursor = x0
    for c in text:
        atoms.append(
            Atom(
                text=c,
                x0=cursor,
                y0=baseline,
                x1=cursor + font_size * 0.6,
                y1=baseline + font_size,
                font_size=font_size,
            ),
        )
        cursor += font_size * 0.6
    return atoms


class TestBuildAtomIndex:
    def test_line_break_inserts_space(self) -> None:
        # ``we`` at the end of one line, ``identified`` at the start of the
        # next.  Without the line-break space the flat string would read
        # ``weidentified``.
        atoms = _line("we", baseline=100.0) + _line("identified", baseline=80.0)
        ai = build_atom_index(atoms)
        assert ai.flat_str == "we identified"

    def test_horizontal_word_gap_inserts_space(self) -> None:
        # Two words on the same line with a horizontal gap > 20 % of font size.
        atoms_a = _line("hello", baseline=100.0)
        atoms_b = _line("world", baseline=100.0, x0=atoms_a[-1].x1 + 5.0)
        ai = build_atom_index(atoms_a + atoms_b)
        assert ai.flat_str == "hello world"

    def test_soft_hyphen_at_line_break_reconnects(self) -> None:
        # Typeset ``induc-`` at the end of one line, ``tion`` at the start of
        # the next — this is a soft-hyphenated word that should reconnect to
        # ``induction`` (matching the Markdown's un-hyphenated form).
        atoms = _line("induc-", baseline=100.0) + _line("tion", baseline=80.0)
        ai = build_atom_index(atoms)
        assert ai.flat_str == "induction"

    def test_hyphen_at_line_break_after_digit_keeps_hyphen(self) -> None:
        # Numeric range ``2009-`` followed by ``2010`` on the next line.  The
        # surrounding atoms aren't alphabetic, so the hyphen-suppression
        # heuristic must NOT fire — a space is inserted as for any line break.
        atoms = _line("2009-", baseline=100.0) + _line("2010", baseline=80.0)
        ai = build_atom_index(atoms)
        assert ai.flat_str == "2009- 2010"

    def test_hyphen_at_line_break_before_digit_keeps_hyphen(self) -> None:
        # Hyphenated identifier ``cohort-`` followed by ``38`` on the next
        # line.  The next atom isn't alphabetic, so we keep the hyphen.
        atoms = _line("cohort-", baseline=100.0) + _line("38", baseline=80.0)
        ai = build_atom_index(atoms)
        assert ai.flat_str == "cohort- 38"

    def test_mid_line_hyphen_unaffected(self) -> None:
        # ``e-mail`` on a single line: the hyphen stays even though it sits
        # between two letters, because it isn't at a line break.
        atoms = _line("e-mail", baseline=100.0)
        ai = build_atom_index(atoms)
        assert ai.flat_str == "e-mail"


class TestBboxFromAtomsOrigin:
    def test_zero_origin_matches_no_origin(self) -> None:
        # Default origin (0, 0) reproduces the unshifted result.
        atoms = _line("hello", baseline=100.0, x0=50.0)
        bbox = bbox_from_atoms(atoms, page_width=600.0, page_height=800.0)
        bbox_zero = bbox_from_atoms(
            atoms,
            page_width=600.0,
            page_height=800.0,
            origin_x=0.0,
            origin_y=0.0,
        )
        assert bbox == bbox_zero

    def test_origin_shift_makes_bbox_page_relative(self) -> None:
        # PDFs with non-zero mediabox origin must subtract that origin so the
        # 0-1000 normalised coords are relative to the page, not the
        # absolute PDF coordinate space.
        atoms = _line("hello", baseline=100.0, x0=50.0)
        unshifted = bbox_from_atoms(atoms, page_width=600.0, page_height=800.0)
        shifted = bbox_from_atoms(
            atoms,
            page_width=600.0,
            page_height=800.0,
            origin_x=50.0,
            origin_y=100.0,
        )
        # The shifted bbox should equal the unshifted bbox computed against
        # atoms whose absolute coords were already pre-subtracted.
        atoms_pre = _line("hello", baseline=0.0, x0=0.0)
        expected = bbox_from_atoms(atoms_pre, page_width=600.0, page_height=800.0)
        assert shifted == expected
        assert shifted != unshifted

    def test_line_bboxes_threads_origin(self) -> None:
        # ``line_bboxes`` must propagate the origin to ``bbox_from_atoms``
        # for each line cluster.
        atoms = _line("a", baseline=100.0, x0=50.0) + _line("b", baseline=80.0, x0=50.0)
        boxes = line_bboxes(
            atoms,
            page_width=600.0,
            page_height=800.0,
            origin_x=50.0,
            origin_y=70.0,
        )
        assert len(boxes) == 2
        for box in boxes:
            assert box.left == 0  # x0 (50) - origin_x (50) = 0


def test_module_exports_public_surface() -> None:
    # Paranoia: the names the README and other modules import must exist.
    assert hasattr(pdf_atoms, "Atom")
    assert hasattr(pdf_atoms, "AtomIndex")
    assert hasattr(pdf_atoms, "PageData")
    assert hasattr(pdf_atoms, "extract_page_data")
    assert hasattr(pdf_atoms, "extract_page_atoms")
    assert hasattr(pdf_atoms, "build_atom_index")
    assert hasattr(pdf_atoms, "bbox_from_atoms")
    assert hasattr(pdf_atoms, "line_bboxes")


# ---------------------------------------------------------------------------
# Rotation: atom-coord normalisation
# ---------------------------------------------------------------------------

# Unrotated Letter page; one glyph near the visual top-left in PDF coords.
_TEST_PAGE_W, _TEST_PAGE_H = 612.0, 792.0
_GLYPH_X, _GLYPH_Y = 50.0, 740.0  # bottom-left of the 'X' bbox in PDF user space


def _make_single_glyph_pdf(rotation: int) -> bytes:
    """Build a Letter-portrait PDF containing one glyph 'X' at PDF (50, 740).

    ``rotation`` is the /Rotate value to set (0, 90, 180, 270).  The glyph
    sits near the visual top-left of the *unrotated* page; the test asserts
    that after extraction the atom lands in the expected quadrant of the
    rotated/displayed frame for each rotation.
    """
    doc = pdfium.PdfDocument.new()
    page = doc.new_page(_TEST_PAGE_W, _TEST_PAGE_H)
    text_obj = pdfium_c.FPDFPageObj_NewTextObj(doc.raw, b"Helvetica", 12.0)
    buf = (ctypes.c_ushort * 2)(ord("X"), 0)
    pdfium_c.FPDFText_SetText(text_obj, buf)
    matrix = pdfium_c.FS_MATRIX(1.0, 0.0, 0.0, 1.0, _GLYPH_X, _GLYPH_Y)
    pdfium_c.FPDFPageObj_SetMatrix(text_obj, ctypes.byref(matrix))
    pdfium_c.FPDFPage_InsertObject(page, text_obj)
    pdfium_c.FPDFPage_GenerateContent(page)
    if rotation:
        pdfium_c.FPDFPage_SetRotation(page, rotation // 90)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


class TestExtractRotation:
    def test_unrotated_page_keeps_native_coords(self) -> None:
        doc = pdfium.PdfDocument(_make_single_glyph_pdf(rotation=0))
        pd = extract_page_data(doc)[0]
        assert pd.rotation == 0
        assert pd.width == _TEST_PAGE_W
        assert pd.height == _TEST_PAGE_H
        assert len(pd.atoms) == 1
        atom = pd.atoms[0]
        # No rotation, no mediabox offset: atom keeps its native bbox
        # (with a small side-bearing offset PDFium applies to the glyph).
        assert abs(atom.x0 - _GLYPH_X) < 1.0
        assert abs(atom.y0 - _GLYPH_Y) < 1.0

    def test_rotate_90_maps_top_left_unrotated_to_top_right_rotated(self) -> None:
        doc = pdfium.PdfDocument(_make_single_glyph_pdf(rotation=90))
        pd = extract_page_data(doc)[0]
        assert pd.rotation == 90
        # /Rotate=90 swaps dims: rotated W = unrotated H, rotated H = unrotated W.
        assert pd.width == _TEST_PAGE_H  # 792
        assert pd.height == _TEST_PAGE_W  # 612
        atom = pd.atoms[0]
        # Visual top-left of the unrotated page → top-right of the rotated frame.
        # Rotated x ≈ unrotated y (≈ 740 → near pd.width=792).
        assert atom.x0 > pd.width * 0.9
        assert atom.x1 <= pd.width
        # Rotated y ≈ unrotated (W − x) (≈ 612 − 50 = 562 → near pd.height=612).
        assert atom.y0 > pd.height * 0.9
        assert atom.y1 <= pd.height

    def test_rotate_180_maps_top_left_unrotated_to_bottom_right_rotated(self) -> None:
        doc = pdfium.PdfDocument(_make_single_glyph_pdf(rotation=180))
        pd = extract_page_data(doc)[0]
        assert pd.rotation == 180
        # /Rotate=180 preserves dims.
        assert pd.width == _TEST_PAGE_W
        assert pd.height == _TEST_PAGE_H
        atom = pd.atoms[0]
        # 180° flip: top-left unrotated → bottom-right rotated.
        assert atom.x0 > pd.width * 0.9
        assert atom.y0 < pd.height * 0.1

    def test_rotate_270_maps_top_left_unrotated_to_bottom_left_rotated(self) -> None:
        doc = pdfium.PdfDocument(_make_single_glyph_pdf(rotation=270))
        pd = extract_page_data(doc)[0]
        assert pd.rotation == 270
        assert pd.width == _TEST_PAGE_H  # 792
        assert pd.height == _TEST_PAGE_W  # 612
        atom = pd.atoms[0]
        # 270° CW: top-left unrotated → bottom-left rotated.
        assert atom.x0 < pd.width * 0.1
        assert atom.y0 < pd.height * 0.1

    def test_rotation_keeps_coords_in_page_box(self) -> None:
        # The atom must always sit inside [0, pd.width] × [0, pd.height] —
        # the pre-fix bug produced atom coords outside this box for /Rotate=90.
        for rot in (0, 90, 180, 270):
            doc = pdfium.PdfDocument(_make_single_glyph_pdf(rotation=rot))
            pd = extract_page_data(doc)[0]
            atom = pd.atoms[0]
            assert 0.0 <= atom.x0 <= pd.width
            assert 0.0 <= atom.x1 <= pd.width
            assert 0.0 <= atom.y0 <= pd.height
            assert 0.0 <= atom.y1 <= pd.height
