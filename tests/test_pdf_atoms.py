"""Tests for the per-atom PDF extractor.

Covers the flat-string assembly rules (intra-line gap, line-break space
insertion, soft-hyphen reconnection) and bbox helpers that convert atom
spans to normalised page coordinates.  Live PDF extraction (``extract_page_atoms``,
``extract_page_data``) is exercised end-to-end via ``test_md_association``
and ``test_pdf_index``.
"""

from anchorite import pdf_atoms
from anchorite.pdf_atoms import Atom, bbox_from_atoms, build_atom_index, line_bboxes


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
