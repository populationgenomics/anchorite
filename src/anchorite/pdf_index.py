"""PDF-quote alignment: resolve verbatim quotes to bounding boxes.

Given raw PDF bytes, ``PdfIndex`` extracts per-atom bounding boxes
from every page and builds a document-wide flat string for batched
Smith-Waterman alignment.  Quote resolution is then a cheap dictionary
lookup against the cached state.

Optionally accepts a Markdown transcription of the document at
construction time.  The markdown is aligned to the extracted PDF atoms
via ``chained_alignment`` — seed-and-extend with chained HSPs — and the
union of matched atoms rebuilds the cached flat string with atoms the
LLM didn't transcribe (running heads, page numbers, footnote markers)
dropped.  Character-level alignment means short fragments (e.g. table
cells of one or two digits) inherit position from their neighbouring
context and survive the cleanup even though they couldn't anchor
themselves.  The markdown is then discarded — the resolver only ever
sees PDF atoms.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import pypdfium2 as pdfium
import seq_smith

from .chained_alignment import chained_alignment
from .normalize import SCORE_MATRIX_STRICT, normalize_strict
from .pdf_atoms import Atom, PageData, build_atom_index, extract_page_data, line_bboxes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .anchors import BBox

logger = logging.getLogger(__name__)


# Default alignment scoring (anchorite-style integers for seq_smith).
_GAP_OPEN = -2
_GAP_EXTEND = -2

# Minimum alignment score to accept a match (≈15 matched chars).
_MIN_ALIGNMENT_SCORE = 15


@dataclasses.dataclass(frozen=True)
class _PageBBoxData:
    """Slim per-page state retained after flat-string assembly.

    ``PageData`` carries a per-page ``AtomIndex`` (``flat_str`` plus a
    ``flat_to_atom`` int list) needed to *build* the document-wide flat
    string.  Once that's done, ``_bboxes_for_alignment`` only needs the
    raw atoms and the page geometry — keep just those for the lifetime
    of the index.
    """

    atoms: list[Atom]
    width: float
    height: float

    @classmethod
    def from_page_data(cls, pd: PageData) -> _PageBBoxData:
        return cls(
            atoms=pd.atoms,
            width=pd.width,
            height=pd.height,
        )


def _build_index_from_atoms(
    page_data: list[PageData],
) -> tuple[str, list[int], list[int]]:
    """Build the markdown-free flat string from every extracted PDF atom.

    Returns ``(flat_str, flat_to_page, flat_to_page_atom)``.  Page boundaries
    are marked by a single space whose ``flat_to_page_atom`` entry is ``-1``.
    Within a page, the per-page flat string from ``build_atom_index`` is
    used verbatim — including its space-at-gap and soft-hyphen logic.
    """
    flat_parts: list[str] = []
    flat_to_page: list[int] = []
    flat_to_page_atom: list[int] = []
    for page_idx, pd in enumerate(page_data):
        if flat_parts:
            flat_parts.append(" ")
            flat_to_page.append(page_idx)
            flat_to_page_atom.append(-1)
        ai = pd.atom_index
        for i, c in enumerate(ai.flat_str):
            flat_parts.append(c)
            flat_to_page.append(page_idx)
            flat_to_page_atom.append(ai.flat_to_atom[i])
    return "".join(flat_parts), flat_to_page, flat_to_page_atom


def _claimed_atoms_per_page(
    full_flat: str,
    full_to_page: list[int],
    full_to_page_atom: list[int],
    markdown: str,
) -> dict[int, list[int]]:
    """Return ``{page_idx: sorted atom indices claimed by the markdown}``.

    Normalises both the full PDF flat string and the markdown, runs the
    seed-and-extend chained alignment between them, and projects matched
    ``pdf_norm`` positions back to per-page atom indices via the
    ``full_to_page`` / ``full_to_page_atom`` maps inherited from
    ``_build_index_from_atoms``.

    Inter-page separator positions (``full_to_page_atom < 0``) and
    ``build_atom_index``-inserted spaces (``full_to_page_atom`` points
    at the *preceding* atom) are handled by the mapping itself: any
    atom referenced via a matched position is added to the claim set,
    so a match landing on an inserted space resolves to the atom before
    the gap rather than vanishing.
    """
    md_norm, _ = normalize_strict(markdown, strip_html=True)
    pdf_norm, pdf_norm_to_flat = normalize_strict(full_flat)
    if not md_norm or not pdf_norm:
        return {}

    pairs = chained_alignment(
        pdf_norm,
        md_norm,
        SCORE_MATRIX_STRICT,
        gap_open=_GAP_OPEN,
        gap_extend=_GAP_EXTEND,
    )

    claimed: dict[int, set[int]] = {}
    for pdf_idx, _md_idx in pairs:
        flat_idx = pdf_norm_to_flat[pdf_idx]
        if flat_idx >= len(full_to_page_atom):
            continue  # sentinel position
        atom_idx = full_to_page_atom[flat_idx]
        if atom_idx < 0:
            continue  # inter-page separator — no atom to keep
        page = full_to_page[flat_idx]
        claimed.setdefault(page, set()).add(atom_idx)

    return {page: sorted(indices) for page, indices in claimed.items()}


def _build_index_from_alignment(
    page_data: list[PageData],
    markdown: str,
) -> tuple[str, list[int], list[int]]:
    """Build the cleaned flat string from atoms claimed by chained alignment.

    1. Build the full PDF flat string from every atom (the markdown-free
       index shape) so we have a single coordinate system to align in.
    2. Run ``chained_alignment`` between the normalised markdown and the
       normalised full flat string; this gives the matched ``(pdf, md)``
       byte pairs.
    3. Project matched PDF byte positions back to per-page atom indices.
    4. For each page in document order, rebuild the page's flat string
       from *only* the claimed atoms (via ``build_atom_index``, which
       re-inserts spaces at coordinate gaps so visually disjoint claimed
       atoms stay separated).
    5. Concatenate per-page flat strings with a single inter-page
       separator space (``flat_to_page_atom = -1``).

    Atoms not claimed by any matched byte are dropped — that's the
    denoise effect.  Because the alignment operates at the byte level,
    short fragments (e.g. table cells of one or two digits) survive
    based on sequence-level coherence with their neighbours, rather
    than needing to anchor themselves.

    Returns ``(flat_str, flat_to_page, flat_to_page_atom)``.
    """
    full_flat, full_to_page, full_to_page_atom = _build_index_from_atoms(page_data)
    claimed = _claimed_atoms_per_page(full_flat, full_to_page, full_to_page_atom, markdown)

    flat_parts: list[str] = []
    flat_to_page: list[int] = []
    flat_to_page_atom: list[int] = []

    for page_idx in range(len(page_data)):
        atom_indices = claimed.get(page_idx)
        if not atom_indices:
            continue

        if flat_parts:
            flat_parts.append(" ")
            flat_to_page.append(page_idx)
            flat_to_page_atom.append(-1)

        atoms_subset = [page_data[page_idx].atoms[j] for j in atom_indices]
        sub_ai = build_atom_index(atoms_subset)
        for i, c in enumerate(sub_ai.flat_str):
            flat_parts.append(c)
            flat_to_page.append(page_idx)
            # ``sub_ai.flat_to_atom`` indexes into ``atoms_subset``; map back
            # to the absolute atom index on ``page_data[page_idx].atoms`` via
            # ``atom_indices``.  Inserted-space positions inherit the
            # preceding atom's absolute index (matches ``build_atom_index``
            # behaviour for in-page atoms), so a match fragment that lands
            # on an inserted space resolves to the atom before the gap
            # rather than vanishing.
            flat_to_page_atom.append(atom_indices[sub_ai.flat_to_atom[i]])

    return "".join(flat_parts), flat_to_page, flat_to_page_atom


class PdfIndex:
    """Pre-extracted PDF atom data for batched quote→bbox resolution.

    Construction extracts per-atom bounding boxes from every page and
    builds a document-wide flat string.  This is the expensive step.  Once
    built, ``resolve`` is cheap and may be called repeatedly.

    When ``markdown`` is supplied, it is aligned to the extracted PDF atoms
    via ``chained_alignment`` (seed-and-extend with chained HSPs) and used
    to clean up the cached flat string — matched-only atoms in document
    order, with running heads / page numbers / footnote markers the LLM
    didn't transcribe dropped.  The markdown is not stored; after
    construction the index is markdown-free, and ``resolve`` matches
    LLM-emitted quotes against the cleaned PDF text.

    Pages in the returned ``(page, BBox)`` tuples are **0-indexed**, matching
    ``anchorite.Anchor.page``.

    Thread safety: construction is *not* thread-safe (PDFium isn't);
    serialise concurrent ``PdfIndex(...)`` calls in the caller.  ``.resolve``
    after construction is thread-safe — it touches only Python data and
    seq_smith (which manages its own threading via ``num_threads``).
    """

    def __init__(self, pdf_data: bytes, *, markdown: str | None = None) -> None:
        doc = pdfium.PdfDocument(pdf_data)
        page_data = extract_page_data(doc)

        if markdown is not None:
            flat_str, flat_to_page, flat_to_page_atom = _build_index_from_alignment(page_data, markdown)
        else:
            flat_str, flat_to_page, flat_to_page_atom = _build_index_from_atoms(page_data)

        # Drop the per-page ``AtomIndex`` now that the flat string is
        # assembled: ``resolve`` only needs atoms + geometry.  Long-lived
        # indices would otherwise carry a redundant per-page flat_str and
        # flat_to_atom list for no reason.
        self._page_data: list[_PageBBoxData] = [_PageBBoxData.from_page_data(pd) for pd in page_data]
        del page_data

        self._flat_str = flat_str
        self._flat_to_page = flat_to_page
        self._flat_to_page_atom = flat_to_page_atom

        # Normalise once for resolve().  ``norm_to_flat`` has a sentinel at
        # position ``len(flat_norm)`` for exclusive-end lookups.
        flat_norm, norm_to_flat = normalize_strict(flat_str)
        self._flat_norm = flat_norm
        self._norm_to_flat = norm_to_flat

    def resolve(
        self,
        quotes: Sequence[str],
        *,
        min_score: int = _MIN_ALIGNMENT_SCORE,
        num_threads: int | None = None,
    ) -> dict[str, list[tuple[int, BBox]]]:
        """Resolve verbatim quotes to ``(page, BBox)`` tuples.

        Each quote is normalised, batched together with the others, and
        Smith-Waterman local-global aligned against the cached document text
        in a single ``seq_smith.local_global_align_many`` call.  Matched
        characters are clustered into one ``BBox`` per visual line per page.

        Args:
            quotes: Verbatim strings to locate in the PDF.  Empty / whitespace
                quotes and quotes that score below ``min_score`` map to ``[]``.
            min_score: Minimum alignment score to accept a match.
            num_threads: Thread count for batched alignment; ``None`` defers
                to seq_smith's default.

        Returns:
            ``{quote: [(page_idx, bbox), ...]}`` for every input quote.
            Pages are 0-indexed.  Quotes appear as keys in the same form
            they were supplied; quotes that normalise to the same bytes
            (including identical inputs) share a single alignment and
            receive identical bbox lists.
        """
        results: dict[str, list[tuple[int, BBox]]] = {q: [] for q in quotes}

        if not quotes or not self._flat_str:
            return results

        # Dedup by normalised bytes: identical inputs and inputs that differ
        # only in characters the normaliser folds (case, whitespace, etc.)
        # share one alignment.  Each unique norm fans back out to every
        # input key it came from.
        norm_to_keys: dict[bytes, list[str]] = {}
        for quote in quotes:
            clean = quote.strip()
            if not clean:
                continue
            nq, _ = normalize_strict(clean)
            if nq:
                norm_to_keys.setdefault(nq, []).append(quote)

        if not norm_to_keys:
            return results

        unique_norms = list(norm_to_keys)
        alignments = seq_smith.local_global_align_many(
            self._flat_norm,
            unique_norms,
            SCORE_MATRIX_STRICT,
            _GAP_OPEN,
            _GAP_EXTEND,
            num_threads=num_threads,
        )

        for norm, aln in zip(unique_norms, alignments, strict=True):
            boxes = self._bboxes_for_alignment(aln, min_score)
            for key in norm_to_keys[norm]:
                results[key] = boxes

        return results

    def _bboxes_for_alignment(
        self,
        aln: seq_smith.Alignment,
        min_score: int,
    ) -> list[tuple[int, BBox]]:
        """Convert one alignment to ``(page, BBox)`` line tuples.

        Returns ``[]`` when ``aln.score < min_score``.  Match fragments are
        walked, each fragment's normalised range is mapped back through
        ``_norm_to_flat`` to the flat string, then to per-page atom indices,
        which are line-clustered to produce one BBox per visual line per
        page.
        """
        if aln.score < min_score:
            return []

        # Per page, the set of matched atom indices (into page_data[p].atoms).
        per_page_atoms: dict[int, set[int]] = {}
        for frag in aln.fragments:
            if frag.fragment_type != seq_smith.FragmentType.Match:
                continue
            flat_start = self._norm_to_flat[frag.sa_start]
            flat_end = self._norm_to_flat[frag.sa_start + frag.len]
            for fi in range(flat_start, min(flat_end, len(self._flat_to_page))):
                page = self._flat_to_page[fi]
                atom_idx = self._flat_to_page_atom[fi]
                if atom_idx < 0:
                    continue  # inter-segment / inter-page separator
                per_page_atoms.setdefault(page, set()).add(atom_idx)

        results: list[tuple[int, BBox]] = []
        for page in sorted(per_page_atoms):
            pd = self._page_data[page]
            atoms = [pd.atoms[i] for i in sorted(per_page_atoms[page])]
            boxes = line_bboxes(atoms, pd.width, pd.height)
            results.extend((page, b) for b in boxes)
        return results
