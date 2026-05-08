"""PDF-quote alignment: resolve verbatim quotes to bounding boxes.

Given raw PDF bytes, ``PdfIndex`` extracts per-character bounding boxes
from every page and builds a document-wide flat string for batched
Smith-Waterman alignment.  Quote resolution is then a cheap dictionary
lookup against the cached state.

Optionally accepts a Markdown transcription of the document at
construction time: the markdown is aligned against the extracted PDF
chars (sharing ``md_association``'s alignment surface), and the matched
chars rebuild the cached flat string in markdown order with chars the
LLM didn't transcribe (running heads, page numbers, footnote markers)
removed.  The markdown is then discarded — the resolver only ever sees
PDF chars.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import pypdfium2 as pdfium
import seq_smith

from . import md_association
from .normalize import SCORE_MATRIX_STRICT, normalize_strict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .anchors import BBox
    from .md_association import _Char

logger = logging.getLogger(__name__)


# Default alignment scoring (anchorite-style integers for seq_smith).
_GAP_OPEN = -2
_GAP_EXTEND = -2

# Minimum alignment score to accept a match (≈15 matched chars).
_MIN_ALIGNMENT_SCORE = 15


@dataclasses.dataclass(frozen=True)
class _PageBBoxData:
    """Slim per-page state retained after flat-string assembly.

    ``_PageData`` carries a per-page ``_CharIndex`` (``flat_str`` plus a
    ``flat_to_char`` int list) needed to *build* the document-wide flat
    string.  Once that's done, ``_bboxes_for_alignment`` only needs the
    raw chars and the page geometry — keep just those for the lifetime
    of the index.
    """

    chars: list[_Char]
    width: float
    height: float
    origin_x: float
    origin_y: float

    @classmethod
    def from_page_data(cls, pd: md_association._PageData) -> _PageBBoxData:
        return cls(
            chars=pd.chars,
            width=pd.width,
            height=pd.height,
            origin_x=pd.origin_x,
            origin_y=pd.origin_y,
        )


def _build_index_from_chars(
    page_data: list[md_association._PageData],
) -> tuple[str, list[int], list[int]]:
    """Build the markdown-free flat string from every extracted PDF char.

    Returns ``(flat_str, flat_to_page, flat_to_page_char)``.  Page boundaries
    are marked by a single space whose ``flat_to_page_char`` entry is ``-1``.
    Within a page, the per-char flat string from ``_build_char_index`` is
    used verbatim — including its space-at-gap and soft-hyphen logic.
    """
    flat_parts: list[str] = []
    flat_to_page: list[int] = []
    flat_to_page_char: list[int] = []
    for page_idx, pd in enumerate(page_data):
        if flat_parts:
            flat_parts.append(" ")
            flat_to_page.append(page_idx)
            flat_to_page_char.append(-1)
        ci = pd.char_index
        for i, c in enumerate(ci.flat_str):
            flat_parts.append(c)
            flat_to_page.append(page_idx)
            flat_to_page_char.append(ci.flat_to_char[i])
    return "".join(flat_parts), flat_to_page, flat_to_page_char


def _build_index_from_alignment(
    page_data: list[md_association._PageData],
    outcome: md_association._AlignmentOutcome,
) -> tuple[str, list[int], list[int]]:
    """Build the cleaned flat string from chars matched by markdown alignment.

    Walks segments in markdown order; for each matched segment, includes
    only the chars the alignment actually claimed, runs them back through
    ``_build_char_index`` for space-at-gap / soft-hyphen handling, and
    appends the result.  Unmatched segments and unmatched-on-a-page chars
    are dropped — that's the "denoise" effect: running heads, page numbers,
    footnote markers the LLM didn't transcribe never enter the cache.

    Segment separators: a single space between segments, with
    ``flat_to_page_char = -1``.

    Returns ``(flat_str, flat_to_page, flat_to_page_char)``.
    """
    flat_parts: list[str] = []
    flat_to_page: list[int] = []
    flat_to_page_char: list[int] = []

    for entry in outcome.matched_chars_per_segment:
        if entry is None:
            continue
        page, char_indices = entry
        if not char_indices:
            continue

        if flat_parts:
            flat_parts.append(" ")
            flat_to_page.append(page)
            flat_to_page_char.append(-1)

        chars_subset = [page_data[page].chars[j] for j in char_indices]
        sub_ci = md_association._build_char_index(chars_subset)  # noqa: SLF001
        for i, c in enumerate(sub_ci.flat_str):
            flat_parts.append(c)
            flat_to_page.append(page)
            # ``sub_ci.flat_to_char`` indexes into ``chars_subset``; map back
            # to the absolute char index on ``page_data[page].chars`` via
            # ``char_indices``.  Inserted-space positions inherit the
            # preceding char's absolute index (matches ``_build_char_index``
            # behaviour for in-page chars), so a match fragment that lands
            # on an inserted space resolves to the char before the gap
            # rather than vanishing.
            flat_to_page_char.append(char_indices[sub_ci.flat_to_char[i]])

    return "".join(flat_parts), flat_to_page, flat_to_page_char


class PdfIndex:
    """Pre-extracted PDF char data for batched quote→bbox resolution.

    Construction extracts per-character bounding boxes from every page and
    builds a document-wide flat string.  This is the expensive step.  Once
    built, ``resolve`` is cheap and may be called repeatedly.

    When ``markdown`` is supplied, it is aligned to the extracted PDF chars
    via ``md_association`` and used to clean up the cached flat string —
    matched-only chars in markdown order, with running heads / page numbers /
    footnote markers the LLM didn't transcribe dropped.  The markdown is
    not stored; after construction the index is markdown-free, and
    ``resolve`` matches LLM-emitted quotes against the cleaned PDF chars.

    Pages in the returned ``(page, BBox)`` tuples are **0-indexed**, matching
    ``anchorite.Anchor.page``.

    Thread safety: construction is *not* thread-safe (PDFium isn't);
    serialise concurrent ``PdfIndex(...)`` calls in the caller.  ``.resolve``
    after construction is thread-safe — it touches only Python data and
    seq_smith (which manages its own threading via ``num_threads``).
    """

    def __init__(self, pdf_data: bytes, *, markdown: str | None = None) -> None:
        doc = pdfium.PdfDocument(pdf_data)
        page_data = md_association._extract_page_data(doc)  # noqa: SLF001

        if markdown is not None:
            outcome = md_association._align_markdown_to_pages(page_data, markdown)  # noqa: SLF001
            flat_str, flat_to_page, flat_to_page_char = _build_index_from_alignment(page_data, outcome)
        else:
            flat_str, flat_to_page, flat_to_page_char = _build_index_from_chars(page_data)

        # Drop the per-page ``_CharIndex`` now that the flat string is
        # assembled: ``resolve`` only needs chars + geometry.  Long-lived
        # indices would otherwise carry a redundant per-page flat_str and
        # flat_to_char list for no reason.
        self._page_data: list[_PageBBoxData] = [_PageBBoxData.from_page_data(pd) for pd in page_data]
        del page_data

        self._flat_str = flat_str
        self._flat_to_page = flat_to_page
        self._flat_to_page_char = flat_to_page_char

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
        ``_norm_to_flat`` to the flat string, then to per-page char indices,
        which are line-clustered to produce one BBox per visual line per
        page.
        """
        if aln.score < min_score:
            return []

        # Per page, the set of matched char indices (into page_data[p].chars).
        per_page_chars: dict[int, set[int]] = {}
        for frag in aln.fragments:
            if frag.fragment_type != seq_smith.FragmentType.Match:
                continue
            flat_start = self._norm_to_flat[frag.sa_start]
            flat_end = self._norm_to_flat[frag.sa_start + frag.len]
            for fi in range(flat_start, min(flat_end, len(self._flat_to_page))):
                page = self._flat_to_page[fi]
                char_idx = self._flat_to_page_char[fi]
                if char_idx < 0:
                    continue  # inter-segment / inter-page separator
                per_page_chars.setdefault(page, set()).add(char_idx)

        results: list[tuple[int, BBox]] = []
        for page in sorted(per_page_chars):
            pd = self._page_data[page]
            chars = [pd.chars[i] for i in sorted(per_page_chars[page])]
            boxes = md_association._line_bboxes(  # noqa: SLF001
                chars,
                pd.width,
                pd.height,
                pd.origin_x,
                pd.origin_y,
            )
            results.extend((page, b) for b in boxes)
        return results
