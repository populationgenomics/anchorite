"""Seed-and-extend alignment of two byte sequences.

Builds an order-preserving correspondence between two normalised byte
sequences (a reference *A* and a query *B*) by composing three classical
sequence-alignment ideas:

1. **Seeding.**  Find the top-scoring non-overlapping ungapped local
   alignments (HSPs) between *A* and *B* using ``seq_smith``.  These are
   high-confidence anchors.
2. **Chaining.**  Select a subset of those seeds that is monotonically
   increasing on *both* axes (LIS-shaped), weighted by seed length, to
   form a *chain* — a skeleton of the global trace.
3. **Gap filling.**  Between adjacent chain seeds (and at the document
   boundaries), run a small Smith-Waterman local alignment to recover
   matches the seed step's score floor was too coarse to catch.  Within
   each gap, repeat the seed+chain+fill recursion until either the gap
   is small enough that one local SW suffices or no significant seed
   remains.

The output is the set of ``(a_idx, b_idx)`` pairs representing every
matched byte across the whole trace, sorted by ``a_idx``.

This shape is BLAST/minimap2's seed-and-extend at a small scale.  It's
significantly cheaper than full Smith-Waterman on long sequences while
remaining principled — no per-segment thresholds, no granularity
mismatches.  It is intentionally agnostic of the source domain: it works
on any pair of normalised byte sequences and any ``seq_smith`` score
matrix, so callers other than ``PdfIndex`` (e.g. a future redesign of
``md_association.associate``) can adopt the same primitive.

Cost model
----------

Let ``|A| = n``, ``|B| = m``.  ``seq_smith.top_k_ungapped_local_align``
is O(n·m) in the worst case (diagonal scan), but with a small constant
because it does no DP traceback.  Chaining via O(k²) weighted LIS over
``k`` seeds is trivial for typical ``k ≤ 500``.  Gap-fill local SW is
O(gap_a · gap_b) per gap; the recursion keeps gap sizes small.  In
practice the runtime is dominated by the top-level seed scan.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import seq_smith

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
# These defaults are calibrated for text-alphabet sequences (the
# ``normalize_strict`` / ``normalize_loose`` outputs in
# ``anchorite.normalize``) where match=+1 / mismatch=-1.  Other alphabets
# may want different defaults — pass them explicitly to ``chained_alignment``.

# Maximum number of seeds returned by the top-K HSP search at the top
# level.  Bounded by ``min(|A|, |B|) / seed_min_score`` in practice
# (filter_overlap_a/b is True, so seeds can't overlap on either axis).
_DEFAULT_MAX_SEEDS = 500

# Minimum HSP score to qualify as a seed.  +1/-1 scoring on a ~37-byte
# alphabet means a length-8 exact stretch is extremely unlikely to be
# coincidental, and shorter stretches are likely to appear by chance.
_DEFAULT_SEED_MIN_SCORE = 8

# Score floor for a gap-fill local alignment.  Below this the fill is
# likely noise (short coincidental matches) — drop it.
_DEFAULT_FILL_MIN_SCORE = 4

# Max recursion depth.  Each level lowers the seed-score floor; once the
# floor reaches its minimum or no seed survives, gap-fill falls through
# to a single local SW.  Three levels is plenty for any realistic input.
_MAX_RECURSION_DEPTH = 3


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Seed:
    """One ungapped HSP between *A* and *B*, sliced from a ``seq_smith`` HSP.

    ``a_start`` / ``b_start`` are positions in the *outer* sequences, not
    in any sliced sub-window — callers walk gaps via slicing and add
    offsets when recursing.
    """

    a_start: int
    b_start: int
    length: int
    score: int

    @property
    def a_end(self) -> int:
        return self.a_start + self.length

    @property
    def b_end(self) -> int:
        return self.b_start + self.length


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chained_alignment(
    seq_a: bytes,
    seq_b: bytes,
    score_matrix: npt.NDArray[np.int32],
    *,
    gap_open: int = -2,
    gap_extend: int = -2,
    max_seeds: int = _DEFAULT_MAX_SEEDS,
    seed_min_score: int = _DEFAULT_SEED_MIN_SCORE,
    fill_min_score: int = _DEFAULT_FILL_MIN_SCORE,
) -> list[tuple[int, int]]:
    """Return the set of matched ``(a_idx, b_idx)`` pairs across the trace.

    Aligns *seq_b* against *seq_a* using seed-and-extend with chained HSPs:
    finds non-overlapping high-scoring ungapped seeds, chains the
    monotonic subset that maximises total seed length, and runs local
    Smith-Waterman in the gaps between (and around) chain anchors to
    recover shorter matches.

    Args:
        seq_a: First normalised byte sequence (the "reference").
        seq_b: Second normalised byte sequence (the "query").  Symmetry
            note: the algorithm is symmetric in *A* and *B*, but the
            return convention is ``(a_idx, b_idx)`` ordered by ``a_idx``.
        score_matrix: ``seq_smith`` score matrix.  Used uniformly across
            the seed scan and every gap-fill SW.
        gap_open: Gap-open penalty for gap-fill SW.  Negative.
        gap_extend: Gap-extend penalty.  Negative.
        max_seeds: Upper bound on seeds returned by the top-K HSP search
            at each recursion level.
        seed_min_score: Minimum HSP score for an HSP to qualify as a
            chain seed at the top level.  Lower at deeper recursion
            levels (see ``_seed_min_score_at_depth``).
        fill_min_score: Minimum score for a gap-fill local SW alignment
            to be accepted.  Below this the gap's best alignment is
            treated as noise.

    Returns:
        Sorted list of ``(a_idx, b_idx)`` pairs.  Each pair is one byte
        of *A* matched to one byte of *B* in the alignment trace.  The
        list is monotonic on ``a_idx`` (by construction) and almost
        always monotonic on ``b_idx`` too, modulo the rare case where a
        gap-fill SW's traceback produces locally non-monotonic
        fragments inside one gap.

        Empty list if either input is empty or no significant alignment
        is found.
    """
    if not seq_a or not seq_b:
        return []
    pairs = _align(
        seq_a,
        seq_b,
        score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        max_seeds=max_seeds,
        seed_min_score=seed_min_score,
        fill_min_score=fill_min_score,
        depth=0,
    )
    pairs.sort()
    return pairs


# ---------------------------------------------------------------------------
# Recursion core
# ---------------------------------------------------------------------------


def _seed_min_score_at_depth(top_level_min: int, depth: int) -> int:
    """Lower the seed-score floor at each recursion level.

    At depth 0 we want highly specific seeds (long, high-confidence
    matches).  At deeper levels we're inside small gaps between top-
    level anchors; a shorter seed is acceptable because the bounded
    context already constrains where it can land.  Floor at 3 to avoid
    chaining trivial 2-byte coincidences.
    """
    return max(3, top_level_min - 2 * depth)


def _align(  # noqa: PLR0913
    seq_a: bytes,
    seq_b: bytes,
    score_matrix: npt.NDArray[np.int32],
    *,
    gap_open: int,
    gap_extend: int,
    max_seeds: int,
    seed_min_score: int,
    fill_min_score: int,
    depth: int,
) -> list[tuple[int, int]]:
    """Inner recursive worker for ``chained_alignment``.

    Returns matched pairs in *local* coordinates (relative to the slices
    passed in).  The caller shifts by the slice offsets when composing
    results.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return []

    # Recursion budget exhausted: stop seeding, just take the best
    # single local alignment for what's left.  Prevents pathological
    # recursion on adversarial inputs where seeds keep splitting the
    # range without ever resolving.
    if depth >= _MAX_RECURSION_DEPTH:
        return _local_sw_pairs(
            seq_a,
            seq_b,
            score_matrix,
            gap_open=gap_open,
            gap_extend=gap_extend,
            min_score=fill_min_score,
        )

    # Step 1: seed scan.  Cheap (linear in n·m with a small constant)
    # and always preferable to a single SW because it exposes the
    # multi-region structure of the input.
    seeds = _find_seeds(
        seq_a,
        seq_b,
        score_matrix,
        max_seeds=max_seeds,
        min_score=_seed_min_score_at_depth(seed_min_score, depth),
    )

    if not seeds:
        # No reliable seeds at this score floor — fall through to a
        # single local SW.  Catches strong contiguous matches the
        # floor was too coarse to admit.
        return _local_sw_pairs(
            seq_a,
            seq_b,
            score_matrix,
            gap_open=gap_open,
            gap_extend=gap_extend,
            min_score=fill_min_score,
        )

    # Step 2: chain via weighted LIS (weight = seed length, so longer
    # seeds bias the chain selection).
    chain = _weighted_lis(seeds)

    # Step 3: emit chain matches and recurse into each inter-seed gap.
    pairs: list[tuple[int, int]] = []
    prev_a, prev_b = 0, 0
    for seed in chain:
        if seed.a_start > prev_a and seed.b_start > prev_b:
            sub = _align(
                seq_a[prev_a : seed.a_start],
                seq_b[prev_b : seed.b_start],
                score_matrix,
                gap_open=gap_open,
                gap_extend=gap_extend,
                max_seeds=max_seeds,
                seed_min_score=seed_min_score,
                fill_min_score=fill_min_score,
                depth=depth + 1,
            )
            for a, b in sub:
                pairs.append((prev_a + a, prev_b + b))
        for k in range(seed.length):
            pairs.append((seed.a_start + k, seed.b_start + k))
        prev_a, prev_b = seed.a_end, seed.b_end

    # Tail gap.
    if prev_a < n and prev_b < m:
        sub = _align(
            seq_a[prev_a:],
            seq_b[prev_b:],
            score_matrix,
            gap_open=gap_open,
            gap_extend=gap_extend,
            max_seeds=max_seeds,
            seed_min_score=seed_min_score,
            fill_min_score=fill_min_score,
            depth=depth + 1,
        )
        for a, b in sub:
            pairs.append((prev_a + a, prev_b + b))

    return pairs


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def _find_seeds(
    seq_a: bytes,
    seq_b: bytes,
    score_matrix: npt.NDArray[np.int32],
    *,
    max_seeds: int,
    min_score: int,
) -> list[_Seed]:
    """Top-K HSPs above the minimum score threshold, disjoint on the *A* axis.

    Uses ``seq_smith.top_k_ungapped_local_align`` with overlap filtering
    on the *A* axis only.  Disjoint-on-A means each *A* byte is claimed
    by at most one seed — the property the chain step needs to avoid
    double-counting matches.

    Filtering on the *B* axis is *off* deliberately: real near-copy
    inputs have repeated short fragments on *B* (e.g. a paragraph break
    space that bookends two adjacent matching regions on *A*), and the
    *B*-axis filter would arbitrarily drop one of the two strong seeds
    that share that fragment.  Allowing *B*-overlap means two seeds can
    legitimately claim the same *B* position from different *A*
    positions; the chain step orders them by *B*-start and the per-pair
    output keeps each ``(a_idx, b_idx)`` distinct.
    """
    if len(seq_a) == 0 or len(seq_b) == 0:
        return []
    hsps = seq_smith.top_k_ungapped_local_align(
        seq_a,
        seq_b,
        score_matrix,
        k=max_seeds,
        filter_overlap_a=True,
        filter_overlap_b=False,
    )
    seeds: list[_Seed] = []
    for hsp in hsps:
        if hsp.score < min_score:
            continue
        # An ungapped HSP has exactly one Match fragment.
        frag = hsp.fragments[0]
        seeds.append(
            _Seed(
                a_start=frag.sa_start,
                b_start=frag.sb_start,
                length=frag.len,
                score=hsp.score,
            ),
        )
    return seeds


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------


def _weighted_lis(seeds: list[_Seed]) -> list[_Seed]:
    """Pick the chain of seeds maximising total seed length.

    Seeds are sorted by ``a_start`` and we find the maximum-weight
    increasing subsequence such that adjacent seeds satisfy *both*:

    * ``sj.a_end <= si.a_start`` — disjoint on the *A* axis.  Guaranteed
      by ``filter_overlap_a=True`` at the seed step, but stated here
      explicitly so the chain contract is self-evident.
    * ``sj.b_start < si.b_start`` — monotonic on the *B* axis.  Strict
      inequality only; a small ``B``-overlap between adjacent chain
      seeds (e.g. a shared whitespace byte) is fine because per-pair
      emission keeps ``(a_idx, b_idx)`` tuples distinct downstream.

    O(n²) DP — fine for ``n ≤ 500``.  The straightforward fenwick-tree
    O(n log n) variant is available if profiling ever flags this.
    """
    if not seeds:
        return []
    sorted_seeds = sorted(seeds, key=lambda s: (s.a_start, s.b_start))
    n = len(sorted_seeds)
    # best[i] = max total weight of a chain ending at seed i
    # prev[i] = predecessor in that chain, or -1
    best = [s.length for s in sorted_seeds]
    prev = [-1] * n
    for i in range(n):
        si = sorted_seeds[i]
        for j in range(i):
            sj = sorted_seeds[j]
            # j must precede i: disjoint on A, strictly earlier on B.
            if sj.a_end <= si.a_start and sj.b_start < si.b_start:
                candidate = best[j] + si.length
                if candidate > best[i]:
                    best[i] = candidate
                    prev[i] = j
    end = max(range(n), key=lambda x: best[x])
    chain: list[_Seed] = []
    while end != -1:
        chain.append(sorted_seeds[end])
        end = prev[end]
    chain.reverse()
    return chain


# ---------------------------------------------------------------------------
# Gap fill
# ---------------------------------------------------------------------------


def _local_sw_pairs(
    seq_a: bytes,
    seq_b: bytes,
    score_matrix: npt.NDArray[np.int32],
    *,
    gap_open: int,
    gap_extend: int,
    min_score: int,
) -> list[tuple[int, int]]:
    """Run one local SW and return ``(a_idx, b_idx)`` pairs from Match fragments.

    Returns ``[]`` if the alignment scores below ``min_score`` (the gap
    has no significant match) or if either input is empty.
    """
    if not seq_a or not seq_b:
        return []
    aln = seq_smith.local_align(seq_a, seq_b, score_matrix, gap_open, gap_extend)
    if aln.score < min_score:
        return []
    pairs: list[tuple[int, int]] = []
    for frag in aln.fragments:
        if frag.fragment_type != seq_smith.FragmentType.Match:
            continue
        for k in range(frag.len):
            pairs.append((frag.sa_start + k, frag.sb_start + k))
    return pairs
