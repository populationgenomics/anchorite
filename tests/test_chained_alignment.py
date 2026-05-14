"""Tests for ``anchorite.chained_alignment``.

The algorithm is exercised against several shapes of input — pure
copies, copies with noise (B-side insertions), partial coverage,
multi-region disjoint matches, the table-cell stress pattern — using a
small custom alphabet so the assertions stay readable.
"""

from __future__ import annotations

import itertools
import string

import seq_smith

from anchorite.chained_alignment import chained_alignment

# Custom alphabet that covers ASCII lowercase, digits, and space.  Match
# = +1, mismatch = -1, which is the same shape as the package's
# ``SCORE_MATRIX_STRICT`` but kept local to the test so changes there
# can't silently break these.
_ALPHABET = string.ascii_lowercase + string.digits + " "
_SCORE = seq_smith.make_score_matrix(_ALPHABET, +1, -1)


def _enc(s: str) -> bytes:
    return seq_smith.encode(s, _ALPHABET)


def _matched_a(pairs: list[tuple[int, int]]) -> set[int]:
    return {a for a, _ in pairs}


def _matched_b(pairs: list[tuple[int, int]]) -> set[int]:
    return {b for _, b in pairs}


def _assert_monotonic_a(pairs: list[tuple[int, int]]) -> None:
    """The trace must be sorted on the A axis (the public contract)."""
    for prev, curr in itertools.pairwise(pairs):
        assert prev[0] <= curr[0], f"non-monotonic A coordinate: {prev} -> {curr}"


def _assert_pair_within_bounds(
    pairs: list[tuple[int, int]],
    n_a: int,
    n_b: int,
) -> None:
    for a, b in pairs:
        assert 0 <= a < n_a, f"a={a} out of range [0, {n_a})"
        assert 0 <= b < n_b, f"b={b} out of range [0, {n_b})"


# ---------------------------------------------------------------------------
# Trivial / boundary cases
# ---------------------------------------------------------------------------


def test_empty_a_returns_empty() -> None:
    assert chained_alignment(b"", _enc("hello"), _SCORE) == []


def test_empty_b_returns_empty() -> None:
    assert chained_alignment(_enc("hello"), b"", _SCORE) == []


def test_both_empty_returns_empty() -> None:
    assert chained_alignment(b"", b"", _SCORE) == []


def test_no_common_content_returns_empty() -> None:
    a = _enc("abcdefghij")
    b = _enc("9876543210")
    pairs = chained_alignment(a, b, _SCORE)
    # Possible to get a stray 1-byte coincidence, but nothing substantive.
    assert len(pairs) < 3


# ---------------------------------------------------------------------------
# Identity and near-identity
# ---------------------------------------------------------------------------


def test_identical_sequences_match_every_position() -> None:
    text = _enc("the quick brown fox jumps over the lazy dog")
    pairs = chained_alignment(text, text, _SCORE)

    assert pairs
    _assert_monotonic_a(pairs)
    _assert_pair_within_bounds(pairs, len(text), len(text))
    # Identity → every A position appears, paired with its own index.
    assert _matched_a(pairs) == set(range(len(text)))
    for a, b in pairs:
        assert a == b


def test_b_subset_of_a_via_b_insertions() -> None:
    """*B* covers *A* fully but has extra bytes interspersed.

    Models the markdown-vs-PDF case where markdown carries pipes /
    asterisks / hashes that collapse to spaces but still occupy bytes,
    so *B* is slightly longer than *A* while every *A* byte still has a
    counterpart.
    """
    a = _enc("hello world the quick brown fox jumps over the lazy dog")
    b = _enc("hello  world  the  quick  brown  fox  jumps  over  the  lazy  dog")
    pairs = chained_alignment(a, b, _SCORE)

    _assert_monotonic_a(pairs)
    _assert_pair_within_bounds(pairs, len(a), len(b))
    # Every non-space A byte is in the trace.  Spaces sometimes fold
    # into the double-space stretches; not required to assert on them.
    non_space_a = {i for i, c in enumerate(a) if chr(c) != " "}
    assert non_space_a.issubset(_matched_a(pairs))


def test_a_has_noise_b_anchors_full_content() -> None:
    """*A* has stretches that *B* doesn't transcribe (running heads).

    The trace should match the shared content; the *A*-only stretches
    are unclaimed — that's the denoising signal.
    """
    a = _enc("running head hello world the quick brown fox page 1 of 3")
    b = _enc("hello world the quick brown fox")
    pairs = chained_alignment(a, b, _SCORE)

    assert pairs
    _assert_monotonic_a(pairs)
    matched = _matched_a(pairs)
    # The shared content "hello world the quick brown fox" sits at A
    # offset 13–44.  Each of its bytes must be claimed.
    start = a.index(_enc("hello"))
    end = start + len(_enc("hello world the quick brown fox"))
    for i in range(start, end):
        assert i in matched, f"unmatched a={i} ({chr(a[i])!r})"
    # And the "running head" prefix bytes must NOT be claimed.
    assert not any(0 <= i < 13 for i in matched)


# ---------------------------------------------------------------------------
# Multi-region matches — where chaining is meaningful
# ---------------------------------------------------------------------------


def test_two_disjoint_regions_both_get_matched() -> None:
    """A pair of widely-separated matching regions should both end up in
    the chain — neither is dropped in favour of the other."""
    a = _enc(
        "first interesting region appears here "
        "then a lot of irrelevant filler content nobody asked for "
        "second meaningful region shows up at the end",
    )
    b = _enc("first interesting region second meaningful region")
    pairs = chained_alignment(a, b, _SCORE)

    matched_a = _matched_a(pairs)
    first_start = a.index(_enc("first"))
    first_end = first_start + len(_enc("first interesting region"))
    second_start = a.index(_enc("second"))
    second_end = second_start + len(_enc("second meaningful region"))

    for i in range(first_start, first_end):
        assert i in matched_a, f"first region unmatched at a={i}"
    for i in range(second_start, second_end):
        assert i in matched_a, f"second region unmatched at a={i}"


def test_order_reversal_picks_one_branch() -> None:
    """If *B* presents two regions in opposite order from *A*, the chain
    has to drop one — SW alignment is monotonic."""
    a = _enc("alpha bravo charlie delta echo foxtrot golf hotel india")
    # B has the regions reversed: "india ... hotel ... alpha bravo".
    b = _enc("india hotel alpha bravo")
    pairs = chained_alignment(a, b, _SCORE)

    matched_a = _matched_a(pairs)
    # Exactly one of the two clusters should be present (whichever the
    # chain picked).  At least one and at most one.
    alpha_start = a.index(_enc("alpha"))
    india_start = a.index(_enc("india"))
    has_alpha = any(alpha_start <= i < alpha_start + 5 for i in matched_a)
    has_india = any(india_start <= i < india_start + 5 for i in matched_a)
    assert has_alpha != has_india, "expected exactly one of the two reversed regions to be in the chain"


# ---------------------------------------------------------------------------
# Table-cell stress pattern (the original bug)
# ---------------------------------------------------------------------------


def test_short_cells_between_anchored_paragraphs() -> None:
    """Reproduces the bug shape without depending on a real PDF.

    *A* = "intro paragraph ... [row of short cells] ... outro paragraph".
    *B* = same content but as it would appear after markdown
    normalisation: paragraphs intact, table row as a sequence of short
    tokens.

    The two paragraphs serve as chain anchors.  The row's short cells
    must end up matched too — that's the property the per-segment
    design fails on.
    """
    a = _enc(
        "introduction paragraph that explains the cohort and methodology "
        "31 m 28 10 syncope 14174 y4725c ct trio de novo none none "
        "discussion paragraph summarising findings and clinical impact",
    )
    b = _enc(
        "introduction paragraph that explains the cohort and methodology "
        "31 m 28 10 syncope 14174 y4725c ct trio de novo none none "
        "discussion paragraph summarising findings and clinical impact",
    )
    pairs = chained_alignment(a, b, _SCORE)

    matched_a = _matched_a(pairs)
    row_start = a.index(_enc("31 m 28"))
    row_end = a.index(_enc(" discussion"))
    # Every byte of the table row must be matched — the bug case.
    for i in range(row_start, row_end):
        assert i in matched_a, f"unmatched a={i} ({chr(a[i])!r}) in table row region"


def test_all_short_cells_row_via_context() -> None:
    """Stress: a row consisting *only* of cells too short to seed.

    Each cell is at most 4 bytes — well below any reasonable seed
    threshold.  The chain has nothing to anchor on inside the row, but
    surrounding paragraph anchors confine the row to a tight window
    where local SW resolves the sequence-level coherence.
    """
    a = _enc(
        "narrative content before the difficult row appears in the document "
        "31 m 28 10 ct 19 f 22 8 pa "
        "narrative content after the difficult row continues to be readable",
    )
    b = _enc(
        "narrative content before the difficult row appears in the document "
        "31 m 28 10 ct 19 f 22 8 pa "
        "narrative content after the difficult row continues to be readable",
    )
    pairs = chained_alignment(a, b, _SCORE)

    matched_a = _matched_a(pairs)
    row_start = a.index(_enc("31 m 28"))
    row_end = a.index(_enc(" narrative content after"))
    for i in range(row_start, row_end):
        assert i in matched_a, f"unmatched a={i} ({chr(a[i])!r}) — all-short-cells row dropped"


# ---------------------------------------------------------------------------
# Output invariants
# ---------------------------------------------------------------------------


def test_output_is_sorted_by_a_index() -> None:
    a = _enc("the same content twice " * 5)
    b = _enc("the same content twice")
    pairs = chained_alignment(a, b, _SCORE)
    _assert_monotonic_a(pairs)


def test_output_indices_within_bounds() -> None:
    a = _enc("alpha beta gamma delta epsilon zeta")
    b = _enc("beta gamma delta")
    pairs = chained_alignment(a, b, _SCORE)
    _assert_pair_within_bounds(pairs, len(a), len(b))


def test_chain_respects_b_monotonicity() -> None:
    """The chain must be non-decreasing on the B axis too — even though
    the output is sorted on A, B indices for matched bytes should
    almost always be non-decreasing across the trace.

    Strictly: SW within a single gap can produce locally non-monotonic
    fragments if the gap has structural reversals (rare in practice).
    We assert global non-decrease at the *chain anchor* boundaries
    indirectly by checking that across a clean linear input, B indices
    don't go backwards.
    """
    a = _enc("alpha beta gamma delta epsilon zeta eta theta iota kappa")
    b = a  # clean identity case
    pairs = chained_alignment(a, b, _SCORE)
    b_indices = [b_ for _, b_ in pairs]
    for prev, curr in itertools.pairwise(b_indices):
        assert prev <= curr, "B coordinate went backwards on identical input"
