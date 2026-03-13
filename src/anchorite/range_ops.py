import heapq
import itertools
from collections.abc import Callable, Iterator, Sequence
from operator import itemgetter

_INTERSECTION_OVERLAP_COUNT = 2


def _generate_range_edges(
    ranges: Sequence[tuple[int, int]],
    start_weight: int,
    end_weight: int,
) -> Iterator[tuple[int, int]]:
    """Generates edge events for a sweep-line algorithm.

    Yields (position, weight) pairs.
    """
    for s, e in ranges:
        yield s, start_weight
        yield e, end_weight


def _sweep_operation(
    ranges_a: Sequence[tuple[int, int]],
    ranges_b: Sequence[tuple[int, int]],
    weights_a: tuple[int, int],
    weights_b: tuple[int, int],
    predicate: Callable[[int], bool],
) -> list[tuple[int, int]]:
    """Generic sweep-line operation.

    Args:
        ranges_a: First sequence of intervals.
        ranges_b: Second sequence of intervals.
        weights_a: (start_weight, end_weight) for A.
        weights_b: (start_weight, end_weight) for B.
        predicate: Function taking current sum 's' and returning True if we should be outputting.

    Returns:
        List of resulting intervals.
    """
    # Merge sorted event streams
    events = heapq.merge(
        _generate_range_edges(ranges_a, *weights_a),
        _generate_range_edges(ranges_b, *weights_b),
        key=itemgetter(0),
    )

    s = 0
    result = []
    start_pos = -1

    # Group events by position to handle simultaneous events (e.g. abutments)
    for pos, group in itertools.groupby(events, key=itemgetter(0)):
        # Calculate total weight change at this position
        delta = sum(weight for _, weight in group)
        s_next = s + delta

        was_active = predicate(s)
        is_active = predicate(s_next)

        if not was_active and is_active:
            # Started satisfying predicate
            start_pos = pos
        elif was_active and not is_active:
            # Stopped satisfying predicate
            assert start_pos != -1  # noqa: S101
            result.append((start_pos, pos))
            start_pos = -1

        s = s_next

    return result


def subtract_ranges(
    ranges_a: Sequence[tuple[int, int]],
    ranges_b: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Calculates the set difference of two sets of disjoint intervals (A - B).

    Args:
        ranges_a: A sequence of half-open intervals [start, end), sorted and disjoint.
        ranges_b: A sequence of half-open intervals [start, end), sorted and disjoint.

    Returns:
        A list of intervals representing the parts of A that are not covered by B.
    """
    # A adds 1, B subtracts 1. We want regions where sum == 1.
    return _sweep_operation(
        ranges_a,
        ranges_b,
        (+1, -1),
        (-1, +1),
        lambda s: s == 1,
    )


def union_ranges(
    ranges_a: Sequence[tuple[int, int]],
    ranges_b: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Calculates the union of two sets of disjoint intervals (A | B).

    Args:
        ranges_a: A sequence of half-open intervals [start, end), sorted and disjoint.
        ranges_b: A sequence of half-open intervals [start, end), sorted and disjoint.

    Returns:
        A list of intervals representing the union of A and B.
        Overlapping or adjacent intervals are merged.
    """
    # Both add 1. We want regions where sum > 0.
    return _sweep_operation(
        ranges_a,
        ranges_b,
        (+1, -1),
        (+1, -1),
        lambda s: s > 0,
    )


def intersect_ranges(
    ranges_a: Sequence[tuple[int, int]],
    ranges_b: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Calculates the intersection of two sets of disjoint intervals (A & B).

    Args:
        ranges_a: A sequence of half-open intervals [start, end), sorted and disjoint.
        ranges_b: A sequence of half-open intervals [start, end), sorted and disjoint.

    Returns:
        A list of intervals representing the overlapping parts of A and B.
    """
    # Both add 1. We want regions where sum == _INTERSECTION_OVERLAP_COUNT.
    return _sweep_operation(
        ranges_a,
        ranges_b,
        (+1, -1),
        (+1, -1),
        lambda s: s == _INTERSECTION_OVERLAP_COUNT,
    )


def in_range(val: int, test_range: tuple[int, int]) -> bool:
    """Return true if `val` is in `test_range`.

    Args:
        val: The value to test.
        test_range: The range to test against.

    Returns:
        True if `val` is in `test_range`.
    """
    return test_range[0] <= val < test_range[1]


def overlaps(r1: tuple[int, int], r2: tuple[int, int]) -> bool:
    """Return true if `r1` overlaps `r2`.

    Args:
        r1: The first range.
        r2: The second range.

    Returns:
        True if `r1` overlaps `r2`.
    """
    return r1[0] < r2[1] and r1[1] > r2[0]


def contained(r1: tuple[int, int], r2: tuple[int, int]) -> bool:
    """Return true if `r1` is contained in `r2`.

    Args:
        r1: The first range.
        r2: The second range.

    Returns:
        True if `r1` is contained in `r2`.
    """
    return r1[0] >= r2[0] and r1[1] <= r2[1]
