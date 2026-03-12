import pytest

from anchorite import range_ops


def test_subtract_ranges_composite() -> None:
    # A: [10, 20), [30, 40)
    # B: [15, 35)
    # Result: [10, 15), [35, 40)
    ranges_a = [(10, 20), (30, 40)]
    ranges_b = [(15, 35)]
    expected = [(10, 15), (35, 40)]
    assert range_ops.subtract_ranges(ranges_a, ranges_b) == expected


def test_subtract_ranges_disjoint() -> None:
    # A: [10, 20)
    # B: [30, 40)
    # Result: [10, 20)
    ranges_a = [(10, 20)]
    ranges_b = [(30, 40)]
    assert range_ops.subtract_ranges(ranges_a, ranges_b) == [(10, 20)]


def test_subtract_ranges_fully_covered() -> None:
    # A: [10, 20)
    # B: [5, 25)
    # Result: empty list
    ranges_a = [(10, 20)]
    ranges_b = [(5, 25)]
    assert range_ops.subtract_ranges(ranges_a, ranges_b) == []


def test_subtract_ranges_multiple_holes() -> None:
    # A: [0, 100)
    # B: [10, 20), [30, 40), [50, 60)
    # Result: [0, 10), [20, 30), [40, 50), [60, 100)
    ranges_a = [(0, 100)]
    ranges_b = [(10, 20), (30, 40), (50, 60)]
    expected = [(0, 10), (20, 30), (40, 50), (60, 100)]
    assert range_ops.subtract_ranges(ranges_a, ranges_b) == expected


def test_union_ranges_overlapping() -> None:
    # A: [10, 20)
    # B: [15, 25)
    # Result: [10, 25)
    ranges_a = [(10, 20)]
    ranges_b = [(15, 25)]
    assert range_ops.union_ranges(ranges_a, ranges_b) == [(10, 25)]


def test_union_ranges_disjoint() -> None:
    # A: [10, 20)
    # B: [30, 40)
    # Result: [10, 20), [30, 40)
    ranges_a = [(10, 20)]
    ranges_b = [(30, 40)]
    assert range_ops.union_ranges(ranges_a, ranges_b) == [(10, 20), (30, 40)]


def test_union_ranges_merging_multiple() -> None:
    # A: [10, 20), [30, 40)
    # B: [15, 35)
    # Result: [10, 40)
    ranges_a = [(10, 20), (30, 40)]
    ranges_b = [(15, 35)]
    assert range_ops.union_ranges(ranges_a, ranges_b) == [(10, 40)]


def test_intersect_ranges_simple() -> None:
    # A: [10, 20)
    # B: [15, 25)
    # Result: [15, 20)
    ranges_a = [(10, 20)]
    ranges_b = [(15, 25)]
    assert range_ops.intersect_ranges(ranges_a, ranges_b) == [(15, 20)]


def test_intersect_ranges_disjoint() -> None:
    # A: [10, 20)
    # B: [30, 40)
    # Result: empty list
    ranges_a = [(10, 20)]
    ranges_b = [(30, 40)]
    assert range_ops.intersect_ranges(ranges_a, ranges_b) == []


def test_intersect_ranges_subset() -> None:
    # A: [10, 50)
    # B: [20, 30), [35, 40)
    # Result: [20, 30), [35, 40)
    ranges_a = [(10, 50)]
    ranges_b = [(20, 30), (35, 40)]
    assert range_ops.intersect_ranges(ranges_a, ranges_b) == [(20, 30), (35, 40)]


def test_intersect_ranges_complex() -> None:
    # A: [10, 20), [30, 40)
    # B: [15, 35)
    # Result: [15, 20), [30, 35)
    ranges_a = [(10, 20), (30, 40)]
    ranges_b = [(15, 35)]
    expected = [(15, 20), (30, 35)]
    assert range_ops.intersect_ranges(ranges_a, ranges_b) == expected


def test_edge_cases_empty() -> None:
    assert range_ops.subtract_ranges([], []) == []
    assert range_ops.subtract_ranges([(10, 20)], []) == [(10, 20)]
    assert range_ops.subtract_ranges([], [(10, 20)]) == []
    assert range_ops.union_ranges([], []) == []
    assert range_ops.union_ranges([(10, 20)], []) == [(10, 20)]
    assert range_ops.union_ranges([], [(10, 20)]) == [(10, 20)]
    assert range_ops.intersect_ranges([], []) == []
    assert range_ops.intersect_ranges([(10, 20)], []) == []


def test_adjacent_ranges() -> None:
    # Subtract adjacent: [10, 20) - [20, 30) -> [10, 20)
    assert range_ops.subtract_ranges([(10, 20)], [(20, 30)]) == [(10, 20)]
    # Union adjacent: [10, 20) U [20, 30) -> [10, 30)
    assert range_ops.union_ranges([(10, 20)], [(20, 30)]) == [(10, 30)]
    # Intersect adjacent: [10, 20) & [20, 30) -> []
    assert range_ops.intersect_ranges([(10, 20)], [(20, 30)]) == []


@pytest.mark.parametrize(
    ("val", "test_range", "expected"),
    [
        (5, (0, 10), True),
        (0, (0, 10), True),
        (9, (0, 10), True),
        (10, (0, 10), False),
        (-1, (0, 10), False),
        (5, (10, 20), False),
    ],
)
def test_in_range(val: int, test_range: tuple[int, int], expected: bool) -> None:
    assert range_ops.in_range(val, test_range) is expected


@pytest.mark.parametrize(
    ("r1", "r2", "expected"),
    [
        # Overlapping cases
        ((0, 10), (5, 15), True),
        ((5, 15), (0, 10), True),
        ((0, 10), (0, 10), True),
        ((0, 20), (5, 10), True),
        ((5, 10), (0, 20), True),
        # Non-overlapping cases / Touching
        ((0, 10), (10, 20), False),
        ((10, 20), (0, 10), False),
        ((0, 5), (15, 20), False),
    ],
)
def test_overlaps(r1: tuple[int, int], r2: tuple[int, int], expected: bool) -> None:
    assert range_ops.overlaps(r1, r2) is expected


@pytest.mark.parametrize(
    ("r1", "r2", "expected"),
    [
        # Contained
        ((5, 10), (0, 20), True),
        ((0, 10), (0, 20), True),
        ((10, 20), (0, 20), True),
        ((0, 20), (0, 20), True),
        # Not contained
        ((0, 20), (5, 10), False),
        ((0, 15), (10, 20), False),
        ((-5, 5), (0, 10), False),
        ((0, 10), (10, 20), False),
    ],
)
def test_contained(r1: tuple[int, int], r2: tuple[int, int], expected: bool) -> None:
    assert range_ops.contained(r1, r2) is expected
