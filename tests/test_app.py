from unittest import TestCase

from parameterized import param, parameterized  # type: ignore

from solver.app import (
    Figure,
    Point,
    Problem,
    compute_statistics,
    distance,
    invalid_intersecting_edges,
    load_problem,
    make_in_hole_matrix,
    make_ranges,
    min_max_edge_length,
)


class TestApp(TestCase):
    @parameterized.expand(
        [param(Point(1, 1), Point(2, 2), 2), param(Point(4, 5), Point(5, 10), 26)]
    )
    def test_distance(self, first: Point, second: Point, expected: int):
        actual = distance(first, second)
        self.assertEqual(actual, expected)

    def test_problem_as_tuples(self):
        problem = load_problem(1)
        self.assertEqual(problem.hole[0].x, 45)
        self.assertEqual(problem.hole[0].y, 80)

        self.assertEqual(problem.hole[-1].x, 55)
        self.assertEqual(problem.hole[-1].y, 80)

    @parameterized.expand(
        [
            param(100000, Point(2, 4), Point(4, 8), 18, 22),
            param(100000, Point(4, 8), Point(2, 4), 18, 22),
            param(40000, Point(4, 8), Point(2, 4), 20, 20),
            param(1000000, Point(4, 8), Point(2, 4), 0, 40),
            param(0, Point(2, 4), Point(4, 8), 20, 20),
        ]
    )
    def test_min_max_edge_length(
        self,
        epsilon: int,
        source: Point,
        target: Point,
        expectedMin: float,
        expectedMax: float,
    ):
        actual = min_max_edge_length(epsilon, source, target)
        self.assertEqual(actual.min, expectedMin)
        self.assertEqual(actual.max, expectedMax)

    def test_simple_problem(self):
        problem = Problem(
            epsilon=0,
            hole=[Point(1, 1), Point(1, 4), Point(4, 4), Point(4, 1)],
            figure=Figure(edges=[], vertices=[]),
        )

        stats = compute_statistics(problem)

        self.assertEqual(stats.min_x, 1)
        self.assertEqual(stats.min_y, 1)
        self.assertEqual(stats.max_x, 4)
        self.assertEqual(stats.max_y, 4)

        map = make_in_hole_matrix(stats, problem)
        print(map)

        self.assertTrue(map.get(Point(2, 2)))
        self.assertTrue(map.get(Point(1, 4)))
        self.assertFalse(map.get(Point(0, 1)))

    def test_problem_one_bug(self):
        problem = load_problem(1)

        stats = compute_statistics(problem)
        map = make_in_hole_matrix(stats, problem)

        self.assertFalse(map.get(Point(7, 9)))
        for x in make_ranges(map, stats):
            if x.x == 7:
                self.assertEqual(x.y_inclusive_ranges[0].start, 5)
                self.assertEqual(x.y_inclusive_ranges[0].end, 8)

    def test_problem_10_ranges(self):
        problem = load_problem(10)

        stats = compute_statistics(problem)
        map = make_in_hole_matrix(stats, problem)
        ranges = make_ranges(map, stats)

        x_lookup = dict((r.x, r.y_inclusive_ranges) for r in ranges)

        print(x_lookup)

        self.assertEqual(x_lookup[12], [(9, 14), (52, 86)])
        self.assertEqual(x_lookup[79], [(45, 74)])
        self.assertEqual(x_lookup[82], [(69, 75)])
        self.assertEqual(x_lookup[83], [(76, 76)])
        self.assertEqual(x_lookup[10], [(9, 9), (62, 86)])
        self.assertEqual(x_lookup[6], [(82, 86)])
        self.assertEqual(x_lookup[5], [(86, 86)])

    def test_invalid_intersecting_edges(self):
        problem = Problem(
            epsilon=0,
            hole=[Point(1, 1), Point(7, 1), Point(3, 3), Point(7, 5), Point(1, 5)],
            figure=Figure(edges=[(0, 1)], vertices=[(2, 2), (2, 4)]),
        )

        stats = compute_statistics(problem)
        lookup = make_in_hole_matrix(stats, problem)
        invalid_edges = invalid_intersecting_edges(lookup, problem.hole)

        print('Invalid edges')
        print(invalid_edges[Point(6, 5)])

        self.assertTrue(Point(4, 2) in invalid_edges)
        self.assertTrue(Point(4, 4) in invalid_edges[Point(4, 2)])
        self.assertFalse(Point(2, 2) in invalid_edges[Point(3, 4)])
        self.assertTrue(False)
