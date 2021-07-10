from unittest import TestCase
from parameterized import parameterized, param
from solver.app import load_problem, Point, Problem
from solver.polygon import do_intersect, in_polygon


class TestPolygon(TestCase):
    @parameterized.expand(
        [
            param(Point(100, 100), False),
            param(Point(30, 7), True),
            param(Point(30, 5), True),
            param(Point(70, 95), True),
            param(Point(95, 95), True),
        ]
    )

    def test_in_polygon(self, point: Point, expected: bool) -> None:
        problem: Problem = load_problem(1)
        actual: bool = in_polygon(point, problem.hole)
        self.assertEqual(actual, expected)

    @parameterized.expand(
        [
            param(Point(1, 1), Point(10, 1), Point(1, 2), Point(10, 2), False),
            param(Point(10, 0), Point(0, 10), Point(0, 0), Point(10, 10), True),
            param(Point(-5, -5), Point(0, 0), Point(1, 1), Point(10, 10), False),
        ]
    )
    def test_intersect(
        self, p1: Point, q1: Point, p2: Point, q2: Point, expected: bool
    ) -> None:
        actual: bool = do_intersect(p1, q1, p2, q2)
        self.assertEqual(actual, expected)
