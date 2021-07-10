from unittest import TestCase
from parameterized import parameterized, param
from solver.app import load_problem, Point, Problem
from solver.polygon import in_polygon


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
