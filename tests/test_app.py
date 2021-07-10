from unittest import TestCase
from parameterized import parameterized, param
from solver.app import distance, Point, load_problem


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
