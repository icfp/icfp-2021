from unittest import TestCase
from parameterized import parameterized, param
from solver.app import (
    distance,
    Point,
    load_problem,
    Problem,
    Figure,
    compute_statistics,
    make_in_hole_matrix,
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
