from typing import List
from unittest import TestCase
from parameterized import parameterized, param
from solver.app import Edge, distance, Point, load_problem, min_max_edge_length


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
        [param(100000, Edge(0, 1), [Point(2, 4), Point(4, 8)], 18, 22),
        param(100000, Edge(0, 1), [Point(4, 8), Point(2, 4)], 18, 22),
        param(40000, Edge(0, 1), [Point(4, 8), Point(2, 4)], 19.2, 20.8),
        param(1000000, Edge(0, 1), [Point(4, 8), Point(2, 4)], 0, 40),
        param(0, Edge(0, 1), [Point(2, 4), Point(4, 8)], 20, 20)]
    )
    def test_min_max_edge_length(
        self, epsilon: int, edge: Edge, vertices: List[Point], expectedMin: float, expectedMax: float):
        actual = min_max_edge_length(epsilon, edge, vertices)
        self.assertEqual(actual.min, expectedMin)
        self.assertEqual(actual.max, expectedMax)
