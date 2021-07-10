from unittest import TestCase
from parameterized import parameterized, param
from solver.app import distance, Point


class TestApp(TestCase):
    @parameterized.expand([param((1, 1), (2, 2), 2), param((4, 5), (5, 10), 26)])
    def test_distance(self, first: Point, second: Point, expected: int):
        actual = distance(first, second)
        self.assertEqual(actual, expected)
