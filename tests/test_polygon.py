from unittest import TestCase
from parameterized import parameterized, param
from solver.app import load_problem, Point, Problem, Hole
from solver.polygon import do_intersect, in_polygon


class TestPolygon(TestCase):
    @parameterized.expand(
        [
            param(Point(100, 100), False),
            param(Point(30, 7), True),
            param(Point(30, 5), True),
            param(Point(70, 95), True),
            param(Point(95, 95), True),
            param(Point(55, 80), True),
            param(Point(60, 80), True),
            param(Point(85, 80), True),
            param(Point(87, 80), False),
        ]
    )
    def test_in_polygon(self, point: Point, expected: bool) -> None:
        problem: Problem = load_problem(1)
        actual: bool = in_polygon(point, problem.hole)
        if expected:
            self.assertTrue(actual, f"Expected {point} to be in or on the hole")
        else:
            self.assertFalse(actual, f"Expected {point} not to be in or on the hole")

    @parameterized.expand(
        [
            param(Point(0, 1), False),
            param(Point(0, 3), False),
            param(Point(1, 1), True),
            param(Point(1, 2), True),
            param(Point(1, 3), True),
            param(Point(2, 1), True),
            param(Point(2, 2), True),
            param(Point(2, 3), True),
            param(Point(3, 1), True),
            param(Point(3, 2), True),
            param(Point(3, 3), True),
            param(Point(4, 1), False),
            param(Point(4, 3), False),
        ]
    )
    def test_in_square(self, point: Point, expected: bool) -> None:
        hole: Hole = [Point(1, 1), Point(3, 1), Point(3, 3), Point(1, 3)]
        actual: bool = in_polygon(point, hole)
        if expected:
            self.assertTrue(actual, f"Expected {point} to be in or on the hole")
        else:
            self.assertFalse(actual, f"Expected {point} not to be in or on the hole")

    @parameterized.expand(
        [
            param(Point(0, 1), False),
        ]
    )
    def test_in_square_two(self, point: Point, expected: bool) -> None:
        hole: Hole = [Point(1, 1), Point(1, 3), Point(3, 3), Point(3, 1)]
        actual: bool = in_polygon(point, hole)
        if expected:
            self.assertTrue(actual, f"Expected {point} to be in or on the hole")
        else:
            self.assertFalse(actual, f"Expected {point} not to be in or on the hole")

    @parameterized.expand(
        [
            param(Point(3, 2), True),
            param(Point(4, 5), True),
            param(Point(2, 3), True),
            param(Point(5, 2), False),
            param(Point(6, 4), False),
            param(Point(5, 6), False),
            param(Point(4, 5), True),
            param(Point(4, 3), True),
            param(Point(2, 3), True),
            param(Point(2, 6), False),
        ]
    )
    def test_in_diamond(self, point: Point, expected: bool) -> None:
        hole: Hole = [Point(3, 2), Point(5, 4), Point(3, 6), Point(1, 4)]
        actual: bool = in_polygon(point, hole)
        if expected:
            self.assertTrue(actual, f"Expected {point} to be in or on the hole")
        else:
            self.assertFalse(actual, f"Expected {point} not to be in or on the hole")

    @parameterized.expand(
        [
            param(Point(6, 4), False),
            param(Point(5, 5), False),
            param(Point(3, 5), True),
            param(Point(6, 3), False),
        ]
    )
    def test_in_diamond_two(self, point: Point, expected: bool) -> None:
        hole: Hole = [Point(5, 4), Point(3, 6), Point(1, 5), Point(3, 2)]
        actual: bool = in_polygon(point, hole)
        if expected:
            self.assertTrue(actual, f"Expected {point} to be in or on the hole")
        else:
            self.assertFalse(actual, f"Expected {point} not to be in or on the hole")

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

    # See discord for source of test case
    # https://discord.com/channels/855340178745720832/855340178745720834/863434048621510686
    @parameterized.expand(
        [
            param(Point(58, 31), True),
            param(Point(60, 31), False),
            param(Point(63, 27), False),
            param(Point(65, 31), False),
            param(Point(70, 31), False),
            param(Point(77, 31), False),
            param(Point(80, 31), False),
            param(Point(79, 45), True),
            param(Point(80, 45), False),
            param(Point(83, 45), False),
            param(Point(10, 83), True),
            param(Point(9, 83), True),
            param(Point(5, 86), True),
            param(Point(6, 85), True),
        ]
    )
    def test_problem_10_bug(self, point: Point, expected: bool) -> None:
        problem = load_problem(10)
        hole: Hole = problem.hole
        actual: bool = in_polygon(point, hole)
        if expected:
            self.assertTrue(actual, f"Expected {point} to be in or on the hole")
        else:
            self.assertFalse(actual, f"Expected {point} not to be in or on the hole")

    @parameterized.expand(
        [
            param(Point(20, 25), True),
            param(Point(500, 25), False),
        ]
    )
    def test_problem_2_bug(self, point: Point, expected: bool) -> None:
        problem = load_problem(2)
        hole: Hole = problem.hole
        actual: bool = in_polygon(point, hole)
        if expected:
            self.assertTrue(actual, f"Expected {point} to be in or on the hole")
        else:
            self.assertFalse(actual, f"Expected {point} not to be in or on the hole")
