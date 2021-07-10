from unittest import TestCase

from solver.app import load_problem
from solver.subpolys import voronoi_polys
from solver.types import Problem


class TestSubPolygons(TestCase):
    def test_in_polygon(self) -> None:
        problem: Problem = load_problem(1)

        result = voronoi_polys(problem.hole)
        print([(p.start, p.end) for p in result])
