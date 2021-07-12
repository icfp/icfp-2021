from unittest import TestCase

import z3

from solver.app import Point, hole_edges, load_problem
from solver.polygon import do_intersect_z3


class TestConstraints(TestCase):
    def test_constraint_edge_intersect(self):
        problem = load_problem(28)
        edges = hole_edges(problem.hole)

        opt = z3.Optimize()

        def bv_point(point: tuple[int, int]):
            return Point(z3.BitVecVal(point[0], 16), z3.BitVecVal(point[1], 16))

        count_intersects = sum(
            z3.If(
                do_intersect_z3(
                    bv_point((12, 14)), bv_point((27, 38)), e.source, e.target
                ),
                1,
                0,
            )
            for e in edges
        )

        opt.add(count_intersects > 1)

        check = opt.check()
        assert check == z3.sat

        pb_count_intersects = z3.PbGe(
            [
                (
                    do_intersect_z3(
                        bv_point((12, 14)), bv_point((27, 38)), e.source, e.target
                    ),
                    1,
                )
                for e in edges
            ],
            2,
        )

        opt.add(pb_count_intersects)

        check = opt.check()
        assert check == z3.sat

        res = opt.model().eval(count_intersects)
        print(res)
        res = opt.model().eval(pb_count_intersects)
        print(res)
