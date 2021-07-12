from unittest import TestCase

import z3

from solver.app import Point, hole_edges, load_problem
from solver.polygon import do_intersect_z3, do_intersect


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
                    bv_point((3, 26)), bv_point((12, 26)), e.source, e.target
                ),
                0,
                1,
            )
            for e in edges
        )

        opt.add(count_intersects > 1)

        check = opt.check()
        print(opt.model().eval(count_intersects))
        assert check == z3.sat

        pb_count_intersects = sum(
            z3.If(
                do_intersect_z3(
                    bv_point((12, 14)), bv_point((27, 38)), e.source, e.target
                ),
                1,
                0
            )
            for e in edges
        )

        opt.add(pb_count_intersects >= 1)

        check = opt.check()

        print('pb_count_intersects', opt.model().eval(pb_count_intersects))

        assert check == z3.sat

        res = opt.model().eval(count_intersects)
        print(res)
        res = opt.model().eval(pb_count_intersects)
        print(res)

    def test_valid_edges_dont_intersect(self):
        problem = load_problem(28)
        edges = hole_edges(problem.hole)

        opt = z3.Optimize()

        def bv_point(point: tuple[int, int]):
            return Point(z3.BitVecVal(point[0], 16), z3.BitVecVal(point[1], 16))

        points = [
            ((13, 15), (13, 13)),
            ((2, 27), (13, 11))
        ]

        test_edge_sums = [
            sum(
                z3.If(
                    do_intersect_z3(
                        bv_point(fst), bv_point(snd), e.source, e.target
                    ),
                    1,
                    0,
                )
                for e in edges
            )
            for fst, snd in points
        ]

        test_edges = [
            edge_sum <= 1 for edge_sum in test_edge_sums
        ]

        opt.add(*test_edges)

        check = opt.check()
        print(opt.model())
        print(opt.model().eval(test_edge_sums[0]))
        print(opt.model().eval(test_edges[0]))
        assert check == z3.sat

    def test_invalid_edges_intersect(self):
        problem = load_problem(28)
        edges = hole_edges(problem.hole)

        opt = z3.Optimize()

        def bv_point(point: tuple[int, int]):
            return Point(z3.BitVecVal(point[0], 16), z3.BitVecVal(point[1], 16))

        points = [
            ((15, 36), (31, 6)),
        ]

        print([
            sum(1 for e in edges if do_intersect(Point(*fst), Point(*snd), e.source, e.target))
            for fst, snd in points
        ])

        test_edge_sums = [
            sum(
                z3.If(
                    do_intersect_z3(
                        bv_point(fst), bv_point(snd), e.source, e.target
                    ),
                    1,
                    0,
                )
                for e in edges
            )
            for fst, snd in points
        ]

        test_edges = [
            edge_sum >= 1 for edge_sum in test_edge_sums
        ]

        opt.add(*test_edges)

        check = opt.check()
        print(opt.model())
        print(opt.model().eval(test_edge_sums[0]))
        print(opt.model().eval(test_edges[0]))
        assert check == z3.sat
