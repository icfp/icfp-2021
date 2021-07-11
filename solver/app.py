from pathlib import Path
from typing import Iterable, List, Dict, NamedTuple, Callable

import click
import math
import time
import datetime

import z3
from pydantic.dataclasses import dataclass

from . import polygon
from .format import to_json
from .types import Point, Pose, Problem, Figure, Hole, EdgeLengthRange, Solution, Output
from collections import defaultdict

ROOT_DIR = Path(__file__).parent.parent
PROBLEMS_DIR = ROOT_DIR / "problems"


def load_problem(problem_number: int) -> Problem:
    return Problem.__pydantic_model__.parse_file(
        PROBLEMS_DIR / f"{problem_number}.json"
    )


def distance(p1: Point, p2: Point):
    delta_x = p1.x - p2.x
    delta_y = p1.y - p2.y
    if isinstance(delta_x, z3.BitVecRef) and isinstance(delta_y, z3.BitVecRef):
        foo = delta_x.size() + delta_y.size()
        d_x = z3.SignExt(max(0, delta_y.size() - delta_x.size()) + foo, delta_x)
        d_y = z3.SignExt(max(0, delta_x.size() - delta_y.size()) + foo, delta_y)
        delta_x, delta_y = d_x, d_y
    pow_x = delta_x * delta_x
    pow_y = delta_y * delta_y
    return pow_x + pow_y


def mh_distance(p1: Point, p2: Point):
    delta_x = p1.x - p2.x
    delta_y = p1.y - p2.y

    return abs(delta_x) + abs(delta_y)


def z3_mh_distance(p1: Point, p2: Point):
    delta_x = p1.x - p2.x
    delta_y = p1.y - p2.y

    return z3.If(delta_x < 0, -delta_x, delta_x) + z3.If(delta_y < 0, -delta_y, delta_y)


DistanceFunc = Callable[[Point, Point], int]


def min_max_edge_length(
    epsilon: int, source: Point, target: Point, distance_func: DistanceFunc = distance
) -> EdgeLengthRange:
    edge_length = distance_func(source, target)

    max_ratio = epsilon / 1000000

    min_length = edge_length * (1 - max_ratio)
    max_length = edge_length * (1 + max_ratio)

    return EdgeLengthRange(min=math.ceil(min_length), max=int(max_length))


def dislikes(hole: Hole, pose: Figure):
    return sum(min(distance(v, h) for v in pose.vertices) for h in hole)


def bits_for(xs: Iterable[int]) -> int:
    return int(math.log2(max(xs)) + 1)


@dataclass(frozen=True)
class ProblemStatistics:
    min_x: int
    min_y: int
    max_x: int
    max_y: int


def compute_statistics(problem: Problem) -> ProblemStatistics:
    hole = problem.hole
    min_x = min(p.x for p in hole)
    min_y = min(p.y for p in hole)

    max_x = max(p.x for p in hole)
    max_y = max(p.y for p in hole)

    return ProblemStatistics(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)


InHoleLookup = Dict[Point, bool]


def make_in_hole_matrix(stats: ProblemStatistics, problem) -> InHoleLookup:
    lookup = defaultdict(lambda: False)
    for x in range(stats.max_x + 1):
        for y in range(stats.max_y + 1):
            point = Point(x, y)
            lookup[point] = polygon.in_polygon(point, problem.hole)

    return lookup


class InclusiveRange(NamedTuple):
    start: int
    end: int


class YPointRange(NamedTuple):
    x: int
    y_inclusive_ranges: List[InclusiveRange]


def make_ranges(
    lookup: InHoleLookup, stats: ProblemStatistics
) -> Iterable[YPointRange]:
    for x in range(stats.max_x + 1):
        y_ranges = []
        y = 0
        while y < stats.max_y + 1:
            if lookup.get(Point(x, y)):
                start_y = y
                while lookup.get(Point(x, y)):
                    y += 1
                y_ranges.append(InclusiveRange(start=start_y, end=y - 1))
            else:
                y += 1
        if y_ranges:
            yield YPointRange(x=x, y_inclusive_ranges=y_ranges)


def _run(problem_number: int, minimize: bool = False, debug: bool = False) -> Solution:
    problem = load_problem(problem_number)

    stats = compute_statistics(problem)

    in_hole_map = make_in_hole_matrix(stats, problem)

    map_points = [[point.x, point.y] for point, inside in in_hole_map.items() if inside]

    print(f"Map Matrix Size {len(in_hole_map)}")

    opt = z3.Optimize()

    x_bits = bits_for(i.x for i in problem.hole)
    y_bits = bits_for(i.y for i in problem.hole)

    # x_sort = z3.BitVecSort(bits_for(max(i.x, i.y) for i in p.hole) * 2)
    x_sort = z3.BitVecSort(x_bits)
    y_sort = z3.BitVecSort(y_bits)

    # translate vertices to z3
    vertices = problem.figure.vertices
    xs = [
        z3.BitVec(f"x_{point.x}_idx{idx}", x_sort) for idx, point in enumerate(vertices)
    ]
    ys = [
        z3.BitVec(f"y_{point.y}_idx{idx}", y_sort) for idx, point in enumerate(vertices)
    ]

    point_vars = [Point(x, y) for x, y in zip(xs, ys)]

    vertex, mk_vertex, (vertex_x, vertex_y) = z3.TupleSort("vertex", (x_sort, y_sort))
    vertices = [mk_vertex(x, y) for x, y in zip(xs, ys)]

    def constrain_to_xy_in_hole():
        ranges = list(make_ranges(in_hole_map, stats))

        for x_var, y_var in point_vars:
            opt.add(
                z3.Or(
                    *[
                        z3.And(
                            x_var == x,
                            z3.Or(
                                *[
                                    z3.And(r.start <= y_var, y_var <= r.end)
                                    for r in y_ranges
                                ]
                            ),
                        )
                        for x, y_ranges in ranges
                    ]
                )
            )

    # add a distinct constraint on the vertex points
    def constrain_unique_positions():
        opt.add(z3.Distinct(*vertices))

    # calculate edge distances
    def constrain_distances(distance_func: DistanceFunc = distance):
        distance_limits = [
            min_max_edge_length(
                problem.epsilon,
                problem.figure.vertices[p1],
                problem.figure.vertices[p2],
                distance_func=distance_func,
            )
            for p1, p2 in problem.figure.edges
        ]
        exact_distances = [
            distance_func(problem.figure.vertices[p1], problem.figure.vertices[p2])
            for p1, p2 in problem.figure.edges
        ]
        distance_values = [
            distance_func(Point(xs[p1], ys[p1]), Point(xs[p2], ys[p2]))
            for p1, p2 in problem.figure.edges
        ]
        # for limit, distance_var in list(zip(distance_limits, distance_vars))[5:7]:
        for limit, distance_value, exact_distance in zip(
            distance_limits, distance_values, exact_distances
        ):
            # print(limit, distance_var)

            # assert limit.min <= exact_distance <= limit.max

            # Exact distances
            # opt.add(distance_value == exact_distance)

            # Min/max
            opt.add(distance_value >= limit.min)
            opt.add(distance_value <= limit.max)

            # opt.add(i-1 <= a)
            # opt.add(a <= i+1)
            # opt.assert_and_track(i == a, f"foo{i}")

    def minimize_dislikes():
        min_dist = []
        for i, h in enumerate(problem.hole):
            dist = []
            for j, p in enumerate(problem.figure.vertices):
                dist.append(z3_mh_distance(h, p))

            b0 = dist[0]
            for b in dist[1:]:
                b0 = z3.If(b < 0, b, b0)
            min_dist.append(b0)

        total_dislikes = sum(min_dist)

        opt.add(total_dislikes < 6000)

        if minimize:
            opt.minimize(total_dislikes)

    constraints = [
        constrain_to_xy_in_hole,
        constrain_unique_positions,
        constrain_distances,
        minimize_dislikes,
    ]

    for c in constraints:
        print(f"Adding constraint: {c.__name__}")

        t0 = time.perf_counter()

        c()
        res = opt.check()

        t1 = time.perf_counter()

        total = datetime.timedelta(seconds=t1 - t0)

        print("Result", res, "- elapsed time:", total)

        # if str(res) != z3.sat:
        #     core = opt.unsat_core()
        #     print(core["foo20"])
        #     print(core)

        assert res == z3.sat, "Failed to solve"

    res = opt.check()

    if res != z3.sat:
        print(
            to_json(
                Output(problem=problem, solution=Solution([]), map_points=map_points)
            )
        )
        raise Exception("Failed to solve!")

    # if str(res) != "sat":
    #     core = opt.unsat_core()
    #     print(core["foo20"])
    #     print(core)

    model = opt.model()

    pose: Pose = [
        Point(model.eval(vertex_x(v)).as_long(), model.eval(vertex_y(v)).as_long())
        for v in vertices
    ]

    solution: Solution = Solution(vertices=pose)
    print("Solution:")

    if debug:
        print(
            to_json(Output(problem=problem, solution=solution, map_points=map_points))
        )
        return solution

    print(to_json(solution))
    return solution


@click.command()
@click.argument("problem_number")
@click.option("--minimize/--no-minimize", default=False)
@click.option("--debug/--no-debug", default=False)
def run(problem_number: int, minimize: bool, debug: bool) -> Solution:
    return _run(problem_number, minimize, debug)


if __name__ == "__main__":
    run()
