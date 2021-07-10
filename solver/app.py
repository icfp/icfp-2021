import json
from pathlib import Path
from typing import Iterable, List, Dict, NamedTuple

import click
import math
import z3
from pydantic.dataclasses import dataclass
from .types import Point, Pose, Problem, Figure, Hole, EdgeLengthRange, Solution
from . import polygon
from collections import defaultdict

ROOT_DIR = Path(__file__).parent.parent
PROBLEMS_DIR = ROOT_DIR / "problems"


def min_max_edge_length(epsilon: int, source: Point, target: Point) -> EdgeLengthRange:
    max_ratio = epsilon / 1000000
    edge_length = distance(source, target)
    min_length = edge_length * (1 - max_ratio)
    max_length = edge_length * (1 + max_ratio)

    return EdgeLengthRange(min=math.ceil(min_length), max=int(max_length))


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


def internal_run(problem_number: int, minimize: bool = False) -> Solution:
    p = load_problem(problem_number)

    stats = compute_statistics(p)

    in_hole_map = make_in_hole_matrix(stats, p)

    json.dump(
        [[point.x, point.y] for point, inside in in_hole_map.items() if inside],
        open("map.json", "w"),
    )

    print(f"Map Matrix Size {len(in_hole_map)}")

    opt = z3.Optimize()

    x_bits = bits_for(i.x for i in p.hole)
    y_bits = bits_for(i.y for i in p.hole)

    # x_sort = z3.BitVecSort(bits_for(max(i.x, i.y) for i in p.hole) * 2)
    x_sort = z3.BitVecSort(x_bits)
    y_sort = z3.BitVecSort(y_bits)

    # translate vertices to z3
    vertices = p.figure.vertices
    xs = [
        z3.BitVec(f"x_{point.x}_idx{idx}", x_sort) for idx, point in enumerate(vertices)
    ]
    ys = [
        z3.BitVec(f"y_{point.y}_idx{idx}", y_sort) for idx, point in enumerate(vertices)
    ]

    point_vars = [Point(x, y) for x, y in zip(xs, ys)]

    vertex, mk_vertex, (vertex_x, vertex_y) = z3.TupleSort("vertex", (x_sort, y_sort))
    vertices = [mk_vertex(x, y) for x, y in zip(xs, ys)]

    # add a distinct constraint on the vertex points
    opt.add(z3.Distinct(*vertices))

    # calculate edge distances
    distance_limits = [
        min_max_edge_length(p.epsilon, p.figure.vertices[p1], p.figure.vertices[p2])
        for p1, p2 in p.figure.edges
    ]
    exact_distances = [
        distance(p.figure.vertices[p1], p.figure.vertices[p2])
        for p1, p2 in p.figure.edges
    ]
    distance_vars = [
        distance(Point(xs[p1], ys[p1]), Point(xs[p2], ys[p2]))
        for p1, p2 in p.figure.edges
    ]
    # for limit, distance_var in list(zip(distance_limits, distance_vars))[5:7]:
    for limit, distance_var, exact_distance in zip(distance_limits, distance_vars, exact_distances):
        # print(limit, distance_var)

        assert limit.min <= exact_distance <= limit.max

        # Exact distances
        # opt.add(distance_var == exact_distance)

        # Min/max
        opt.add(distance_var >= limit.min)
        opt.add(distance_var <= limit.max)

        # opt.add(i-1 <= a)
        # opt.add(a <= i+1)
        # opt.assert_and_track(i == a, f"foo{i}")

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

    min_hole_dist_points = []
    for idx, h in enumerate(p.hole):
        p_x = z3.BitVec(f"hole_idx{idx}_dist_x", x_sort)
        p_y = z3.BitVec(f"hole_idx{idx}_dist_y", y_sort)

        vertex = mk_vertex(p_x, p_y)
        min_hole_dist_points.append(vertex)

        opt.add(z3.Or(*[vertex == figure_point for figure_point in vertices]))

        # opt.minimize(distance(Point(p_x, p_y), h))

    opt.add(z3.Distinct(*min_hole_dist_points))

    # mixed_size = xs[0].size() + ys[0].size()
    # result_size = z3.BitVecSort(
    #     max(xs[0].size() - ys[0].size(), ys[0].size() - xs[0].size()) + mixed_size
    # )

    total_dislikes = z3.BitVec(
        "dislikes", 21
    )  # what size should this be and why is it 21??

    opt.add(
        total_dislikes
        == sum(
            distance(Point(vertex_x(v), vertex_y(v)), h)
            for v, h in zip(min_hole_dist_points, p.hole)
        )
    )

    if minimize:
        opt.minimize(total_dislikes)

    # print(distances)

    # edges = p.figure.edges
    # print(edges)

    # b = opt.maximize(foo)
    # print(b)

    res = opt.check()
    assert res == z3.sat, 'Failed to solve'
    # if str(res) != "sat":
    #     core = opt.unsat_core()
    #     print(core["foo20"])
    #     print(core)
    print(res)

    model = opt.model()

    for v in vertices:
        x = model.eval(vertex_x(v))
        y = model.eval(vertex_y(v))
        print(f"[{x}, {y}],")

    pose: Pose = [
        Point(model.eval(vertex_x(v)).as_long(), model.eval(vertex_y(v)).as_long())
        for v in vertices
    ]

    solution: Solution = Solution(vertices=pose)
    print(solution)
    return solution


@click.command()
@click.argument("problem_number")
@click.option("--minimize/--no-minimize", default=False)
def run(problem_number: int, minimize: bool) -> Solution:
    return internal_run(problem_number)


if __name__ == "__main__":
    run()
