import datetime
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import click
import z3
from pydantic.dataclasses import dataclass

from . import polygon
from .format import to_json
from .types import (
    ConstraintFunc,
    DebugVars,
    DistanceFunc,
    EdgeLengthRange,
    Figure,
    Hole,
    InclusiveRange,
    InHoleLookup,
    Output,
    Point,
    Pose,
    Problem,
    Solution,
    YPointRange,
)

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


class Constraint:
    def __init__(self, func: ConstraintFunc, name: str = None):
        self.func = func
        self._name = name if name is not None else func.__name__

    def __call__(self, *args, **kwargs) -> DebugVars:
        return self.func()

    @property
    def name(self) -> str:
        return self._name

    @property
    def disable(self) -> "Constraint":
        return Constraint(lambda: {}, name=f"{self.name} (disabled)")


def constraint(f: ConstraintFunc) -> Constraint:
    return Constraint(f)


def compute_statistics(problem: Problem) -> ProblemStatistics:
    hole = problem.hole
    min_x = min(p.x for p in hole)
    min_y = min(p.y for p in hole)

    max_x = max(p.x for p in hole)
    max_y = max(p.y for p in hole)

    return ProblemStatistics(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)


def make_in_hole_matrix(stats: ProblemStatistics, problem) -> InHoleLookup:
    lookup = defaultdict(lambda: False)
    for x in range(stats.max_x + 1):
        for y in range(stats.max_y + 1):
            point = Point(x, y)
            lookup[point] = polygon.in_polygon(point, problem.hole)

    return lookup


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


def invalid_intersecting_edges(
    lookup: InHoleLookup, hole: Hole
) -> Dict[Point, set[Point]]:
    inside_points = [point for point, inside in lookup.items() if inside]
    hole_edges = list(zip(hole, hole[1:] + [hole[0]]))
    intersecting_edges = defaultdict(set)
    for p1 in inside_points:

        for p2 in inside_points:
            if p1.x == p2.x and p1.y == p2.y:
                continue

            for e1, e2 in hole_edges:
                if polygon.do_intersect(p1, p2, e1, e2):
                    # we are excluding valid edges where the line terminates on an edge vertex
                    # but not fixing yet because there are also invalid edges that terminate
                    # on an edge vertex
                    intersecting_edges[p1].add(p2)
                    intersecting_edges[p2].add(p1)

    return intersecting_edges


def _run(problem_number: int, minimize: bool = False, debug: bool = False) -> Output:
    problem = load_problem(problem_number)

    stats = compute_statistics(problem)

    in_hole_map = make_in_hole_matrix(stats, problem)

    map_points = [[point.x, point.y] for point, inside in in_hole_map.items() if inside]

    print(f"Building allowed edges for {problem_number}")
    t0 = time.perf_counter()
    disallowed_edges: Dict[Point, List[Point]] = invalid_intersecting_edges(in_hole_map, problem.hole)

    t1 = time.perf_counter()

    total = datetime.timedelta(seconds=t1 - t0)

    print("Done!.. elapsed time:", total)

    # print(disallowed_edges)

    print(f"Map Matrix Size {len(in_hole_map)}")

    opt = z3.Optimize()

    x_bits = bits_for(i.x for i in problem.hole)
    y_bits = bits_for(i.y for i in problem.hole)

    # x_sort = z3.BitVecSort(bits_for(max(i.x, i.y) for i in p.hole) * 2)
    x_sort = z3.BitVecSort(x_bits + 1)
    y_sort = z3.BitVecSort(y_bits + 1)

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

    figure_points = [Point(x, y) for x, y in zip(xs, ys)]
    hole_points = [
        Point(z3.BitVecVal(p.x, x_sort), z3.BitVecVal(p.y, y_sort))
        for p in problem.hole
    ]

    # hole = [mk_vertex(p.x, p.y) for p in problem.hole]

    edges = problem.figure.edges
    epsilon = problem.epsilon
    original_figure_points = problem.figure.vertices

    @constraint
    def constrain_to_xy_in_hole() -> DebugVars:
        ranges = list(make_ranges(in_hole_map, stats))

        for x_var, y_var in point_vars:
            x_constraints = []
            for x, y_ranges in ranges:
                x_match = x_var == x
                x_constraints.append(x_match)
                opt.add(
                    z3.Implies(
                        x_match,
                        z3.Or(
                            *[
                                z3.And(r.start <= y_var, y_var <= r.end)
                                for r in y_ranges
                            ]
                        ),
                    )
                )

            opt.add(z3.Or(*x_constraints))

        return {}

    @constraint
    def constrain_to_edges_in_hole() -> DebugVars:
        for source, target in problem.figure.edges:
            v_source = vertices[source]
            v_target = vertices[target]

            conditions = []
            for allowed_source, disallowed_targets in disallowed_edges.items():
                if_sources_match = v_source == mk_vertex(
                    allowed_source.x, allowed_source.y
                )
                conditions.append(if_sources_match)

                target_constraints = [
                    v_target != mk_vertex(disallowed_target.x, disallowed_target.y)
                    for disallowed_target in disallowed_targets
                ]

                opt.add(z3.Implies(if_sources_match, z3.And(*target_constraints)))

            opt.add(z3.Or(*conditions))

        return {}

    # add a distinct constraint on the vertex points
    @constraint
    def constrain_unique_positions() -> DebugVars:
        opt.add(z3.Distinct(*vertices))

        return {}

    # calculate edge distances
    @constraint
    def constrain_distances(distance_func: DistanceFunc = distance) -> DebugVars:
        distance_limits = [
            min_max_edge_length(
                epsilon,
                original_figure_points[p1],
                original_figure_points[p2],
                distance_func=distance_func,
            )
            for p1, p2 in edges
        ]
        distance_values = [
            distance_func(figure_points[p1], figure_points[p2]) for p1, p2 in edges
        ]
        # for limit, distance_var in list(zip(distance_limits, distance_vars))[5:7]:
        for limit, distance_value in zip(distance_limits, distance_values):
            # print(limit, distance_var)

            # assert limit.min <= exact_distance <= limit.max

            # Exact distances
            # opt.add_soft(distance_value == exact_distance)
            # opt.add(distance_value == exact_distance)

            # Min/max
            opt.add(distance_value >= limit.min)
            opt.add(distance_value <= limit.max)

            # opt.add(i-1 <= a)
            # opt.add(a <= i+1)
            # opt.assert_and_track(i == a, f"foo{i}")

        return {}

    @constraint
    def minimize_dislikes() -> DebugVars:
        min_dist = []
        for hole_idx, hole_point in enumerate(hole_points):
            dist = []
            for figured_idx, figure_point in enumerate(figure_points):
                dist.append(distance(hole_point, figure_point))

            b0 = dist[0]
            for b in dist[1:]:
                b0 = z3.If(b < b0, b, b0)
            min_dist.append(b0)

        # dislikes are the sum of squared distances
        # total_dislikes = sum(i * i for i in min_dist)
        total_dislikes = sum(z3.SignExt(len(min_dist), i) for i in min_dist)

        # opt.add(total_dislikes < 10000)
        # opt.add(total_dislikes < 3000)
        # opt.add(total_dislikes < 6000)

        if minimize:
            opt.minimize(total_dislikes)

        return {
            "total_dislikes": total_dislikes,
        }

    @constraint
    def virtual_points():
        min_hole_dist_points = []
        for idx, h in enumerate(problem.hole):
            p_x = z3.BitVec(f"hole_idx{idx}_dist_x", x_sort)
            p_y = z3.BitVec(f"hole_idx{idx}_dist_y", y_sort)

            vertex = mk_vertex(p_x, p_y)
            min_hole_dist_points.append(vertex)

            opt.add(z3.Or(*[vertex == figure_point for figure_point in vertices]))

            opt.minimize(distance(Point(p_x, p_y), h))

        opt.add(z3.Distinct(*min_hole_dist_points))

        # min_dislike_sum = sum(
        #     distance(Point(vertex_x(v), vertex_y(v)), h)
        #     for v, h in zip(min_hole_dist_points, problem.hole)
        # )

        # total_dislikes = z3.BitVec("dislikes", min_dislike_sum.size())

        # opt.add(total_dislikes == min_dislike_sum)
        # opt.add(total_dislikes < 6000)
        # opt.minimize(total_dislikes)

        return {}

    constraints: List[Constraint] = [
        constrain_to_xy_in_hole,
        constrain_to_edges_in_hole,
        constrain_unique_positions.disable,
        minimize_dislikes,
        virtual_points.disable,
        constrain_distances.disable,
    ]

    debug_vars = {}
    for c in constraints:
        print(f"Adding constraint: {c.name}")

        t0 = time.perf_counter()

        debug_vars.update(**c())

        print("constraint added... checking...")

        res = opt.check()

        t1 = time.perf_counter()

        total = datetime.timedelta(seconds=t1 - t0)

        print("Result", res, "- elapsed time:", total)

        # if str(res) != z3.sat:
        #     core = opt.unsat_core()
        #     print(core["foo20"])
        #     print(core)

        if res == z3.unsat:
            break

    if res != z3.sat:
        print(
            to_json(
                Output(problem=problem, solution=Solution([]), map_points=map_points)
            )
        )
        raise Exception("Failed to solve!")

    model = opt.model()
    for k, v in debug_vars.items():
        print(f"{k}: {model.eval(v)}")

    print(model)

    pose: Pose = [
        Point(model.eval(vertex_x(v)).as_long(), model.eval(vertex_y(v)).as_long())
        for v in vertices
    ]

    solution: Solution = Solution(vertices=pose)
    output = Output(problem=problem, solution=solution, map_points=map_points)
    print("Solution:")

    print(to_json(output))
    return output


@click.command()
@click.argument("problem_number")
@click.option("--minimize/--no-minimize", default=False)
@click.option("--debug/--no-debug", default=False)
def run(problem_number: int, minimize: bool, debug: bool) -> Output:
    from z3 import set_option

    set_option("parallel.enable", True)
    set_option("parallel.threads.max", 32)

    return _run(problem_number, minimize, debug)


if __name__ == "__main__":
    run()
