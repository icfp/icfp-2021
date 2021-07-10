from pathlib import Path
from typing import Iterable, List, NamedTuple
import click
import math
import z3
from pydantic.dataclasses import dataclass


ROOT_DIR = Path(__file__).parent.parent
PROBLEMS_DIR = ROOT_DIR / "problems"


class Point(NamedTuple):
    x: int
    y: int


Hole = List[Point]
VertexIndex = int


class Edge(NamedTuple):
    source: VertexIndex
    target: VertexIndex


@dataclass(frozen=True)
class EdgeLengthRange:
    min: float
    max: float


@dataclass(frozen=True)
class Figure:
    edges: List[Edge]
    vertices: List[Point]


@dataclass(frozen=True)
class Problem:
    epsilon: int
    hole: Hole
    figure: Figure


def min_max_edge_length(
    epsilon: int, edge: Edge, vertices: List[Point]
) -> EdgeLengthRange:
    max_ratio = epsilon / 1000000
    edge_length = distance(vertices[edge.source], vertices[edge.target])
    min_length = edge_length * (1 - max_ratio)
    max_length = edge_length * (1 + max_ratio)

    return EdgeLengthRange(min=min_length, max=max_length)


def load_problem(problem_number: int) -> Problem:
    return Problem.__pydantic_model__.parse_file(
        PROBLEMS_DIR / f"{problem_number}.json"
    )


def distance(p1: Point, p2: Point):
    delta_x = p1.x - p2.x
    delta_y = p1.y - p2.y
    return (delta_x * delta_x) + (delta_y * delta_y)


def dislikes(hole: Hole, pose: Figure):
    return sum(min(distance(v, h) for v in pose["vertices"]) for h in hole)


def bits_for(xs: Iterable[int]) -> int:
    return int(math.log2(max(xs)) + 1)


@click.command()
@click.argument("problem_number")
def run(problem_number: int):
    p = load_problem(problem_number)

    opt = z3.Optimize()
    x_sort = z3.BitVecSort(bits_for(i.x for i in p.hole))
    y_sort = z3.BitVecSort(bits_for(i.y for i in p.hole))

    # translate vertices to z3
    vertices = p.figure.vertices
    xs = [z3.BitVec(f"x_{i.x}", x_sort) for i in vertices]
    ys = [z3.BitVec(f"y_{i.y}", y_sort) for i in vertices]

    vertex, mk_vertex, (vertex_x, vertex_y) = z3.TupleSort("vertex", (x_sort, y_sort))
    vertices = [mk_vertex(x, y) for x, y in zip(xs, ys)]

    # add a distinct constraint on the vertex points
    opt.add(z3.Distinct(*vertices))

    edges = p.figure.edges
    print(edges)

    # b = opt.maximize(foo)
    # print(b)

    res = opt.check()  # == sat
    print(res)

    model = opt.model()
    for v in vertices:
        x = model.eval(vertex_x(v))
        y = model.eval(vertex_y(v))
        print(f"{x},{y}")

    # print(model)
    # print(model[foo])


if __name__ == "__main__":
    run()
