from pathlib import Path
from typing import List, NamedTuple
import click
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


@dataclass
class Figure:
    edges: List[Edge]
    vertices: List[Point]


@dataclass
class Problem:
    hole: Hole
    epsilon: int
    figure: Figure


def load_problem(problem_number: int) -> Problem:
    return Problem.__pydantic_model__.parse_file(
        PROBLEMS_DIR / f"{problem_number}.json"
    )


def distance(p1: Point, p2: Point):
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2


def dislikes(hole: Hole, pose: Figure):
    return sum(min(distance(v, h) for v in pose["vertices"]) for h in hole)


@click.command()
@click.argument("problem_number")
def run(problem_number: int):
    print("Hello", problem_number, load_problem(problem_number))


if __name__ == "__main__":
    run()
