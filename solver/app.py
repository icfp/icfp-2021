from pathlib import Path
from typing import Tuple, List
import click
from pydantic.dataclasses import dataclass


ROOT_DIR = Path(__file__).parent.parent
PROBLEMS_DIR = ROOT_DIR / "problems"


Point = Tuple[int, int]
Hole = List[Point]
VertexIndex = int


@dataclass
class Figure:
    edges: List[Tuple[VertexIndex, VertexIndex]]
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
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def dislikes(hole: Hole, pose: Figure):
    return sum(min(distance(v, h) for v in pose["vertices"]) for h in hole)


@click.command()
@click.argument("problem_number")
def run(problem_number: int):
    print("Hello", problem_number, load_problem(problem_number))


if __name__ == "__main__":
    run()
