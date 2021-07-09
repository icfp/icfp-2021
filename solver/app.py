from typing import TypedDict
import json
import os.path as path
import click

ROOT_DIR = path.abspath(path.join(path.dirname(__file__), '..'))
PROBLEMS_DIR = path.join(ROOT_DIR, 'problems')


Point = tuple[int, int]
Hole = list[Point]
VertexIndex = int


class Figure(TypedDict):
    edges: list[tuple[VertexIndex, VertexIndex]]
    vertices: list[Point]


class Problem(TypedDict):
    hole: Hole
    epsilon: int
    figure: Figure


def load_problem(problem_number: int) -> Problem:
    return json.load(open(path.join(PROBLEMS_DIR, f'{problem_number}.json')))


def distance(p1: Point, p2: Point):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def dislikes(hole: Hole, pose: Figure):
    return sum(min(distance(v, h) for v in pose["vertices"]) for h in hole)


@click.command()
@click.argument('problem_number')
def run(problem_number: int):
    print('Hello', problem_number, load_problem(problem_number))


if __name__ == "__main__":
    run()
