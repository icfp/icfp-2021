from typing import TypedDict
import json
import os.path as path
import click

ROOT_DIR = path.abspath(path.join(path.dirname(__file__), '..'))
PROBLEMS_DIR = path.join(ROOT_DIR, 'problems')


Point = tuple[int, int]


class Figure(TypedDict):
    edges: list[Point]
    vertices: list[Point]


class Problem(TypedDict):
    hole: list[Point]
    epsilon: int
    figure: Figure


def load_problem(problem_number: int) -> Problem:
    return json.load(open(path.join(PROBLEMS_DIR, f'{problem_number}.json')))


@click.command()
@click.argument('problem_number')
def run(problem_number: int):
    print('Hello', problem_number, load_problem(problem_number))


if __name__ == "__main__":
    run()
