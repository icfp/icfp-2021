import os
from typing import Iterator

import click
import requests
from pydantic import parse_obj_as

from .app import _run
from .format import to_json
from .types import Identifier


def get_api_token() -> str:
    token = os.environ.get("API_TOKEN")
    if os.environ.get("API_TOKEN") is None:
        raise Exception("Must set API_TOKEN environment variable to submit a solution!")

    return str(token)


def submit(problem_id: int, timeout: int) -> Iterator[Identifier]:
    for output in _run(problem_id, minimize=True, timeout=timeout):
        solution = output.solution

        print(f"Submitting solution {solution.vertices} for problem {problem_id}")

        res = requests.post(
            f"https://poses.live/api/problems/{problem_id}/solutions",
            data=to_json(solution),
            headers={
                "Authorization": f"Bearer {get_api_token()}",
                "Content-Type": "application/json",
            },
        )

        res.raise_for_status()

        response = parse_obj_as(Identifier, res.json())
        print(f"Successfully submitted ({response.id})!")
        yield response


@click.command()
@click.argument("problem_id", type=click.INT)
@click.option("--timeout", default=5 * 60, type=click.INT)
def submit_problem(problem_id: int, timeout: int) -> Identifier:
    for pose in submit(problem_id, timeout=timeout * 1000):
        print(pose)
