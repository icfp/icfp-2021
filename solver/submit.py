import os

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


def submit(problem_id: int) -> Identifier:
    solution = _run(problem_id).solution
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
    return response


@click.command()
@click.argument("problem_id")
def submit_problem(problem_id: int) -> Identifier:
    return submit(problem_id)
