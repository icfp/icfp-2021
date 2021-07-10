import json
import os

import click
import requests
from pydantic import parse_obj_as
from pydantic.json import pydantic_encoder

from .app import internal_run
from .types import Identifier


def get_api_token() -> str:
    token = os.environ.get("API_TOKEN")
    if os.environ.get("API_TOKEN") is None:
        raise Exception("Must set API_TOKEN environment variable to submit a solution!")

    return str(token)


def submit(problem_id: int) -> Identifier:
    solution = internal_run(problem_id)
    print(f"Submitting solution {solution.vertices} for problem {problem_id}")

    res = requests.post(
        f"https://poses.live/api/problems/{problem_id}/solutions",
        data=json.dumps(solution, default=pydantic_encoder),
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
