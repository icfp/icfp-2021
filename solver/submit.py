# import requests
import click
import json
import os

from .app import internal_run
from pydantic.json import pydantic_encoder


if os.environ.get('API_TOKEN') is None:
    print("Must set API_TOKEN environment variable to submit a solution!")

HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['API_TOKEN']}"}


@click.command()
@click.argument("problem_id")
def submit_problem(problem_id: int) -> None:
    solution = internal_run(problem_id)
    print(f"Submitting solution {solution} for problem {problem_id}")

    # response = requests.post(f"https://poses.live/api/problems/{problem_id}/solutions",
    #               data=json.dumps(solution, default=pydantic_encoder),
    #               headers=HEADERS)

    # print(response)
