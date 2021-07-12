from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .app import ROOT_DIR, _run, compute_statistics, load_problem, make_in_hole_matrix
from .types import Output, Solution

app = FastAPI()


@app.get("/api/problems/{number}")
async def get_problem(number: int):
    return Output(
        problem=load_problem(number), solution=Solution(vertices=[]), map_points=[]
    )


@app.get("/api/solve/{number}")
async def solve_problem(number: int):
    return _run(number, minimize=True)


@app.get("/api/points/{number}")
async def get_map_points(number: int):
    problem = load_problem(number)

    stats = compute_statistics(problem)

    in_hole_map = make_in_hole_matrix(stats, problem)

    map_points = [[point.x, point.y] for point, inside in in_hole_map.items() if inside]
    return Output(
        problem=load_problem(number),
        solution=Solution(vertices=[]),
        map_points=map_points,
    )


app.mount(
    "/", StaticFiles(directory=f"{ROOT_DIR}/visualizer/", html=True), name="static"
)
