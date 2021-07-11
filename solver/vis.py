from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .app import ROOT_DIR, _run, load_problem
from .types import Output, Solution

app = FastAPI()


@app.get("/api/problems/{number}")
async def get_problem(number):
    return Output(
        problem=load_problem(number), solution=Solution(vertices=[]), map_points=[]
    )


@app.get("/api/solve/{number}")
async def solve_problem(number):
    return _run(number, minimize=True)


app.mount(
    "/", StaticFiles(directory=f"{ROOT_DIR}/visualizer/", html=True), name="static"
)
