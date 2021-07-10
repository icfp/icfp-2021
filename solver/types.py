from typing import List, NamedTuple
from pydantic.dataclasses import dataclass


class Point(NamedTuple):
    x: int
    y: int


Hole = List[Point]
VertexIndex = int


class Edge(NamedTuple):
    source: VertexIndex
    target: VertexIndex


@dataclass(frozen=True)
class EdgeLengthRange:
    min: float
    max: float


@dataclass(frozen=True)
class Figure:
    edges: List[Edge]
    vertices: List[Point]


@dataclass(frozen=True)
class Problem:
    epsilon: int
    hole: Hole
    figure: Figure