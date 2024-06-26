from typing import Any, Callable, Dict, List, NamedTuple

import z3
from pydantic.dataclasses import dataclass


class Point(NamedTuple):
    x: int
    y: int


Hole = List[Point]
Pose = List[Point]
VertexIndex = int
DebugVars = Dict[str, z3.AstRef]
InHoleLookup = Dict[Point, bool]
ConstraintFunc = Callable[[], DebugVars]
DistanceFunc = Callable[[Point, Point], int]


class Edge(NamedTuple):
    source: VertexIndex
    target: VertexIndex


class EdgeSegment(NamedTuple):
    source: Point
    target: Point


class InclusiveRange(NamedTuple):
    start: int
    end: int


class YPointRange(NamedTuple):
    x: int
    y_inclusive_ranges: List[InclusiveRange]


@dataclass(frozen=True)
class EdgeLengthRange:
    min: int
    max: int


@dataclass(frozen=True)
class Figure:
    edges: List[Edge]
    vertices: List[Point]


@dataclass(frozen=True)
class Problem:
    epsilon: int
    hole: Hole
    figure: Figure


@dataclass(frozen=True)
class Solution:
    vertices: Pose


@dataclass(frozen=True)
class Identifier:
    id: str


@dataclass(frozen=True)
class Output:
    problem: Any
    solution: Solution
    map_points: List[List[int]]
