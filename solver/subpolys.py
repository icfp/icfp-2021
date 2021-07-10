from typing import List

import voronoi.diagram
import voronoi.point

from .types import Hole


def voronoi_polys(hole: Hole) -> List[voronoi.diagram.Edge]:
    dia = voronoi.diagram.Diagram()
    dia.construct([voronoi.point.Point(p.x, p.y) for p in hole], [])

    return dia.edges
