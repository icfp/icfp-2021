from solver.app import Point
from solver.app import Hole


def in_polygon(p: Point, h: Hole) -> bool:
    inside = False
    pairs = zip(h, h[1:] + [h[0]])
    for pair in pairs:
        # Are both y coordinates of the vertices either above or below the point's y?
        if not (
            (pair[0][1] < p[1] and pair[1][1] < p[1])
            or (
                pair[0][1] > p[1]
                and pair[1][1] > p[1]
                or (pair[0][1] == p[1] and pair[1][1] == p[1])
            )
        ):
            # Compute sx
            ratio = (p[1] - pair[0][1]) / (pair[1][1] - pair[0][1])
            product = (pair[1][0] - pair[0][0]) * ratio
            sx = pair[0][0] + product
            if p[0] > sx:
                inside = not inside

    return inside
