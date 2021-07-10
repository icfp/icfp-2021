from solver.app import Point
from solver.app import Hole


def in_polygon(p: Point, h: Hole) -> bool:
    # From https://www.baeldung.com/cs/geofencing-point-inside-polygon
    inside = False
    pairs = zip(h, h[1:] + [h[0]])
    for pair in pairs:
        # Are both y coordinates of the vertices either above or below the point's y?
        if not (
            (pair[0].y < p.y and pair[1].y < p.y)
            or (
                pair[0].y > p.y
                and pair[1].y > p.y
                or (pair[0].y == p.y and pair[1].y == p.y)
            )
        ):
            # Compute sx
            ratio = (p.y - pair[0].y) / (pair[1].y - pair[0].y)
            product = (pair[1].x - pair[0].x) * ratio
            sx = pair[0].x + product
            if p.x > sx:
                inside = not inside

    return inside
