from solver.app import Point
from solver.app import Hole


# Check if a point is inside a closed polygon
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


# Given three colinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def on_segment(p: Point, q: Point, r: Point):
    if (
        (q.x <= max(p.x, r.x))
        and (q.x >= min(p.x, r.x))
        and (q.y <= max(p.y, r.y))
        and (q.y >= min(p.y, r.y))
    ):
        return True
    return False


def orientation(p: Point, q: Point, r: Point):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if val > 0:

        # Clockwise orientation
        return 1
    elif val < 0:

        # Counterclockwise orientation
        return 2
    else:

        # Colinear orientation
        return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def do_intersect(p1: Point, q1: Point, p2: Point, q2: Point):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0) and on_segment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if (o2 == 0) and on_segment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0) and on_segment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0) and on_segment(p2, q1, q2):
        return True

    # If none of the cases
    return False
