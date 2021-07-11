from .types import Hole, Point


# Check if a point is on a line
def on_line(p: Point, start: Point, end: Point):
    dxc = p.x - start.x
    dyc = p.y - start.y
    dxl = end.x - start.x
    dyl = end.y - start.y
    cross = dxc * dyl - dyc * dxl
    if cross != 0:
        return False

    if abs(dxl) >= abs(dyl):
        if dxl > 0:
            return start.x <= p.x and p.x <= end.x
        else:
            return end.x <= p.x and p.x <= start.x
    else:
        if dyl > 0:
            return start.y <= p.y and p.y <= end.y
        else:
            return end.y <= p.y and p.y <= start.y


# Check if a point is inside a closed polygon
def in_polygon(p: Point, h: Hole) -> bool:
    # From https://www.baeldung.com/cs/geofencing-point-inside-polygon
    # But this algorithm assumes the point on the polygon is "outside" but this
    # doesn't work for us. So we will first test if the point is on the polygon.
    inside = False
    pairs = zip(h, h[1:] + [h[0]])

    for (fst, snd) in pairs:
        if on_line(p, fst, snd):
            return True

    # Pairs is going to start with the last edge (which is_first_element will skip ray testing on) to prime the state
    # pairs = zip([h[len(h)-1]] + h, h + [h[0]])
    pairs = zip(h, h[1:] + [h[0]])
    hit_end_vertex = False
    previous_start = Point(0, 0)

    for (index, (fst, snd)) in enumerate(pairs):
        # Are both y coordinates of the vertices either above or below the point's y?
        if (fst.y < p.y and snd.y < p.y) or (fst.y > p.y and snd.y > p.y):
            continue
        else:
            # Test whether all the points are on the same Y line
            if fst.y == p.y == snd.y:
                if (fst.x <= p.x <= snd.x) or (snd.x <= p.x <= fst.x):
                    return True  # on the x edge between the two points
            # Test whether all the points are on the same X line
            elif fst.x == p.x == snd.x:
                if (fst.y <= p.y <= snd.y) or (snd.y <= p.y <= fst.y):
                    return True  # on the y edge between the two points

            # Skip all horizontal lines, but remember the previous state
            # This is because horizontal lines don't affect "inside-ness"
            if fst.y == snd.y:
                continue

            # Test whether we hit the second point. This means we hit the vertex at the end of the line.
            # We will remember this, and not test intersection.
            # However, only do this if we are not the last element
            if snd.y == p.y and snd.x < p.x and index < (len(h) - 1):
                print("End Vertex hits ray")
                hit_end_vertex = True
                previous_start = fst
                continue

            # Test whether the previous point was on a vertex
            if hit_end_vertex:
                # Ok - so now we need to check whether this line is on the same plane as the previous line
                # we compare previous_start with snd
                # If they are on the same side of the ray, ignore.
                # If they cross the ray, drop below and test the intersection.
                hit_end_vertex = False
                if previous_start.y >= p.y and snd.y >= p.y:
                    continue
                elif previous_start.y <= p.y and snd.y <= p.y:
                    continue
            else:
                # Otherwise, if the ray crosses fst (this can really only happen at the start of the iteration)
                # Ignore this hit.
                if fst.y == p.y and fst.x < p.x:
                    continue

            # If we are the last element and we hit the end vertex, check what the first element
            # is, and see if both edges are on the same side. If they are, skip. Otherwise, lets test if
            # the ray crosses this last edge. This is a manual lookahead that is captured in the state for all the other
            # iterations.
            if snd.y == p.y and index == (len(h) - 1):
                # We have to skip ahead of any horizontal lines
                idx = 1
                next_snd = h[idx]
                while snd.y == next_snd.y:
                    idx += 1
                    next_snd = h[idx]

                if (fst.y < p.y and next_snd.y < p.y) or (fst.y > p.y and next_snd.y > p.y):
                    continue

            # Compute sx
            ratio = (p.y - fst.y) / (snd.y - fst.y)
            product = (snd.x - fst.x) * ratio
            sx = fst.x + product
            if p.x > sx:
                print(fst)
                print(snd)
                print("flip")
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
