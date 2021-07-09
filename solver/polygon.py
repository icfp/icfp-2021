from solver.app import Point, load_problem
from solver.app import Hole

def in_polygon(p : Point, h : Hole) -> bool:
  inside = False
  pairs = zip(h, h[1:] + [h[0]])
  for pair in pairs:
    # Are both y coordinates of the vertices either above or below the point's y?
    if (pair[0][1] < p[1] and pair[1][1] < p[1]) or (pair[0][1] > p[1] and pair[1][1] > p[1] or (pair[0][1] == p[1] and pair[1][1] == p[1])):
      continue
    else:
      # Compute sx
      ratio = (p[1] - pair[0][1]) / (pair[1][1] - pair[0][1])
      product = (pair[1][0] - pair[0][0]) * ratio
      sx = pair[0][0] + product
      if p[0] > sx:
        inside = not inside

  return inside

def test():
  problem = load_problem(1)
  print(in_polygon((100, 100), problem["hole"]))
  print(in_polygon((30, 7), problem["hole"]))
  print(in_polygon((30, 5), problem["hole"]))
  print(in_polygon((70, 95), problem["hole"]))
  print(in_polygon((95, 95), problem["hole"]))