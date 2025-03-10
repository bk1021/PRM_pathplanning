import numpy as np
import pylab as pl
import argparse
from utils.environment_2d import Environment
from utils.prm import PRM

def options():
  parser = argparse.ArgumentParser(description="PRM Path Planner options")
  parser.add_argument('--num_nodes', type=int, help="Number of nodes sampled", default=1000)
  parser.add_argument('--R', type=int, help="Maximum edge length", default=0.5)
  parser.add_argument('--x_size', type=int, help="Environment X size", default=10)
  parser.add_argument('--y_size', type=int, help="Environment Y size", default=6)
  parser.add_argument('--obs', type=int, help="Number of obstacles", default=5)
  parser.add_argument('--plot_nodes', help="Plot nodes sampled", action="store_true")
  parser.add_argument('--plot_edges', help="Plot edges", action="store_true")
  parser.add_argument('--shortcut', help="Path shortcutting", action="store_true")
  parser.add_argument('--iter', type=int, help="Path shortcutting iterations", default=5)

  return parser.parse_args()


def main():
  opt=options()
  pl.ion()
  np.random.seed(4)
  env = Environment(opt.x_size, opt.y_size, opt.obs)
  pl.clf()
  env.plot()
  q = env.random_query()

  if q is not None:
    x_start, y_start, x_goal, y_goal = q
    env.plot_query(x_start, y_start, x_goal, y_goal)

  print("Environment simulated.")

  prm = PRM(env, (x_start,y_start), (x_goal,y_goal), opt)
  path = prm.find_path()
  prm.plot_prm(opt.plot_nodes, opt.plot_edges, path)
  
  pl.ioff()
  pl.show()

if __name__ == "__main__":
  main()
