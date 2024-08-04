import numpy as np
import pylab as pl
from utils.environment_2d import Environment
from utils.prm import PRM

def main():
  pl.ion()
  np.random.seed(4)
  env = Environment(10, 6, 5)
  pl.clf()
  env.plot()
  q = env.random_query()

  if q is not None:
    x_start, y_start, x_goal, y_goal = q
    env.plot_query(x_start, y_start, x_goal, y_goal)

  print("Environment simulated.")

  prm = PRM(env, (x_start,y_start), (x_goal,y_goal))
  path = prm.find_path()
  shortcut_path = prm.pathshortcut(path)
  prm.plot_prm(path=path, shortcut_path=shortcut_path)
  
  pl.ioff()
  pl.show()

if __name__ == "__main__":
  main()
