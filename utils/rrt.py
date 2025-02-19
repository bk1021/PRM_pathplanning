import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import random
from matplotlib.lines import Line2D
from environment_2d import Environment

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, index):
        return (self.x, self.y)[index]

class RRT:
    def __init__(self, env, start, goal, expand_dist=1.0, goal_bias_percent=5, max_iter=500):
        self.env = env
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.expand_dist = expand_dist
        self.goal_bias_percent = goal_bias_percent
        self.max_iter = max_iter
        self.node_list = [self.start]

    def get_random_node(self):
        if random.randint(1, 100) > self.goal_bias_percent:
            rnd_x = np.random.rand()*self.env.size_x
            rnd_y = np.random.rand()*self.env.size_y
        else:
            rnd_x, rnd_y = self.goal.x, self.goal.y
        return Node(rnd_x, rnd_y)

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        min_index = dlist.index(min(dlist))
        return min_index

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.x += min(extend_length, d) * np.cos(theta)
        new_node.y += min(extend_length, d) * np.sin(theta)
        new_node.parent = from_node

        if not self.env.check_collision_line(from_node, new_node):
            return new_node
        return None

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta
    
    def goal_reachable(self, node):
        return self.calc_distance_and_angle(node, self.goal)[0] <= self.expand_dist \
        and not self.env.check_collision_line(node, self.goal)
    
    def generate_path(self, last_node):
        path = [(self.goal.x, self.goal.y)]
        parent_node = last_node
        while parent_node:
            path.append((parent_node.x, parent_node.y))
            parent_node = parent_node.parent
        return path[::-1]

    def planning(self):
        best_path = None
        best_path_length = float('inf')
        best_path_lines = []

        for _ in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_idx = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_idx]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dist)

            if new_node:
                self.node_list.append(new_node)
                pl.plot([new_node.parent.x, new_node.x], [new_node.parent.y, new_node.y], color="orange", linewidth=0.5)
                pl.plot(new_node.x, new_node.y, "o", color="orange", markersize=2)
                pl.pause(0.01)

                if self.goal_reachable(new_node):
                    path = self.generate_path(new_node)
                    path_length = sum(np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) for i in range(len(path) - 1))
                    if path_length < best_path_length:
                        best_path = path
                        best_path_length = path_length
                        # Remove previous best path
                        for line in best_path_lines:
                            line.remove()
                        best_path_lines = []
                        # Plot new best path
                        for i in range(len(path) - 1):
                            x0, y0 = path[i]
                            x1, y1 = path[i + 1]
                            dot, = pl.plot(x0, y0, "o", color="green", markersize=3)
                            best_path_lines.append(dot)
                            line, = pl.plot([x0, x1], [y0, y1], color='green', linewidth=1.5)
                            best_path_lines.append(line)
                        pl.plot(path[-1][0], path[-1][1], "o", color="green", markersize=3)
        
        return best_path

def main():   
    np.random.seed(4)
    env = Environment(10, 6, 5)
    env.plot()
    q = env.random_query()

    if q is not None:
        x_start, y_start, x_goal, y_goal = q
        env.plot_query(x_start, y_start, x_goal, y_goal)

        start = (x_start, y_start)
        goal = (x_goal, y_goal)
        rrt = RRT(env, start, goal, expand_dist=0.5, goal_bias_percent=5, max_iter=3000)
        path = rrt.planning()
        if path is None:
            print("No path found!")
        else:
            print("Path found!")
    else:
        print("Failed to generate a valid start and goal.")
    pl.ioff()
    pl.show()

if __name__ == '__main__':
    main()