import numpy as np
import pylab as pl
import random
import signal
from matplotlib.lines import Line2D
from environment_2d import Environment

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.neighbours = []

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, index):
        return (self.x, self.y)[index]

class RRT:
    def __init__(self, env, start, goal, expand_dist=1.0, goal_bias_percent=5, max_iter=5000):
        self.env = env
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.expand_dist = expand_dist
        self.goal_bias_percent = goal_bias_percent
        self.max_iter = max_iter
        # self.node_list = [self.start]
        self.bias_node = self.goal

        self.tree_a = [self.start]
        self.tree_b = [self.goal]
        self.root_a = self.start
        self.root_b = self.goal
        self.color_a = 'orange'
        self.color_b = 'magenta'

    def get_random_node(self, bias_goal=True):
        if random.randint(1, 100) > self.goal_bias_percent:
            rnd_x = np.random.rand()*self.env.size_x
            rnd_y = np.random.rand()*self.env.size_y
        else:
            rnd_x, rnd_y = self.bias_node.x, self.bias_node.y
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

        if not self.env.check_collision_line(from_node, new_node):
            new_node.parent = from_node
            new_node.neighbours.append(from_node)
            from_node.neighbours.append(new_node)
            return new_node
        return None

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta
    
    def check_connection(self, node1, node2):
        return self.calc_distance_and_angle(node1, node2)[0] <= self.expand_dist \
        and not self.env.check_collision_line(node1, node2)
    
    def generate_path(self, last_node):
        path = []
        while last_node:
            path.append((last_node.x, last_node.y))
            last_node = last_node.parent
        return path[::-1]

    def planning(self, bidir=False, animate=True):
        # best_path = None
        # best_path_length = float('inf')
        # best_path_lines = []

        for _ in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_idx_a = self.get_nearest_node_index(self.tree_a, rnd_node)
            nearest_node_a = self.tree_a[nearest_idx_a]
            new_node_a = self.steer(nearest_node_a, rnd_node, self.expand_dist)

            if new_node_a:
                self.tree_a.append(new_node_a)
                pl.plot([new_node_a.parent.x, new_node_a.x], [new_node_a.parent.y, new_node_a.y], color=self.color_a, linewidth=0.5)
                pl.plot(new_node_a.x, new_node_a.y, "o", color=self.color_a, markersize=2)
                if animate:
                    pl.pause(0.01)

                if not bidir:
                    if self.check_connection(new_node_a, self.goal):
                        self.goal.parent = new_node_a
                        self.goal.neighbours.append(new_node_a)
                        new_node_a.neighbours.append(self.goal)
                        path = self.generate_path(self.goal)
                        pl.plot([x for x,_ in path], [y for _, y in path], color='green', linewidth=1.5)
                        pl.plot([x for x,_ in path], [y for _, y in path], "o", color='green', markersize=3)
                        return path
                else:
                    nearest_idx_b = self.get_nearest_node_index(self.tree_b, new_node_a)
                    nearest_node_b = self.tree_b[nearest_idx_b]
                    new_node_b = self.steer(nearest_node_b, new_node_a, self.expand_dist)

                    if new_node_b:
                        self.tree_b.append(new_node_b)
                        pl.plot([new_node_b.parent.x, new_node_b.x], [new_node_b.parent.y, new_node_b.y], color=self.color_b, linewidth=0.5)
                        pl.plot(new_node_b.x, new_node_b.y, "o", color=self.color_b, markersize=2)
                        if animate:
                            pl.pause(0.01)

                        if self.check_connection(new_node_a, new_node_b):
                            if self.root_a == self.start:
                                path_a = self.generate_path(new_node_a)
                                path_b = self.generate_path(new_node_b)[::-1]
                            else:
                                path_a = self.generate_path(new_node_b)
                                path_b = self.generate_path(new_node_a)[::-1]
                            path = path_a + path_b
                            pl.plot([x for x,_ in path], [y for _, y in path], color='green', linewidth=1.5)
                            pl.plot([x for x,_ in path], [y for _, y in path], "o", color='green', markersize=3)
                            return path
                        
                    self.tree_a, self.tree_b = self.tree_b, self.tree_a
                    self.root_a, self.root_b = self.root_b, self.root_a
                    self.color_a, self.color_b = self.color_b, self.color_a
                    self.bias_node = self.start if self.bias_node == self.goal else self.goal
                        
                    # path_length = sum(np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) for i in range(len(path) - 1))
                    # if path_length < best_path_length:
                    #     best_path = path
                    #     best_path_length = path_length
                    #     # Remove previous best path
                    #     for line in best_path_lines:
                    #         line.remove()
                    #     best_path_lines = []
                    #     # Plot new best path
                    #     for i in range(len(path) - 1):
                    #         x0, y0 = path[i]
                    #         x1, y1 = path[i + 1]
                    #         dot, = pl.plot(x0, y0, "o", color="green", markersize=3)
                    #         best_path_lines.append(dot)
                    #         line, = pl.plot([x0, x1], [y0, y1], color='green', linewidth=1.5)
                    #         best_path_lines.append(line)
                    #     pl.plot(path[-1][0], path[-1][1], "o", color="green", markersize=3)
        
        # return best_path
        return None

def signal_handler(sig, frame):
    print("\nProgram interrupted! Exiting gracefully...")
    pl.close('all')
    exit(0)

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
        rrt = RRT(env, start, goal, expand_dist=0.5, goal_bias_percent=0, max_iter=3000)
        path = rrt.planning(animate=True, bidir=True)
        if path is None:
            print("No path found!")
        else:
            print("Path found!")
    else:
        print("Failed to generate a valid start and goal.")
    pl.ioff()
    pl.show()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()