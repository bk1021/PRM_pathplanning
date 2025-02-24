import numpy as np
import pylab as pl
import random
import signal
from tqdm import tqdm
from matplotlib.lines import Line2D
from environment_2d import Environment

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.children = []
        self.cost = 0.0

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, index):
        return (self.x, self.y)[index]

class RRT:
    def __init__(self, env, start, goal, expand_dist=1.0, goal_bias_percent=5, max_iter=1000):
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
        self.edge_plot = {}

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

    def steer(self, from_node, to_node, extend_length=float("inf"), animate=False, color='orange'):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.x += min(extend_length, d) * np.cos(theta)
        new_node.y += min(extend_length, d) * np.sin(theta)

        if not self.env.check_collision_line(from_node, new_node):
            self.update_parent(new_node, from_node, rewire=False, animate=animate, color=color)
            return new_node
        return None

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta

    def get_nearby_nodes(self, tree, new_node):
        nearby_nodes = []
        for node in tree:
            d, _ = self.calc_distance_and_angle(node, new_node)
            if node != new_node and d <= self.expand_dist:
                nearby_nodes.append(node)
        return nearby_nodes

    def update_parent(self, node, parent, rewire=False, animate=False, color='orange'):
        if rewire:
            node.parent.children.remove(node)
            red_line, = pl.plot([node.parent.x, node.x], [node.parent.y, node.y], color='tomato', linewidth=2)
            if animate:
                pl.pause(0.01)
            self.edge_plot[(node.parent, node)].remove()
            red_line.remove()

        node.parent = parent
        parent.children.append(node)
        node.cost = parent.cost + self.calc_distance_and_angle(parent, node)[0]
        
        if rewire:
            green_line, = pl.plot([parent.x, node.x], [parent.y, node.y], color='springgreen', linewidth=2)
            if animate:
                pl.pause(0.01)
            green_line.remove()
            line, = pl.plot([parent.x, node.x], [parent.y, node.y], color=color, linewidth=0.5)
            self.edge_plot[(parent, node)] = line

        if not rewire:
            line, = pl.plot([parent.x, node.x], [parent.y, node.y], color=color, linewidth=0.5)
            self.edge_plot[(parent, node)] = line
            pl.plot(node.x, node.y, "o", color=color, markersize=2)
            if animate:
                pl.pause(0.01)

    # def update_children(self, node, children, animate=False):
    #     for child in children:
    #         self.update_parent(child, node, rewire=False, animate=animate)

    def update_subsequent_cost(self, tree, node):
        for child in node.children:
            d, _ = self.calc_distance_and_angle(node, child)
            child.cost = node.cost + d
            self.update_subsequent_cost(tree, child)

    def rewire(self, tree, node, animate=False, color='orange'):
        nearby_nodes = self.get_nearby_nodes(tree, node)
        # Rewire parent
        rewire_parent = False
        best_parent = node.parent
        min_cost = node.cost
        for parent in nearby_nodes:
            d, _ = self.calc_distance_and_angle(parent, node)
            if parent.cost + d < min_cost and not self.env.check_collision_line(parent, node):
                rewire_parent = True
                best_parent = parent
                min_cost = parent.cost + d
        if rewire_parent:
            self.update_parent(node, best_parent, rewire=True, animate=animate, color=color)
        # Rewire children
        for child in nearby_nodes:
            d, _ = self.calc_distance_and_angle(node, child)
            if node.cost + d < child.cost and not self.env.check_collision_line(node, child):
                self.update_parent(child, node, rewire=True, animate=animate, color=color)
        
        self.update_subsequent_cost(tree, node)

    
    def check_connection(self, node1, node2):
        return self.calc_distance_and_angle(node1, node2)[0] <= 2*self.expand_dist \
        and not self.env.check_collision_line(node1, node2)
    
    def generate_path(self, last_node):
        path = []
        path_cost = last_node.cost
        while last_node:
            path.append((last_node.x, last_node.y))
            last_node = last_node.parent
        return path[::-1], path_cost

    def plot_path(self, path):
        plots = []
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            dot, = pl.plot(x0, y0, "o", color="green", markersize=3)
            plots.append(dot)
            line, = pl.plot([x0, x1], [y0, y1], color='green', linewidth=1.5)
            plots.append(line)
        dot, = pl.plot(path[-1][0], path[-1][1], "o", color="green", markersize=3)
        plots.append(dot)
        return plots

    def planning(self, bidir=False, star=False, sample_max=False, animate=True):
        best_path = None
        min_path_cost = float('inf')
        connection_node = []

        for _ in tqdm(range(self.max_iter)):
            rnd_node = self.get_random_node()
            nearest_idx_a = self.get_nearest_node_index(self.tree_a, rnd_node)
            nearest_node_a = self.tree_a[nearest_idx_a]
            new_node_a = self.steer(nearest_node_a, rnd_node, self.expand_dist, animate=animate, color=self.color_a)

            if new_node_a:
                self.tree_a.append(new_node_a)

                if star:
                    self.rewire(self.tree_a, new_node_a, animate=animate, color=self.color_a)

                if not bidir:
                    if self.check_connection(new_node_a, self.goal):
                        if self.goal not in self.tree_a:
                            self.update_parent(self.goal, new_node_a)
                            self.tree_a.append(self.goal)
                            path, path_cost = self.generate_path(self.goal)
                            path_plots = self.plot_path(path)
                            if not sample_max:
                                return path, path_cost
                            else:
                                best_path = path
                                min_path_cost = path_cost
                        elif new_node_a.cost + self.calc_distance_and_angle(new_node_a, self.goal)[0] < self.goal.cost:
                            self.update_parent(self.goal, new_node_a, rewire=True, animate=animate, color=self.color_a)

                    if best_path:
                        path, path_cost = self.generate_path(self.goal)
                        if path_cost < min_path_cost:
                            best_path = path
                            min_path_cost = path_cost
                            for plot in path_plots:
                                plot.remove()
                            path_plots = self.plot_path(best_path)

                else:
                    nearest_idx_b = self.get_nearest_node_index(self.tree_b, new_node_a)
                    nearest_node_b = self.tree_b[nearest_idx_b]
                    new_node_b = self.steer(nearest_node_b, new_node_a, self.expand_dist, animate=animate, color=self.color_b)

                    if new_node_b:
                        self.tree_b.append(new_node_b)

                        if star:
                            self.rewire(self.tree_b, new_node_b, animate=animate, color=self.color_b)

                        if self.check_connection(new_node_a, new_node_b):
                            if len(connection_node) == 0:
                                if new_node_a[0] == new_node_b[0] and new_node_a[1] == new_node_b[1]:
                                    pl.plot([new_node_a.x, new_node_b.x], [new_node_a.y, new_node_b.y], "o", color="brown", markersize=3)
                                else:
                                    pl.plot([new_node_a.x, new_node_b.x], [new_node_a.y, new_node_b.y], color="brown", linewidth=0.8)
                                if animate: 
                                    pl.pause(0.01)
                                if self.root_a == self.start:
                                    connection_node.append([new_node_a, new_node_b])
                                    path_a, path_cost_a = self.generate_path(new_node_a)
                                    path_b, path_cost_b = self.generate_path(new_node_b)
                                    path_b = path_b[::-1]
                                else:
                                    connection_node.append([new_node_b, new_node_a])
                                    path_a, path_cost_a = self.generate_path(new_node_b)
                                    path_b, path_cost_b = self.generate_path(new_node_a)
                                    path_b = path_b[::-1]     
                                path = path_a + path_b
                                path_cost = path_cost_a + path_cost_b + self.calc_distance_and_angle(new_node_a, new_node_b)[0]
                                path_plots = self.plot_path(path)       
                                if not sample_max:
                                    return path, path_cost
                                else:
                                    best_path = path
                                    min_path_cost = path_cost
                        
                            if self.root_a == self.start:
                                connection = [new_node_a, new_node_b]
                            else:
                                connection = [new_node_b, new_node_a]

                            if connection not in connection_node:
                                if self.root_a == self.start:
                                    connection_node.append([new_node_a, new_node_b])
                                else:
                                    connection_node.append([new_node_b, new_node_a])
                                if new_node_a[0] == new_node_b[0] and new_node_a[1] == new_node_b[1]:
                                    pl.plot([new_node_a.x, new_node_b.x], [new_node_a.y, new_node_b.y], "o", color="brown", markersize=3)
                                else:
                                    pl.plot([new_node_a.x, new_node_b.x], [new_node_a.y, new_node_b.y], color="brown", linewidth=0.8)
                                if animate: 
                                    pl.pause(0.01)
                        
                        if best_path:
                            for node_a, node_b in connection_node:
                                path_a, path_cost_a = self.generate_path(node_a)
                                path_b, path_cost_b = self.generate_path(node_b)
                                path_b = path_b[::-1]
                                path = path_a + path_b
                                path_cost = path_cost_a + path_cost_b + self.calc_distance_and_angle(new_node_a, new_node_b)[0]
                                if path_cost < min_path_cost:
                                    best_path = path
                                    min_path_cost = path_cost
                                    for plot in path_plots:
                                        plot.remove()
                                    path_plots = self.plot_path(best_path)
                        
                    self.tree_a, self.tree_b = self.tree_b, self.tree_a
                    self.root_a, self.root_b = self.root_b, self.root_a
                    self.color_a, self.color_b = self.color_b, self.color_a
                    self.bias_node = self.start if self.bias_node == self.goal else self.goal
        
        if bidir:
            for connection in connection_node:
                connection[0].parent = connection[1]
                connection[1].children.append(connection[0])
                node_list = self.tree_a + self.tree_b
        else:
            node_list = self.tree_a
        return best_path, min_path_cost, node_list

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
        path, path_cost, node_list = rrt.planning(animate=True, bidir=True, star=False, sample_max=False)
        if path is None:
            print("No path found!")
        else:
            print("Path found!")
            print("Path cost:", path_cost)
    else:
        print("Failed to generate a valid start and goal.")
    pl.ioff()
    pl.show()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()