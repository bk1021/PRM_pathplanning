import numpy as np
from numpy.polynomial import Polynomial
import heapq
import pylab as pl

class PRM(object):
    # define PRM path controller object
    def __init__(self, env, start, goal, opt):
        self.env = env
        self.start = start
        self.goal = goal
        self.num_nodes = opt.num_nodes
        self.edge_length = opt.R
        self.shortcut = opt.shortcut
        self.iter = opt.iter
        self.nodes = [start, goal]
        self.edges = []
        # print PRM info
        print("\nPRM object setting:")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Number of nodes: {opt.num_nodes}")
        print(f"Edge Length: <={opt.R}")
        if opt.shortcut:
            print(f"Shortcut: Enabled ({opt.iter} iterations)\n")
        else:
            print("Shortcut: Disabled\n")


    def sample_nodes(self):
        # samples random nodes within the environment that are collision-free.
        max_attempts = 1.5*self.num_nodes
        attempts = 0
        print("Sampling node...")

        while len(self.nodes) < self.num_nodes and attempts < max_attempts:
            x = np.random.rand()*self.env.size_x
            y = np.random.rand()*self.env.size_y
            if not self.env.check_collision(x, y) and (x, y) not in self.nodes:
                self.nodes.append((x, y))
            attempts += 1
        
        if attempts >= max_attempts:
            print("Warning: Reached maximum attempts while sampling nodes.")
            print(f"Number of nodes sampled: {len(self.nodes)}")
        else:
            print(f"Sampling completed with {len(self.nodes)} nodes.")
    
    def add_edge(self, node1, node2):
        # adds an edge between node1 and node2 if it does not exist and there are no collisions.
        if node1 != node2 and node2 not in [e[1] for e in self.edges if e[0] == node1]:
            if not self.env.check_collision_line(node1, node2):
                self.edges.append((node1, node2))

    def distance(self, node1, node2):
        return np.linalg.norm(np.array(node1) - np.array(node2))
    
    def find_path(self):
        # sample nodes, connect edge, and find path using astar search algorithm
        self.sample_nodes()
        print("Connecting nodes...")
        for node in self.nodes:
            # nodes are considered neighbours if they are within certain range
            neighbors = [n for n in self.nodes if self.distance(node, n) <= self.edge_length and n != node]
            for neighbor in neighbors:
                self.add_edge(node, neighbor)
        print("Nodes connected. Start finding path..")
        path = self.a_star()
        if path:
            print("Path found. Listing path nodes...")
            for i, node in enumerate(path):
                print(f"Node {i}: {node}")
            if self.shortcut:
                path = self.pathshortcut(path, self.iter)
        else:
            print("No path found.")

        return path


    def a_star(self):
        open_list = []
        heapq.heappush(open_list, (0, self.start))
        came_from = {}
        g_score = {node: float('inf') for node in self.nodes}
        g_score[self.start] = 0
        f_score = {node: float('inf') for node in self.nodes}
        f_score[self.start] = self.distance(self.start, self.goal)

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == self.goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                return path[::-1]

            for _, neighbor in [e for e in self.edges if e[0] == current]:
                tentative_g_score = g_score[current] + self.distance(current, neighbor)
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.distance(neighbor, self.goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return None
    

    def pathshortcut(self, path):
        # generate shortcut path by random sampling of two points on two adjacent edge if it is collision-free 
        print(f"Start path shortcutting. Total {self.iter} path shortcutting iterations.")
        new_path = path.copy()
        iter = 0
        while iter < self.iter:
            count = 0
            for i in range(0,len(path)-2,3):
                x1, x2, x3 = path[i][0], path[i+1][0], path[i+2][0]
                y1, y2, y3 = path[i][1], path[i+1][1], path[i+2][1]

                eq1 = Polynomial.fit([x1,x2],[y1,y2],1)
                eq2 = Polynomial.fit([x2,x3],[y2,y3],1)
                
                #generate random points on the line
                rand_x1 = np.random.uniform(min(x1,x2),max(x1,x2))
                rand_x2 = np.random.uniform(min(x2,x3),max(x2,x3))
                rand_y1 = eq1(rand_x1)
                rand_y2 = eq2(rand_x2)

                if not(self.env.check_collision_line((rand_x1,rand_y1),(rand_x2,rand_y2))):
                    new_path[i+count+1] = (rand_x1, rand_y1)
                    new_path.insert(i+count+2,(rand_x2, rand_y2))
                    count += 1
            print(f"Path shortcutting iteration {iter+1} completed. {count} shortcuts generated.")
            path = new_path.copy()
            iter += 1
        print("Path shortcutting completed. Listing new nodes...")
        for i, node in enumerate(new_path):
            print(f"Node {i}: {node}")
        return new_path

    def plot_prm(self, plot_nodes=False, plot_edges=False, path=None):
        # visualize PRM
        print("Generating PRM plot..")
        if plot_nodes:
            pl.scatter([node[0] for node in self.nodes], [node[1] for node in self.nodes], color='green', s=6, marker='o')

        if plot_edges:
            for edge in self.edges:
                (x1, y1), (x2, y2) = edge
                pl.plot([x1,x2], [y1,y2], "g" , linewidth = 0.5)

        if path:
            path_x = [node[0] for node in path]
            path_y = [node[1] for node in path]
            pl.plot(path_x, path_y, 'b-', lw=2)
        
        print("PRM plot generated.")



    



    
