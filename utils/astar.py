import numpy as np
import heapq

class astar_search():
    def __init__(self, nodes, edges, start, goal):
        self.nodes = nodes
        self.edges = edges
        self.start = start
        self.goal = goal

    def distance(self, node1, node2):
        return np.linalg.norm(np.array(node1) - np.array(node2))

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