import numpy as np
import math
from matplotlib import pyplot as plt

class Sweeper():
    """
    A class that creates a simple vehicle dispatch route using the 
    Gillet Miller Sweep approach.

    This is used for initial estimations in the genetic algorithm.

    """

    def __init__(self, depot, nodes):
        """
        Instantiates a new instance of the Sweeper class

        Args:
            depot(``list``): coords of the depot
            nodes(``list of lists``): list of nodes to connect
        """
        self.depot = depot
        self.nodes = nodes 
        self.route = set()

    def sort_nodes(self):
        """
        Sorts the nodes in self.nodes by doing an anticlockwise 
        sweep around self.depot
        """

        self.route.add((0.0, self.depot))
        self.route.add((2 * np.pi, self.depot))

        for node in self.nodes:
            angle = self.calculate_angle(self.depot, node)
            self.route.add((angle, node))

        self.route = sorted(self.route, key=lambda x: x[0])  

    def plot_route(self):
        """
        Plots the ordered route starting and ending at the depot
        """    

        plt.scatter(self.depot[0], self.depot[1])
        prev_node = self.depot

        for _, node in self.route: 
            plt.plot([prev_node[0], node[0]], [prev_node[1], node[1]])
            plt.scatter(node[0], node[1])
            prev_node = node

        plt.show()

    def calculate_unit_vector(self, node):
        """
        Safely determined the unit vector of node, accounting for the 
        case where the node is the origin.

        Args:
            node(``list``): coords of a node
        """

        node = list(node)

        if np.linalg.norm(node) == 0:
            return 0
        else:
            return node / np.linalg.norm(node)

    def calculate_angle(self, node1, node2):
        """
        Determines the radians between two nodes, anticlockwise from 
        node1.

        Args:
            node1(``list``): coords of the reference node. This is
                usually the depot
            node2(``list``): coords of the second node.
        """
        node1, node2 = list(node1), list(node2)
        xDiff = node1[0] - node2[0]
        yDiff = node1[1] - node2[1]

        if yDiff == 0:
            return np.pi if node1[0] > node2[0] else 0
        elif yDiff > 0:
            return -1 * math.atan2(yDiff, xDiff)
        else:
            return np.pi + math.atan2(yDiff, xDiff)
