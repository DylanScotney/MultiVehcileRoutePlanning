import numpy as np
import math
from matplotlib import pyplot as plt

class Sweeper():
    """
    A class that creates a simple vehicle dispatch route using the 
    Gillet Miller Sweep approach.

    This is a candidate for initial population selection in the
    genetic algorithm.

    Example Usage:
    >>> sweeper = Sweeper(depot, nodes)
    ... sweeper.sort_nodes()
    ... sweeper.plot_route()

    Members: 
        self.depot(``iterable``): coords of the depot
        self.nodes(``2D iterable``): list of coords of nodes
        self.num_stops(``int``): number of nodes
        self.vehicles(``int``): number of delivery vehicles
        self.route(``list of RouteStops``): Full route with a single 
            vehicle and no depot stops.
        self.routes_with_depot(``list of RouteStops``):
            self.route split into vehicles and including depot stops

    """

    def __init__(self, depot, nodes, vehicles=1):
        """
        Instantiates a new instance of the Sweeper class

        Args:
            depot(``list``): coords of the depot
            nodes(``list of lists``): list of nodes to connect
            vehicles(``int``): number of vehicles to deliver
        """
        self.depot = depot
        self.nodes = nodes
        self.num_stops = len(nodes)
        self.vehicles = vehicles
        self.route = []
        self.routes_with_depot = []

    def _chunk(self, iterable, chunk_size):
        """
        Break iterable into chunks. Used to split up the route into 
        seperate drivers.

        Args:
            iterable(``iterable``)
            chunk_size(``int``): maximum size of a chunk

        To Do: 
            Break into more uniformly distributed chunk sizes. Very 
            very hard to do, P = NP... :(
        """
        
        for i in range(0, len(iterable), chunk_size):
            yield iterable[i : i + chunk_size]

    def _split_route(self, start=0):
        """
        Rearranges the route to start at index == start then chunks the 
        route into self.vehicles smaller routes and adds depot as the 
        first and last stop. 

        Args:
            start(``int``, optional): index of starting node
        """
        route = (self.route[start:self.num_stops] 
                   + self.route[:start])

        chunk_size = math.ceil(float(self.num_stops) / self.vehicles)

        return [self._get_route_with_depot(route)
                for route in self._chunk(route, chunk_size)]

    def _get_route_with_depot(self, route):
        """
        Constructs a route with depot as the first and last stops

        Args:
            route(``list of RouteStops``)
        """
        route = [self.RouteStop(0, self.depot)] + route
        route += [self.RouteStop(2 * np.pi, self.depot)]

        return route

    def _sort_nodes(self):
        """
        Sorts the nodes in order of anticlockwise angle around the depot
        and stores in self.route.

        This is the core of the sweeping algorithm.        
        """
        route = []

        for node in self.nodes:
            angle = self._calculate_angle(self.depot, node)
            stop = self.RouteStop(angle, node)
            route += [stop]

        # sort route by anticlockwise angle from depot
        self.route = sorted(route, key=lambda x: x.angle)

    def _calculate_angle(self, node1, node2):
        """
        Determines the radians between two nodes, anticlockwise from 
        node1.

        Args:
            node1(``list``): coords of the reference node. This is
                usually the depot
            node2(``list``): coords of the second node.
        """
        
        node1, node2 = list(node1), list(node2)
        xDiff = node2[0] - node1[0]
        yDiff = node2[1] - node1[1]

        if node1[1] == node2[1]:
            return np.pi if node1[0] > node2[0] else 0
        elif node2[1] > node1[1]:
            return math.atan2(yDiff, xDiff)
        else:
            return 2 * np.pi + math.atan2(yDiff, xDiff)

    def construct_route(self, start=0):
        """
        Constructs the full delivery route using a Gillet-Miller 
        Anticlockwise Sweep.

        Will split the route into self.vehciles sections, each one 
        starting and ending at the depot.

        Args:
            start(``int``, optional): index of starting node
        """
        self._sort_nodes()
        self.routes_with_depot = self._split_route(start=0)

    def plot_route(self):
        """
        Plots the entire route for all vehicles including depot stops.

        Does some pretty formatting so each vehicle is a different 
        colour.
        """

        routes = self.routes_with_depot
        colours = [x for x in 'bgrcmyk'] * len(routes)

        for i, route in enumerate(routes):
            colour = colours[i]
            self._plot_single_route(route, colour=colour)
        plt.show()

    def _plot_single_route(self, route, colour=None):
        """
        Plots an input route. Must call pyplot.show() after calling 
        in order to show graph

        Args:
            route(``list of RouteStops``)
        """    

        prev_stop = self.RouteStop(0, self.depot)

        for stop in self._get_route_with_depot(route): 
            plt.plot([prev_stop.x, stop.x],
                     [prev_stop.y, stop.y],
                     c=colour)

            plt.scatter(stop.x, stop.y, c=colour)

            prev_stop = stop


    class RouteStop(object):
        """
        Simple object that holds information about a stop on a route

        Args:
            angle(``float``): anticlockwise angle from depot
            node(``iterable``): coords of node
        """

        def __init__(self, angle, node):
            self._angle = angle
            self._node = node

        def __repr__(self):
            return "angle: {a}, node: {n}".format(self.angle, self.node)

        @property
        def angle(self):
            return self._angle
        
        @property
        def node(self):
            return self._node

        @property
        def x(self):
            return self._node[0]

        @property
        def y(self):
            return self._node[1]