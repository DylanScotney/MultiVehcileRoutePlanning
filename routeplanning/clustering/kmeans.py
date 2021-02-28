from matplotlib import pyplot as plt
from matplotlib import cm as cm
import random
import numpy as np

class KMeansCluster():
    """
    Class that carries out a K means clustering (KNN) algorithm for
    a given set of nodes and clusters in an arbitrary number of
    dimensions

    Example usage:
    >>> k = KMeansCluster(num_clusters, nodes)
    ... k.run(iters=1000)

    Members:
        k(``int``): number of clusters 
        nodes(``set of tuples``): set of co-ordinates of each node
        self._clusters(``dict``): dict containing cluster infomation
        self._best_clusters(``dict``): dict of clusters with the least
            variance
    """

    def __init__(self, k, nodes, seed=None):
        """
        Instantiates a new instance of KMeansCluster
        """
        self.k = k
        self.nodes = nodes
        self._clusters = {}
        self._best_clusters = {}
        self.seed = seed
    
    def _assign_initial_clusters(self):
        """
        Randomly assigns k nodes as the initial estimates for each 
        cluster location
        """

        # wipe any existing clusters clean before assigning
        self._clusters = {}

        nodes = random.sample(self.nodes, self.k)
        for node in nodes:
            self._clusters[node] = set()

    def _assign_node_to_cluster(self, node):
        """
        Assigns a node to a the closest cluster

        Args:
            node(``tuple of floats``):     contains node coords
        """

        if node in set.union(*self._clusters.values()):
            raise RuntimeError("Node already assigned")

        for i, cluster in enumerate(self._clusters):
            dist = self.euclidean_dist(node, cluster)
            if i == 0:
                min_dist = dist
                closest_cluster = cluster
            elif dist < min_dist:
                min_dist = dist
                closest_cluster = cluster
            else:
                pass

        self._clusters[closest_cluster].add(node)

    def _build_clusters(self):
        """
        Assigns all the nodes to a closest cluster
        """
        for node in self.nodes:
            self._assign_node_to_cluster(node)

        return False

    def _reassign_clusters(self):
        """
        Reassigns cluster location to the mean postion of coords in the
        cluster.

        Returns:
            True/False to illustrate if the clustering algorithm has 
            finished. Will return False if at least one cluster centre
            point has been updated.

        To Do:
            - Improve efficency for when large number of new clusters
            are assigned
        """

        # Keep track of number of clusters that have switched coords
        switch_count = 0
        switched_clusters = []

        for cluster in self._clusters:            
            new_pos = self.average_coord(self._clusters[cluster])
            if new_pos not in self._clusters:
                switched_clusters += [[new_pos, cluster]]
                switch_count += 1

        for new_cluster, old_cluster in switched_clusters:
            self._clusters[new_cluster] = set()
            self._clusters.pop(old_cluster)

        # If we have moved at least one cluster wipe the others clean
        # This is inefficent for large number of new clusters as they 
        # are already clean
        if switch_count > 0:
            for cluster in self._clusters:
                self._clusters[cluster] = set()
            
            return False

        return True

    def _calculate_variance(self):
        """
        Calculates the total variance of all the clusters. This is used
        as a fitness measure of the final result.
        """

        var = 0.0

        for cluster in self._clusters:
            for node in self._clusters[cluster]:
                var += np.linalg.norm(np.subtract(node, cluster)) ** 2

        return var

    def _plot_clusters(self):
        """
        Scatter plots each cluster in a different colour

        To Do:
        - Generalise for multple dimensions
        """

        for cluster in self._best_clusters:
            nodes = self._best_clusters[cluster]
            x = [coords[0] for coords in nodes]
            y = [coords[1] for coords in nodes]
            plt.scatter(x, y)
        plt.show()

    def run(self, iters=1):
        """
        Runs k means clustering for given number of iterations and 
        gives the result with the lowest variance

        Args:
            iters(``int``, optional): number of iterations to complete
        """

        best_var = 0.0
        random.seed(self.seed)
        np.random.seed(self.seed)

        for i in range(iters):
            
            solved = False
            self._assign_initial_clusters()
            self._build_clusters()

            while not solved:
                solved = self._reassign_clusters()
                if not solved:
                    self._build_clusters()

            if i == 0:
                best_var = self._calculate_variance()
                self._best_clusters = self._clusters
            else:
                new_var = self._calculate_variance()
                if new_var < best_var:
                    best_var = new_var
                    self._best_clusters = self._clusters
            
            print(i, best_var)

            self._plot_clusters()

    @staticmethod
    def average_coord(nodes):
        """
        Calculates the average coord of input nodes of dimension N

        Args:
            nodes(``iterable``): Nodes containing coords of dimension N

        """
        num_nodes = len(nodes)
        dimensions = len(list(nodes)[0])
        ave = np.zeros(dimensions)

        for node in nodes:
            ave += node

        return tuple(ave / num_nodes)

    @staticmethod
    def euclidean_dist(n1, n2):
        """
        Determines the euclidean distance between two nodes in N
        dimensional space. 

        Args:
            n1(``iterable``): N dimensional iterable containing coord
                points
            n2(``iterable``): N dimensional iterable containing coord
                points
        """

        if(len(n1) != len(n2)):
            raise RuntimeError("""
                Nodes are not of the same dimensions. {N1}!={N2}.
            """.format(N1=len(n1), N2=len(n2))
            )

        return np.linalg.norm(np.subtract(n1, n2))    

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def best_clusters(self):
        return self.best_clusters

class MockClusterGenerator():
    """
    Class that handles and generates mock clusters and nodes in 2D

    Args:
        num_clusters(``int``): Number of clusters to generate
        nodes_per_cluster(``int``): Total nodes per cluster
        x(``float``): Max x coord
        y(``float``): Max y coord
        stddev(``float``): stddev of nodes around each cluster centre

    Example Usage:
    >>> mock_clusters = MockClusterGenerator(5, 40, 10, 10, 2)
    ... mock_clusters.generate(plot=True)
    """

    def __init__(self, num_clusters, nodes_per_cluster, x, y, stddev, seed=None):
        """
        Instantiates a new instance of MockClusterGenerator. 

        Args:
            num_clusters(``int``): Number of clusters to generate
            nodes_per_cluster(``int``): Total nodes per cluster
            x(``float``): Max x coord
            y(``float``): Max y coord
            stddev(``float``): stddev of nodes around each cluster centre
            seed(``int``): seed for random number generation
        """
        self.num_clusters = num_clusters
        self.nodes_per_cluster = nodes_per_cluster
        self.x = x
        self.y = y
        self.stddev = stddev
        self._clusters = dict()
        self._nodes = set()
        self.seed = seed

    def generate(self, plot=False):
        """
        Generates the clusters

        Args:
            plot(``bool``, optional): plots the clusters using pyplot
        """
        
        random.seed(self.seed)
        np.random.seed(self.seed)

        for _ in range(self.num_clusters):

            # get random unique coord for num_clusters
            coords = tuple([random.randint(0, self.x), random.randint(0, self.y)])

            while coords in self._clusters:
                coords = tuple([random.randint(0, self.x), random.randint(0, self.y)])

            self._clusters[coords] = set()                
            
            # assign normally distributed nodes to each cluster
            nodes = np.random.normal(0, self.stddev, size=(self.nodes_per_cluster, 2))

            for node in nodes:
                self._clusters[coords].add(tuple(np.add(node, coords)))

        if plot:
            for cluster in self._clusters:
                nodes = self._clusters[cluster]
                x = [coords[0] for coords in nodes]
                y = [coords[1] for coords in nodes]
                plt.scatter(x, y)
            plt.show()

    @property
    def clusters(self):
        return self._clusters

    @property
    def nodes(self):
        return [tuple(node) for cluster in self._clusters for node in self._clusters[cluster]]
