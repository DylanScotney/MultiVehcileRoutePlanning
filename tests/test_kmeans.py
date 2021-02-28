
import numpy as np
from routeplanning.clustering.kmeans import KMeansCluster
from routeplanning.clustering.kmeans import MockClusterGenerator

def test_num_nodes():
    nodes = [[1,2], [1,4], [1,3], [4,5]]
    kmeans = KMeansCluster(3, nodes)

    assert kmeans.num_nodes == 4


def test_assign_intial_clusters():
    nodes = [(1,2), (1,3), (1,4), (4,5)]
    kmeans = KMeansCluster(3, nodes)

    kmeans._assign_initial_clusters()
    assert len(kmeans._clusters.keys()) == 3

def test_euclidean_distanct():

    assert KMeansCluster.euclidean_dist([-7, 11], [5, 6]) == 13.0
