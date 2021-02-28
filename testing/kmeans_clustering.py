from routeplanning.clustering.kmeans import MockClusterGenerator, KMeansCluster

mock_clusters = MockClusterGenerator(5, 40, 35, 35, 1.5, seed=5)
mock_clusters.generate(plot=True)

nodes = mock_clusters.nodes

k = KMeansCluster(5, nodes)
k.run(iters=10)
