from routeplanning.clustering.kmeans import MockClusterGenerator

mock_clusters = MockClusterGenerator(5, 40, 35, 35, 1.5, seed=5)
mock_clusters.generate(plot=True)
