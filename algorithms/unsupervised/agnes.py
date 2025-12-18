import numpy as np

class AGNES:
    """
    AGNES (AGglomerative NESting) - Hierarchical Clustering Algorithm
    Implemented from scratch with Single, Complete, and Average linkage
    """
    
    def __init__(self, n_clusters=3, linkage='single'):
        """
        Initialize AGNES
        
        Parameters:
        - n_clusters: Number of clusters to form
        - linkage: Linkage criterion ('single', 'complete', 'average')
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels = None
        self.dendrogram_data = []
        
    def fit(self, X):
        """
        Fit AGNES to the data
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        """
        n_samples = len(X)
        
        # Initialize each point as its own cluster
        clusters = [[i] for i in range(n_samples)]
        
        # Calculate initial distance matrix
        dist_matrix = self._calculate_distance_matrix(X, clusters)
        
        # Merge clusters until we have n_clusters
        while len(clusters) > self.n_clusters:
            # Find two closest clusters
            i, j = self._find_closest_clusters(dist_matrix)
            
            # Store merge information for dendrogram
            self.dendrogram_data.append({
                'cluster1': clusters[i].copy(),
                'cluster2': clusters[j].copy(),
                'distance': dist_matrix[i][j]
            })
            
            # Merge clusters
            clusters[i].extend(clusters[j])
            clusters.pop(j)
            
            # Update distance matrix
            dist_matrix = self._calculate_distance_matrix(X, clusters)
        
        # Assign labels
        self.labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels[point_idx] = cluster_id
        
        return self
    
    def _calculate_distance_matrix(self, X, clusters):
        """
        Calculate distance matrix between all clusters
        """
        n_clusters = len(clusters)
        dist_matrix = np.full((n_clusters, n_clusters), np.inf)
        
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = self._cluster_distance(X, clusters[i], clusters[j])
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
        
        return dist_matrix
    
    def _cluster_distance(self, X, cluster1, cluster2):
        """
        Calculate distance between two clusters based on linkage criterion
        """
        distances = []
        
        for i in cluster1:
            for j in cluster2:
                dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                distances.append(dist)
        
        if self.linkage == 'single':
            # Single linkage: minimum distance
            return np.min(distances)
        elif self.linkage == 'complete':
            # Complete linkage: maximum distance
            return np.max(distances)
        elif self.linkage == 'average':
            # Average linkage: average distance
            return np.mean(distances)
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")
    
    def _find_closest_clusters(self, dist_matrix):
        """
        Find the two closest clusters
        """
        min_dist = np.inf
        closest_i, closest_j = 0, 1
        
        for i in range(len(dist_matrix)):
            for j in range(i + 1, len(dist_matrix)):
                if dist_matrix[i][j] < min_dist:
                    min_dist = dist_matrix[i][j]
                    closest_i, closest_j = i, j
        
        return closest_i, closest_j
    
    def fit_predict(self, X):
        """
        Fit and return cluster labels
        """
        self.fit(X)
        return self.labels
    
    def get_dendrogram_data(self):
        """
        Return dendrogram merge information
        """
        return self.dendrogram_data
