import numpy as np

class DBSCAN:
    """
    DBSCAN Clustering Algorithm - Implemented from scratch
    Density-Based Spatial Clustering of Applications with Noise
    """
    
    def __init__(self, eps=0.5, min_pts=5):
        """
        Initialize DBSCAN
        
        Parameters:
        - eps: Maximum distance between two points to be considered neighbors
        - min_pts: Minimum number of points to form a dense region (core point)
        """
        self.eps = eps
        self.min_pts = min_pts
        self.labels = None
        
    def fit(self, X):
        """
        Fit DBSCAN to the data
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        """
        n_samples = len(X)
        self.labels = np.full(n_samples, -1)  # Initialize all as noise (-1)
        
        cluster_id = 0
        
        for point_idx in range(n_samples):
            # Skip if already processed
            if self.labels[point_idx] != -1:
                continue
            
            # Find neighbors
            neighbors = self._get_neighbors(X, point_idx)
            
            # If not a core point, mark as noise (will be updated if border point)
            if len(neighbors) < self.min_pts:
                continue
            
            # Start new cluster
            self.labels[point_idx] = cluster_id
            
            # Expand cluster
            self._expand_cluster(X, neighbors, cluster_id)
            
            cluster_id += 1
        
        return self
    
    def _get_neighbors(self, X, point_idx):
        """
        Find all points within eps distance from point_idx
        """
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        neighbors = np.where(distances <= self.eps)[0]
        return neighbors
    
    def _expand_cluster(self, X, neighbors, cluster_id):
        """
        Expand cluster from seed points
        """
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If noise, change to border point
            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = cluster_id
            
            # If unprocessed
            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = cluster_id
                
                # Find neighbors of this point
                new_neighbors = self._get_neighbors(X, neighbor_idx)
                
                # If core point, add its neighbors to the cluster
                if len(new_neighbors) >= self.min_pts:
                    neighbors = np.concatenate([neighbors, new_neighbors])
                    neighbors = np.unique(neighbors)
            
            i += 1
    
    def fit_predict(self, X):
        """
        Fit and return cluster labels
        """
        self.fit(X)
        return self.labels
    
    def get_core_samples(self, X):
        """
        Return indices of core samples
        """
        core_samples = []
        for i in range(len(X)):
            if self.labels[i] >= 0:  # Not noise
                neighbors = self._get_neighbors(X, i)
                if len(neighbors) >= self.min_pts:
                    core_samples.append(i)
        return np.array(core_samples)
