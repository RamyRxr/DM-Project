import numpy as np

class KMeans:
    """
    K-Means Clustering Algorithm - Implemented from scratch
    """
    
    def __init__(self, n_clusters=3, max_iterations=100, random_state=42):
        """
        Initialize K-Means
        
        Parameters:
        - n_clusters: Number of clusters (k)
        - max_iterations: Maximum number of iterations
        - random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
    def fit(self, X):
        """
        Fit K-Means to the data
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        """
        np.random.seed(self.random_state)
        
        # Randomly initialize centroids
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices].copy()
        
        for iteration in range(self.max_iterations):
            # Store old centroids to check for convergence
            old_centroids = self.centroids.copy()
            
            # Assign each point to nearest centroid
            self.labels = self._assign_clusters(X)
            
            # Update centroids
            self._update_centroids(X)
            
            # Check for convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged at iteration {iteration + 1}")
                break
        
        # Calculate inertia (sum of squared distances to nearest centroid)
        self.inertia_ = self._calculate_inertia(X)
        
        return self
    
    def _assign_clusters(self, X):
        """
        Assign each data point to the nearest centroid
        """
        distances = np.zeros((len(X), self.n_clusters))
        
        for k in range(self.n_clusters):
            # Calculate Euclidean distance to each centroid
            distances[:, k] = np.sqrt(np.sum((X - self.centroids[k]) ** 2, axis=1))
        
        # Return the index of the closest centroid for each point
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X):
        """
        Update centroids as the mean of all points in each cluster
        """
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            
            if len(cluster_points) > 0:
                self.centroids[k] = np.mean(cluster_points, axis=0)
    
    def _calculate_inertia(self, X):
        """
        Calculate inertia (within-cluster sum of squares)
        """
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                distances = np.sum((cluster_points - self.centroids[k]) ** 2, axis=1)
                inertia += np.sum(distances)
        return inertia
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        """
        return self._assign_clusters(X)
    
    def get_centroids(self):
        """
        Return the centroids
        """
        return self.centroids
