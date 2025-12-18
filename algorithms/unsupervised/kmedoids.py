import numpy as np

class KMedoids:
    """
    K-Medoids Clustering Algorithm - Implemented from scratch
    """
    
    def __init__(self, n_clusters=3, max_iterations=100, random_state=42):
        """
        Initialize K-Medoids
        
        Parameters:
        - n_clusters: Number of clusters (k)
        - max_iterations: Maximum number of iterations
        - random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.medoid_indices = None
        self.labels = None
        self.inertia_ = None
        
    def fit(self, X):
        """
        Fit K-Medoids to the data
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        """
        np.random.seed(self.random_state)
        
        # Randomly initialize medoids
        self.medoid_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        
        for iteration in range(self.max_iterations):
            # Store old medoids to check for convergence
            old_medoids = self.medoid_indices.copy()
            
            # Assign each point to nearest medoid
            self.labels = self._assign_clusters(X)
            
            # Update medoids
            self._update_medoids(X)
            
            # Check for convergence
            if np.array_equal(old_medoids, self.medoid_indices):
                print(f"Converged at iteration {iteration + 1}")
                break
        
        # Calculate inertia
        self.inertia_ = self._calculate_inertia(X)
        
        return self
    
    def _assign_clusters(self, X):
        """
        Assign each data point to the nearest medoid
        """
        distances = np.zeros((len(X), self.n_clusters))
        
        for k in range(self.n_clusters):
            # Calculate Euclidean distance to each medoid
            medoid = X[self.medoid_indices[k]]
            distances[:, k] = np.sqrt(np.sum((X - medoid) ** 2, axis=1))
        
        # Return the index of the closest medoid for each point
        return np.argmin(distances, axis=1)
    
    def _update_medoids(self, X):
        """
        Update medoids by selecting the point with minimum total distance to other points in cluster
        """
        for k in range(self.n_clusters):
            cluster_indices = np.where(self.labels == k)[0]
            
            if len(cluster_indices) > 0:
                # Calculate total distance from each point to all other points in cluster
                min_cost = float('inf')
                best_medoid_idx = self.medoid_indices[k]
                
                for candidate_idx in cluster_indices:
                    candidate = X[candidate_idx]
                    # Calculate total distance to all points in cluster
                    distances = np.sum(np.sqrt(np.sum((X[cluster_indices] - candidate) ** 2, axis=1)))
                    
                    if distances < min_cost:
                        min_cost = distances
                        best_medoid_idx = candidate_idx
                
                self.medoid_indices[k] = best_medoid_idx
    
    def _calculate_inertia(self, X):
        """
        Calculate inertia (within-cluster sum of squares)
        """
        inertia = 0
        for k in range(self.n_clusters):
            medoid = X[self.medoid_indices[k]]
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                distances = np.sum((cluster_points - medoid) ** 2, axis=1)
                inertia += np.sum(distances)
        return inertia
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        """
        return self._assign_clusters(X)
    
    def get_medoids(self, X):
        """
        Return the medoids
        """
        return X[self.medoid_indices]
