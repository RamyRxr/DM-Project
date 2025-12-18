import numpy as np

class DIANA:
    """
    DIANA (DIvisive ANAlysis) - Hierarchical Clustering Algorithm
    Implemented from scratch - Top-down divisive clustering
    """
    
    def __init__(self, n_clusters=3):
        """
        Initialize DIANA
        
        Parameters:
        - n_clusters: Number of clusters to form
        """
        self.n_clusters = n_clusters
        self.labels = None
        self.dendrogram_data = []
        
    def fit(self, X):
        """
        Fit DIANA to the data
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        """
        n_samples = len(X)
        
        # Start with all points in one cluster
        clusters = [list(range(n_samples))]
        
        # Split clusters until we have n_clusters
        while len(clusters) < self.n_clusters:
            # Find cluster with largest diameter
            cluster_to_split_idx = self._find_largest_diameter_cluster(X, clusters)
            cluster_to_split = clusters[cluster_to_split_idx]
            
            # If cluster has only one point, stop
            if len(cluster_to_split) <= 1:
                break
            
            # Split the cluster
            cluster1, cluster2 = self._split_cluster(X, cluster_to_split)
            
            # Store split information for dendrogram
            self.dendrogram_data.append({
                'original_cluster': cluster_to_split.copy(),
                'cluster1': cluster1,
                'cluster2': cluster2,
                'diameter': self._calculate_diameter(X, cluster_to_split)
            })
            
            # Replace old cluster with two new clusters
            clusters.pop(cluster_to_split_idx)
            clusters.append(cluster1)
            clusters.append(cluster2)
        
        # Assign labels
        self.labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels[point_idx] = cluster_id
        
        return self
    
    def _find_largest_diameter_cluster(self, X, clusters):
        """
        Find the cluster with the largest diameter (max distance between any two points)
        """
        max_diameter = -1
        largest_cluster_idx = 0
        
        for idx, cluster in enumerate(clusters):
            if len(cluster) > 1:
                diameter = self._calculate_diameter(X, cluster)
                if diameter > max_diameter:
                    max_diameter = diameter
                    largest_cluster_idx = idx
        
        return largest_cluster_idx
    
    def _calculate_diameter(self, X, cluster):
        """
        Calculate diameter of a cluster (maximum distance between any two points)
        """
        max_dist = 0
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                dist = np.sqrt(np.sum((X[cluster[i]] - X[cluster[j]]) ** 2))
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    
    def _split_cluster(self, X, cluster):
        """
        Split a cluster into two sub-clusters
        """
        if len(cluster) <= 1:
            return cluster, []
        
        # Find the point with maximum average distance to all other points
        max_avg_dist = -1
        splinter_point = cluster[0]
        
        for point_idx in cluster:
            avg_dist = 0
            for other_idx in cluster:
                if point_idx != other_idx:
                    avg_dist += np.sqrt(np.sum((X[point_idx] - X[other_idx]) ** 2))
            avg_dist /= (len(cluster) - 1)
            
            if avg_dist > max_avg_dist:
                max_avg_dist = avg_dist
                splinter_point = point_idx
        
        # Initialize splinter group and old party
        splinter_group = [splinter_point]
        old_party = [p for p in cluster if p != splinter_point]
        
        # Iteratively move points from old party to splinter group
        changed = True
        while changed and len(old_party) > 0:
            changed = False
            
            for point_idx in old_party[:]:  # Iterate over a copy
                # Calculate average distance to splinter group
                dist_to_splinter = np.mean([
                    np.sqrt(np.sum((X[point_idx] - X[s]) ** 2))
                    for s in splinter_group
                ])
                
                # Calculate average distance to old party
                if len(old_party) > 1:
                    dist_to_old = np.mean([
                        np.sqrt(np.sum((X[point_idx] - X[o]) ** 2))
                        for o in old_party if o != point_idx
                    ])
                else:
                    dist_to_old = float('inf')
                
                # Move to splinter if closer
                if dist_to_splinter < dist_to_old:
                    old_party.remove(point_idx)
                    splinter_group.append(point_idx)
                    changed = True
        
        return old_party, splinter_group
    
    def fit_predict(self, X):
        """
        Fit and return cluster labels
        """
        self.fit(X)
        return self.labels
    
    def get_dendrogram_data(self):
        """
        Return dendrogram split information
        """
        return self.dendrogram_data
