import pandas.api.types as pdt
import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px
import plotly.figure_factory as ff

# --- Custom DIANA From-Scratch Implementation ---

def _split_cluster(X_cluster, indices, metric):
    """
    Splits a single cluster into two using the from-scratch DIANA splinter group method.
    """
    n_pts = X_cluster.shape[0]
    if n_pts <= 1:
        return None, None

    dissimilarity_matrix = squareform(pdist(X_cluster, metric=metric))
    avg_dissimilarity = np.mean(dissimilarity_matrix, axis=1)
    splinter_seed_idx = np.argmax(avg_dissimilarity)

    splinter_group_indices = {splinter_seed_idx}
    old_party_indices = {i for i in range(n_pts) if i != splinter_seed_idx}

    while True:
        moved_a_point = False
        points_to_move = set()

        for i in old_party_indices:
            d_to_splinter = np.mean([dissimilarity_matrix[i, j] for j in splinter_group_indices]) if splinter_group_indices else np.inf
            
            other_old_party_indices = old_party_indices - {i}
            d_to_old_party = np.mean([dissimilarity_matrix[i, j] for j in other_old_party_indices]) if other_old_party_indices else 0

            if d_to_splinter < d_to_old_party:
                points_to_move.add(i)
                moved_a_point = True

        if moved_a_point:
            splinter_group_indices.update(points_to_move)
            old_party_indices.difference_update(points_to_move)
        else:
            break
            
    new_cluster_1_orig_indices = [indices[i] for i in old_party_indices]
    new_cluster_2_orig_indices = [indices[i] for i in splinter_group_indices]
    
    return new_cluster_1_orig_indices, new_cluster_2_orig_indices


class Diana:
    """
    From-scratch DIANA implementation for generating flat cluster labels.
    """
    def __init__(self, n_clusters=2, metric='euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.labels_ = None
        self.clusters = []

    def fit(self, X):
        n_samples = X.shape[0]
        if n_samples == 0:
            self.labels_ = np.array([])
            self.clusters = []
            return

        # Start with all samples in one cluster
        self.clusters = [list(range(n_samples))]

        # Keep splitting clusters until we reach the desired number
        while len(self.clusters) < self.n_clusters:
            max_diameter = -1
            cluster_to_split_idx = -1

            # 1. Find the cluster with the largest diameter to split next
            for i, cluster_indices in enumerate(self.clusters):
                if len(cluster_indices) > 1:
                    pairwise_distances = pdist(X[cluster_indices], metric=self.metric)
                    if pairwise_distances.size > 0:
                        current_diameter = np.max(pairwise_distances)
                        if current_diameter > max_diameter:
                            max_diameter = current_diameter
                            cluster_to_split_idx = i

            if cluster_to_split_idx == -1:
                break # No more clusters can be split

            # 2. Perform the split using the from-scratch method
            cluster_to_split_orig_indices = self.clusters[cluster_to_split_idx]
            data_to_split = X[cluster_to_split_orig_indices]
            
            new_c1, new_c2 = _split_cluster(data_to_split, cluster_to_split_orig_indices, self.metric)
            
            # 3. Update the list of clusters
            self.clusters.pop(cluster_to_split_idx)
            if new_c1: self.clusters.append(new_c1)
            if new_c2: self.clusters.append(new_c2)

        # 4. Assign final labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, indices in enumerate(self.clusters):
            self.labels_[indices] = cluster_id

    def predict(self, X):
        if self.labels_ is None:
            raise RuntimeError("Fit the model first before predicting.")
        return self.labels_

# --- Streamlit Integration ---

def create_histogram(data):
    fig_hist = px.histogram(data, x='Cluster', title='Cluster Distribution', color='Cluster', color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_hist.update_layout(xaxis_title="Cluster", yaxis_title="Number of data points", height=400)
    return fig_hist

def create_dendrogram_plotly(linkage_matrix, labels):
    fig = ff.create_dendrogram(linkage_matrix, labels=labels)
    fig.update_layout(title='DIANA Dendrogram', xaxis_title="Data Points", yaxis_title="Distance", height=600)
    return fig

@st.cache_data
def run_diana(data, params):
    if data.shape[1] < 2:
        st.error("The dataset must have at least two features for 2D visualization.")
        return None, None, None, None, None

    col_x = params.get('col_x', data.columns[0])
    col_y = params.get('col_y', data.columns[1])

    # Ensure selected columns are numeric and handle NaNs
    if not pdt.is_numeric_dtype(data[col_x]) or not pdt.is_numeric_dtype(data[col_y]):
        st.error(f"Selected columns '{col_x}' and '{col_y}' must be numeric for DIANA clustering.")
        return None, None, None, None, None
    
    # Drop rows with NaN values in the selected columns
    original_rows = data.shape[0]
    data.dropna(subset=[col_x, col_y], inplace=True)
    if data.shape[0] == 0:
        st.warning(f"No valid data points left after dropping rows with missing values in '{col_x}' and '{col_y}'. Cannot perform clustering.")
        return None, None, None, None, None
    if data.shape[0] < original_rows:
        st.info(f"Dropped {original_rows - data.shape[0]} rows due to missing values in selected columns: '{col_x}' and '{col_y}'.")

    if data.shape[0] < 2:
        st.error("Not enough data points remaining after preprocessing for DIANA clustering.")
        return None, None, None, None, None

    X = data[[col_x, col_y]].values.astype(np.float64)
    n_clusters = params.get('n_clusters', 3)

    # --- From-Scratch Clustering for Scatter Plot & Metrics ---
    diana = Diana(n_clusters=n_clusters)
    diana.fit(X)
    labels = diana.predict(X)
    data['Cluster'] = labels
    
    # --- Dendrogram Generation using stable library function ---
    # This provides a correct and non-buggy dendrogram of the same data.
    try:
        # Ensure linkage uses the same X data used for clustering
        linkage_matrix = linkage(X, method='ward') # 'ward' is a common default for library dendrograms
        dendro_labels = data.index.astype(str) if len(data) < 100 else None
        fig_dendrogram = create_dendrogram_plotly(linkage_matrix, dendro_labels)
    except Exception as e:
        st.warning(f"Could not generate dendrogram: {e}")
        fig_dendrogram = None

    # --- Scatter Plot (using from-scratch labels) ---
    fig_scatter = px.scatter(
        data, 
        x=col_x, 
        y=col_y, 
        color='Cluster',
        title=f'DIANA Clustering (k={n_clusters})',
        color_continuous_scale=px.colors.qualitative.Vivid
    )
    fig_scatter.update_layout(showlegend=True, legend_title_text='Cluster', height=600)

    # --- Histogram ---
    fig_hist = create_histogram(data)
    
    # --- Evaluation Metrics ---
    metrics = {}
    n_clusters_found = len(set(labels))
    if 1 < n_clusters_found < X.shape[0]:
        metrics['Silhouette Score'] = silhouette_score(X, labels)
        metrics['Davies-Bouldin Index'] = davies_bouldin_score(X, labels)
        metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(X, labels)

    return data, fig_scatter, fig_dendrogram, fig_hist, metrics
