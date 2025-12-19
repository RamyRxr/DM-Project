import pandas.api.types as pdt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import fcluster
import plotly.figure_factory as ff
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# --- Custom AGNES Implementation ---

def _get_linkage_distance(cluster1_pts, cluster2_pts, method):
    """Calculates the distance between two clusters based on the linkage method."""
    distances = np.linalg.norm(cluster1_pts[:, np.newaxis] - cluster2_pts, axis=2)
    
    if method == 'single':
        return np.min(distances)
    elif method == 'complete':
        return np.max(distances)
    elif method == 'average':
        return np.mean(distances)
    elif method == 'ward':
        # This is a simplified placeholder, not the true Ward's method.
        # True Ward's method minimizes the increase in variance.
        centroid1 = np.mean(cluster1_pts, axis=0)
        centroid2 = np.mean(cluster2_pts, axis=0)
        return np.linalg.norm(centroid1 - centroid2)
    else:
        raise ValueError(f"Unknown linkage method: {method}")

def agnes_manual(X, linkage_method='ward'):
    """
    Performs AGNES clustering from scratch and builds the full linkage matrix.
    
    Returns:
        - linkage_matrix: The linkage matrix for dendrogram plotting.
    """
    n_samples = X.shape[0]
    
    # Start with each sample in its own cluster
    clusters = {i: [i] for i in range(n_samples)}
    cluster_points = {i: X[i:i+1] for i in range(n_samples)}
    
    linkage_matrix = []
    next_cluster_id = n_samples

    # Loop until only one cluster remains to build the full hierarchy
    while len(clusters) > 1:
        min_dist = np.inf
        closest_pair = (-1, -1)
        
        # Find the two closest clusters
        cluster_ids = list(clusters.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                id1 = cluster_ids[i]
                id2 = cluster_ids[j]
                
                dist = _get_linkage_distance(cluster_points[id1], cluster_points[id2], linkage_method)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (id1, id2)
        
        id1, id2 = closest_pair
        
        # Ensure id1 is smaller than id2 for consistent linkage matrix
        if id1 > id2:
            id1, id2 = id2, id1

        # Get original indices and points for the new merged cluster
        merged_indices = clusters[id1] + clusters[id2]
        merged_points = np.vstack([cluster_points[id1], cluster_points[id2]])
        
        # Record the merge in the linkage matrix
        # Format: [cluster_id_1, cluster_id_2, distance, num_samples_in_new_cluster]
        linkage_row = [float(id1), float(id2), min_dist, len(merged_indices)]
        linkage_matrix.append(linkage_row)

        # Remove the old clusters and their points
        del clusters[id1]
        del clusters[id2]
        del cluster_points[id1]
        del cluster_points[id2]

        # Add the new merged cluster
        clusters[next_cluster_id] = merged_indices
        cluster_points[next_cluster_id] = merged_points
        
        next_cluster_id += 1
        
    return np.array(linkage_matrix)

# --- Plotting and Execution ---

def create_histogram(data):
    """Creates a histogram of the cluster distribution."""
    fig_hist = px.histogram(
        data, 
        x='Cluster',
        title='Cluster Distribution',
        color='Cluster',
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_hist.update_layout(xaxis_title="Cluster", yaxis_title="Number of data points", height=400)
    return fig_hist

def create_dendrogram_plotly(linkage_matrix, labels):
    """Creates a dendrogram using Plotly's figure factory."""
    fig = ff.create_dendrogram(linkage_matrix, labels=labels)
    fig.update_layout(
        title='AGNES Dendrogram',
        xaxis_title="Data Points",
        yaxis_title="Distance",
        height=600
    )
    return fig

@st.cache_data
def run_agnes(data, params):
    """
    Runs the custom AGNES algorithm and returns the results.
    """
    col_x = params.get('col_x', data.columns[0])
    col_y = params.get('col_y', data.columns[1])

    # Ensure selected columns are numeric and handle NaNs
    if not pdt.is_numeric_dtype(data[col_x]) or not pdt.is_numeric_dtype(data[col_y]):
        st.error(f"Selected columns '{col_x}' and '{col_y}' must be numeric for AGNES clustering.")
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
        st.error("Not enough data points remaining after preprocessing for AGNES clustering.")
        return None, None, None, None, None

    X = data[[col_x, col_y]].values.astype(np.float64)
    
    n_clusters = params.get('n_clusters', 3)
    linkage_method = params.get('linkage', 'ward')
    
    if linkage_method == 'ward' and X.shape[1] > 1:
        st.warning("The from-scratch 'ward' implementation is a simplified approximation (distance between centroids) and may not match standard libraries.")

    # --- Run custom AGNES to get the full linkage matrix ---
    linkage_matrix = agnes_manual(X, linkage_method=linkage_method)
    
    # --- Get flat cluster labels by "cutting" the tree ---
    # fcluster returns 1-indexed labels, so we subtract 1
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
    data['Cluster'] = labels
    
    # --- Dendrogram ---
    # The dendrogram can be large; create labels only if the dataset is small
    dendro_labels = None
    if len(data) < 100:
        dendro_labels = data.index.astype(str)
    fig_dendrogram = create_dendrogram_plotly(linkage_matrix, dendro_labels)

    # --- Scatter Plot ---
    fig_scatter = px.scatter(
        data, 
        x=col_x, 
        y=col_y, 
        color='Cluster',
        title=f'AGNES Clustering (k={n_clusters}, Linkage: {linkage_method})',
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
