import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px

# --- Custom DBSCAN From-Scratch Implementation ---

def dbscan_manual(X, eps, min_samples):
    """
    From-scratch implementation of the DBSCAN algorithm.
    """
    n_samples = X.shape[0]
    
    # -1 = Unvisited, -2 = Noise
    labels = np.full(n_samples, -1)
    cluster_id = 0

    def region_query(point_idx):
        """Finds all points within eps of a given point."""
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances < eps)[0]

    for i in range(n_samples):
        # Skip if point is already part of a cluster
        if labels[i] != -1:
            continue

        # Find neighbors
        neighbors = region_query(i)

        # If not enough neighbors, mark as noise (for now)
        if len(neighbors) < min_samples:
            labels[i] = -2 # Mark as Noise
            continue
        
        # We have a core point, start a new cluster
        labels[i] = cluster_id
        
        # Expand the cluster from this core point
        seed_set = set(neighbors)
        seed_set.remove(i) # Remove the core point itself from the seed set

        while seed_set:
            current_point_idx = seed_set.pop()
            
            # If this point was marked as noise, it's now a border point
            # If it's unvisited, it will be part of the current cluster
            if labels[current_point_idx] == -2 or labels[current_point_idx] == -1:
                labels[current_point_idx] = cluster_id
                
                # Check if this new point is also a core point
                new_neighbors = region_query(current_point_idx)
                if len(new_neighbors) >= min_samples:
                    # Add its neighbors to the seed set to be processed
                    seed_set.update(new_neighbors)
        
        # Move to the next cluster ID
        cluster_id += 1

    # Final labels: Noise points are labeled -1 as per sklearn convention
    labels[labels == -2] = -1
    return labels

# --- Streamlit Integration ---

def create_histogram(data):
    """Creates a histogram of the cluster distribution."""
    # Ensure 'Cluster' column is treated as a categorical variable for coloring
    data['Cluster'] = data['Cluster'].astype(str)
    
    fig_hist = px.histogram(
        data, 
        x='Cluster',
        title='Cluster Distribution (Noise points are in cluster -1)',
        color='Cluster',
        color_discrete_map={'-1': 'grey'}, # Color noise points grey
        category_orders={'Cluster': sorted(data['Cluster'].unique(), key=int)} # Sort cluster labels
    )
    fig_hist.update_layout(xaxis_title="Cluster", yaxis_title="Number of data points", height=400)
    return fig_hist

@st.cache_data
def run_dbscan(data, params):
    """
    Runs the from-scratch DBSCAN algorithm and returns the results.
    """
    col_x = params.get('col_x', data.columns[0])
    col_y = params.get('col_y', data.columns[1])

    # Ensure selected columns are numeric and handle NaNs
    if data[col_x].dtype not in [np.number] or data[col_y].dtype not in [np.number]:
        st.error(f"Selected columns '{col_x}' and '{col_y}' must be numeric for DBSCAN clustering.")
        return None, None, None, None
    
    # Drop rows with NaN values in the selected columns
    original_rows = data.shape[0]
    data.dropna(subset=[col_x, col_y], inplace=True)
    if data.shape[0] == 0:
        st.warning(f"No valid data points left after dropping rows with missing values in '{col_x}' and '{col_y}'. Cannot perform clustering.")
        return None, None, None, None
    if data.shape[0] < original_rows:
        st.info(f"Dropped {original_rows - data.shape[0]} rows due to missing values in selected columns: '{col_x}' and '{col_y}'.")

    if data.shape[0] < 2:
        st.error("Not enough data points remaining after preprocessing for DBSCAN clustering.")
        return None, None, None, None

    X = data[[col_x, col_y]].values
    eps = params.get('eps', 0.5)
    min_samples = params.get('min_samples', 5)

    # Run the from-scratch DBSCAN implementation
    clusters = dbscan_manual(X, eps=eps, min_samples=min_samples)
    data['Cluster'] = clusters
    
    # --- Visualization ---
    # Ensure 'Cluster' column is treated as a categorical variable for consistent coloring
    data_vis = data.copy()
    data_vis['Cluster'] = data_vis['Cluster'].astype(str)

    fig = px.scatter(
        data_vis, 
        x=col_x, 
        y=col_y, 
        color='Cluster',
        title=f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})',
        color_discrete_map={'-1': 'grey'}, # Assign a specific color to noise points
        category_orders={'Cluster': sorted(data_vis['Cluster'].unique(), key=int)} # Sort legend
    )
    
    fig.update_layout(showlegend=True, legend_title_text='Cluster', height=600)

    # --- Histogram ---
    fig_hist = create_histogram(data_vis)
    
    # --- Evaluation Metrics ---
    # Metrics like Silhouette are not well-defined if noise points (-1) are included.
    # We calculate metrics only on the clustered points.
    metrics = {}
    clustered_indices = data['Cluster'] != -1
    labels_clustered = data['Cluster'][clustered_indices]
    X_clustered = X[clustered_indices]
    
    # Metrics require at least 2 clusters (excluding noise)
    n_clusters_found = len(set(labels_clustered))
    if n_clusters_found > 1 and len(X_clustered) > 0:
        metrics['Silhouette Score (excl. noise)'] = silhouette_score(X_clustered, labels_clustered)
        metrics['Davies-Bouldin Index (excl. noise)'] = davies_bouldin_score(X_clustered, labels_clustered)
        metrics['Calinski-Harabasz Index (excl. noise)'] = calinski_harabasz_score(X_clustered, labels_clustered)
    else:
        st.warning("Not enough clusters found (excluding noise) to compute evaluation metrics.")

    return data, fig, fig_hist, metrics