import pandas.api.types as pdt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# --- Custom K-Medoids (PAM) Helper Functions ---

def initialize_medoids(X, k):
    """Initialiser k medoids aléatoires à partir des points de données"""
    np.random.seed(42)
    random_idx = np.random.permutation(X.shape[0])
    medoids = X[random_idx[:k]]
    return medoids

def assign_clusters_medoids(X, medoids):
    """Assigner chaque point au medoid le plus proche"""
    distances = np.linalg.norm(X[:, np.newaxis] - medoids, axis=2)
    return np.argmin(distances, axis=1)

def update_medoids(X, labels, k, old_medoids): # Added old_medoids to parameters for consistent handling
    """Mettre à jour les medoids en choisissant le point qui minimise la distance intra-cluster"""
    new_medoids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) == 0:
            # If a cluster is empty, re-initialize its medoid randomly from the dataset X
            new_medoids[i] = X[np.random.choice(X.shape[0])]
            continue

        # Calculer la somme des distances pour chaque point du cluster
        distances = np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2)
        total_distances = np.sum(distances, axis=1)
        
        # Le nouveau medoid est le point qui minimise cette somme
        new_medoid_index = np.argmin(total_distances)
        new_medoids[i] = cluster_points[new_medoid_index]
        
    return new_medoids

def compute_inertia_medoids(X, labels, medoids):
    """Calculer l'inertie (somme des distances aux medoids)"""
    inertia = 0
    for i in range(medoids.shape[0]):
        cluster_points = X[labels == i]
        if cluster_points.shape[0] > 0:
            inertia += np.sum(np.linalg.norm(cluster_points - medoids[i], axis=1))
    return inertia

def kmedoids_manual(X, k, max_iters=100, tol=1e-4):
    """Algorithme K-Medoids (PAM) complet"""
    medoids = initialize_medoids(X, k)
    
    for iteration in range(max_iters):
        old_medoids = medoids.copy()
        
        labels = assign_clusters_medoids(X, medoids)
        medoids = update_medoids(X, labels, k, old_medoids) # Pass old_medoids
        
        # Vérifier la convergence
        if np.sum(np.abs(medoids - old_medoids)) < tol:
            break
            
    inertia = compute_inertia_medoids(X, labels, medoids)
    return labels, medoids, inertia


def create_histogram(data):
    """
    Creates a histogram of the cluster distribution.
    """
    fig_hist = px.histogram(
        data,
        x='Cluster',
        title='Cluster Distribution',
        color='Cluster',
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_hist.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Number of data points",
        height=400
    )
    return fig_hist

def plot_elbow_curve(inertias, k_range):
    """
    Plots the Elbow curve.
    """
    fig_elbow = px.line(
        x=[str(k) for k in k_range],
        y=inertias,
        markers=True,
        title='Elbow Method to find optimal k for K-Medoids',
        labels={'x': 'Number of clusters (k)', 'y': 'Sum of distances to closest medoid'}
    )
    fig_elbow.update_layout(height=400)
    return fig_elbow

@st.cache_data
def run_kmedoids(data, params):
    """
    Runs the K-Medoids algorithm and returns the results.
    """
    col_x = params.get('col_x', data.columns[0])
    col_y = params.get('col_y', data.columns[1])

    # Ensure selected columns are numeric and handle NaNs
    if not pdt.is_numeric_dtype(data[col_x]) or not pdt.is_numeric_dtype(data[col_y]):
        st.error(f"Selected columns '{col_x}' and '{col_y}' must be numeric for K-Medoids clustering.")
        return None, None, None, None, None, None

    # Drop rows with NaN values in the selected columns
    original_rows = data.shape[0]
    data.dropna(subset=[col_x, col_y], inplace=True)
    if data.shape[0] == 0:
        st.warning(f"No valid data points left after dropping rows with missing values in '{col_x}' and '{col_y}'. Cannot perform clustering.")
        return None, None, None, None, None, None
    if data.shape[0] < original_rows:
        st.info(f"Dropped {original_rows - data.shape[0]} rows due to missing values in selected columns: '{col_x}' and '{col_y}'.")

    if data.shape[0] < 2:
        st.error("Not enough data points remaining after preprocessing for K-Medoids clustering.")
        return None, None, None, None, None, None

    X_np = data[[col_x, col_y]].values.astype(np.float64)
    n_clusters = params.get('n_clusters', 3)

    labels, medoids, inertia = kmedoids_manual(X_np, n_clusters)
    data['Cluster'] = labels

    # --- Visualization (2D Scatter Plot) ---
    fig_scatter_2d = px.scatter(
        data,
        x=col_x,
        y=col_y,
        color='Cluster',
        title=f'K-Medoids Clustering (2D View, k={n_clusters})',
        color_continuous_scale=px.colors.qualitative.Vivid
    )

    colors = px.colors.qualitative.Vivid
    unique_clusters = sorted(data['Cluster'].unique())
    color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}

    for i, medoid in enumerate(medoids):
        fig_scatter_2d.add_trace(go.Scatter(
            x=[medoid[0]],
            y=[medoid[1]],
            mode='markers',
            marker=dict(
                symbol='circle',
                color=color_map.get(i, 'black'),
                size=20,
                line=dict(width=3, color='black')
            ),
            name=f'Medoid {i}',
            showlegend=True
        ))

    fig_scatter_2d.update_layout(
        showlegend=True,
        legend_title_text='Cluster / Medoid',
        height=600
    )

    # --- Visualization (3D Scatter Plot - if applicable) ---
    fig_scatter_3d = None
    if data.shape[1] >= 3:
        # For 3D, we need a third feature. Let's just pick the next available numeric column.
        available_3d_cols = [c for c in data.columns if c not in [col_x, col_y] and data[c].dtype in [np.number]]
        if available_3d_cols:
            col_z = available_3d_cols[0]
            st.info(f"Using '{col_z}' as the Z-axis for 3D visualization.")
            data_3d_plot = data.dropna(subset=[col_x, col_y, col_z]).copy()
            
            if data_3d_plot.shape[0] > 0:
                X_np_3d = data_3d_plot[[col_x, col_y, col_z]].values.astype(np.float64)
                labels_3d, medoids_3d, _ = kmedoids_manual(X_np_3d, n_clusters)
                data_3d = data.copy() # Need to use the original `data` for consistency
                data_3d['Cluster'] = labels_3d
                
                fig_scatter_3d = px.scatter_3d(
                    data_3d,
                    x=col_x,
                    y=col_y,
                    z=col_z,
                    color='Cluster',
                    title=f'K-Medoids Clustering (3D View, k={n_clusters})',
                    color_continuous_scale=px.colors.qualitative.Vivid
                )
                for i, medoid_3d in enumerate(medoids_3d):
                    fig_scatter_3d.add_trace(go.Scatter3d(
                        x=[medoid_3d[0]],
                        y=[medoid_3d[1]],
                        z=[medoid_3d[2]],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            color=color_map.get(i, 'black'),
                            size=20,
                            line=dict(width=3, color='black')
                        ),
                        name=f'Medoid {i}',
                        showlegend=True
                    ))
                fig_scatter_3d.update_layout(height=600)
            else:
                st.warning(f"Not enough data for 3D visualization after dropping NaNs in '{col_x}', '{col_y}', '{col_z}'.")
        else:
            st.warning("Not enough numeric columns available for 3D visualization.")
    
    # --- Elbow Method ---
    k_range = range(2, min(11, data.shape[0]))  # From 2 to 10 clusters
    inertias = []
    for k_test in k_range:
        _, _, inertia_test = kmedoids_manual(X_np, k_test)
        inertias.append(inertia_test)
    
    fig_elbow = plot_elbow_curve(inertias, k_range)
    
    # --- Create Histogram ---
    fig_hist = create_histogram(data)
    
    # --- Evaluation Metrics ---
    metrics = {}
    if len(set(labels)) > 1:  # Need at least 2 clusters
        metrics['Silhouette Score'] = silhouette_score(X_np, labels)
        metrics['Davies-Bouldin Index'] = davies_bouldin_score(X_np, labels)
        metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(X_np, labels)
        metrics['Inertia (Sum of distances)'] = inertia
    else:
        st.warning("Only one cluster found. Cannot compute evaluation metrics.")
    
    return data, fig_scatter_2d, fig_scatter_3d, fig_elbow, fig_hist, metrics
