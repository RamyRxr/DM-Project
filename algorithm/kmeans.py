import pandas.api.types as pdt
import streamlit as st
import pandas as pd
import numpy as np # Import numpy
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# --- Custom K-Means Helper Functions from TP2_Clustering.ipynb ---
def initialize_centroids(X, k):
    """Initialiser k centroids aléatoires"""
    np.random.seed(42)
    random_idx = np.random.permutation(X.shape[0])
    centroids = X[random_idx[:k]]
    return centroids

def assign_clusters(X, centroids):
    """Assigner chaque point au centroid le plus proche"""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k, old_centroids):
    """Mettre à jour les centroids (moyenne des points de chaque cluster)"""
    new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else X[np.random.choice(X.shape[0])] for i in range(k)])
    return new_centroids

def compute_inertia(X, labels, centroids):
    """Calculer l'inertie (somme des distances au carré)"""
    inertia = 0
    for i in range(centroids.shape[0]):
        cluster_points = X[labels == i]
        if cluster_points.shape[0] > 0:
            inertia += np.sum(np.linalg.norm(cluster_points - centroids[i], axis=1)**2)
    return inertia

def kmeans_manual(X, k, max_iters=100, tol=1e-4):
    """Algorithme K-Means complet"""
    centroids = initialize_centroids(X, k)
    
    for iteration in range(max_iters):
        old_centroids = centroids.copy()
        labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, k, old_centroids)
        
        # Vérifier la convergence
        if np.linalg.norm(centroids - old_centroids) < tol:
            break
    
    inertia = compute_inertia(X, labels, centroids)
    return labels, centroids, inertia

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
        title='Elbow Method to find optimal k',
        labels={'x': 'Number of clusters (k)', 'y': 'Within-Cluster Sum of Squares (WCSS)'}
    )
    fig_elbow.update_layout(height=400)
    return fig_elbow

@st.cache_data
def run_kmeans(data, params):
    """
    Runs the K-Means algorithm and returns the results.
    """
    col_x = params.get('col_x', data.columns[0])
    col_y = params.get('col_y', data.columns[1])

    # Ensure selected columns are numeric and handle NaNs
    if not pdt.is_numeric_dtype(data[col_x]) or not pdt.is_numeric_dtype(data[col_y]):
        st.error(f"Selected columns '{col_x}' and '{col_y}' must be numeric for K-Means clustering.")
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
        st.error("Not enough data points remaining after preprocessing for K-Means clustering.")
        return None, None, None, None, None, None

    X_np = data[[col_x, col_y]].values.astype(np.float64)
    n_clusters = params.get('n_clusters', 3)

    # --- Run custom K-Means ---
    labels, centroids, inertia = kmeans_manual(X_np, n_clusters)
    data['Cluster'] = labels
    
    # --- Visualization (2D Scatter Plot) ---
    fig_scatter_2d = px.scatter(
        data, 
        x=col_x, 
        y=col_y, 
        color='Cluster',
        title=f'K-Means Clustering (2D View, k={n_clusters})',
        color_continuous_scale=px.colors.qualitative.Vivid
    )

    # Add centroids to the 2D plot with matching colors
    colors = px.colors.qualitative.Vivid
    unique_clusters = sorted(data['Cluster'].unique())
    color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}

    for i, centroid in enumerate(centroids):
        fig_scatter_2d.add_trace(go.Scatter(
            x=[centroid[0]],
            y=[centroid[1]],
            mode='markers',
            marker=dict(
                symbol='circle',
                color=color_map.get(i, 'black'),
                size=20,
                line=dict(width=3, color='black')
            ),
            name=f'Centroid {i}',
            showlegend=True
        ))

    fig_scatter_2d.update_layout(
        showlegend=True,
        legend_title_text='Cluster / Centroid',
        height=600
    )
    
    # --- Visualization (3D Scatter Plot - if applicable) ---
    fig_scatter_3d = None
    if data.shape[1] >= 3: # Check if there are at least 3 columns available in the dataframe
        # For 3D, we need a third feature. Let's just pick the next available numeric column.
        available_3d_cols = [c for c in data.columns if c not in [col_x, col_y] and data[c].dtype in [np.number]]
        if available_3d_cols:
            col_z = available_3d_cols[0]
            st.info(f"Using '{col_z}' as the Z-axis for 3D visualization.")
            data_3d_plot = data.dropna(subset=[col_x, col_y, col_z]).copy()
            
            if data_3d_plot.shape[0] > 0:
                X_np_3d = data_3d_plot[[col_x, col_y, col_z]].values.astype(np.float64)
                labels_3d, centroids_3d, _ = kmeans_manual(X_np_3d, n_clusters)
                data_3d_plot['Cluster'] = labels_3d
                
                fig_scatter_3d = px.scatter_3d(
                    data_3d_plot,
                    x=col_x,
                    y=col_y,
                    z=col_z,
                    color='Cluster',
                    title=f'K-Means Clustering (3D View, k={n_clusters})',
                    color_continuous_scale=px.colors.qualitative.Vivid
                )
                for i, centroid_3d in enumerate(centroids_3d):
                    fig_scatter_3d.add_trace(go.Scatter3d(
                        x=[centroid_3d[0]],
                        y=[centroid_3d[1]],
                        z=[centroid_3d[2]],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            color=color_map.get(i, 'black'),
                            size=20,
                            line=dict(width=3, color='black')
                        ),
                        name=f'Centroid {i}',
                        showlegend=True
                    ))
                fig_scatter_3d.update_layout(height=600)
            else:
                st.warning(f"Not enough data for 3D visualization after dropping NaNs in '{col_x}', '{col_y}', '{col_z}'.")
        else:
            st.warning("Not enough numeric columns available for 3D visualization.")
