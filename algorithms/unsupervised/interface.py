import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from .kmeans import KMeans
from .kmedoids import KMedoids
from .dbscan import DBSCAN
from .agnes import AGNES
from .diana import DIANA


def render_unsupervised_interface():
    """
    Render the unsupervised learning interface
    """
    st.header("🔍 Phase 7: Unsupervised Learning")
    st.caption("Discover hidden patterns and clusters in your data")
    
    # Use normalized data if available, otherwise use cleaned data
    if st.session_state.get('normalized_df') is not None:
        working_df = st.session_state.normalized_df
        st.success("📊 Using normalized data for clustering")
    elif st.session_state.get('cleaned_data_for_ml') is not None:
        working_df = st.session_state.cleaned_data_for_ml
        st.warning("⚠️ Using cleaned data (not normalized). Normalization is recommended for better results.")
    else:
        st.error("❌ No data available. Please complete data cleaning first.")
        return
    
    # ==================
    # DISPLAY CURRENT DATA
    # ==================
    st.subheader("📋 Current Data")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", working_df.shape[0])
    with col2:
        st.metric("Columns", working_df.shape[1])
    with col3:
        st.metric("Numeric Columns", len(working_df.select_dtypes(include=[np.number]).columns))
    
    st.dataframe(working_df, use_container_width=True, height=300)
    
    st.divider()
    
    # Get numeric columns only
    numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("❌ Need at least 2 numeric columns for clustering analysis")
        return
    
    # ==================
    # ALGORITHM SELECTION
    # ==================
    st.subheader("🎯 Select Clustering Algorithm")
    
    algorithm = st.selectbox(
        "Choose an algorithm:",
        [
            "K-Means",
            "K-Medoids",
            "DBSCAN",
            "AGNES (Hierarchical)",
            "DIANA (Hierarchical)"
        ]
    )
    
    st.divider()
    
    # ==================
    # PARAMETER CONFIGURATION
    # ==================
    st.subheader("⚙️ Configure Parameters")
    
    if algorithm in ["K-Means", "K-Medoids"]:
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3)
        with col2:
            max_iter = st.slider("Maximum iterations", min_value=50, max_value=500, value=100)
        random_state = st.number_input("Random seed", min_value=0, value=42)
        
    elif algorithm == "DBSCAN":
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        with col2:
            min_pts = st.slider("MinPts", min_value=2, max_value=20, value=5)
            
    elif algorithm in ["AGNES (Hierarchical)", "DIANA (Hierarchical)"]:
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
        if algorithm == "AGNES (Hierarchical)":
            with col2:
                linkage = st.selectbox("Linkage method", ["single", "complete", "average"])
    
    st.divider()
    
    # ==================
    # FEATURE SELECTION
    # ==================
    st.subheader("📊 Select Features for Clustering")
    
    selected_features = st.multiselect(
        "Choose features (columns):",
        numeric_cols,
        default=numeric_cols[:min(2, len(numeric_cols))]
    )
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features for clustering")
        return
    
    st.divider()
    
    # ==================
    # RUN CLUSTERING
    # ==================
    st.subheader("🚀 Run Clustering")
    
    if st.button("▶️ Execute Algorithm", type="primary", use_container_width=True):
        # Prepare data
        X = working_df[selected_features].dropna().values
        
        with st.spinner(f"Running {algorithm}..."):
            try:
                # Run selected algorithm
                if algorithm == "K-Means":
                    model = KMeans(n_clusters=n_clusters, max_iterations=max_iter, random_state=random_state)
                    model.fit(X)
                    labels = model.labels
                    st.success(f"✅ K-Means completed! Inertia: {model.inertia_:.2f}")
                    
                elif algorithm == "K-Medoids":
                    model = KMedoids(n_clusters=n_clusters, max_iterations=max_iter, random_state=random_state)
                    model.fit(X)
                    labels = model.labels
                    st.success(f"✅ K-Medoids completed! Inertia: {model.inertia_:.2f}")
                    
                elif algorithm == "DBSCAN":
                    model = DBSCAN(eps=eps, min_pts=min_pts)
                    labels = model.fit_predict(X)
                    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    st.success(f"✅ DBSCAN completed! Found {n_clusters_found} clusters and {n_noise} noise points")
                    
                elif algorithm == "AGNES (Hierarchical)":
                    model = AGNES(n_clusters=n_clusters, linkage=linkage)
                    labels = model.fit_predict(X)
                    st.success(f"✅ AGNES completed with {linkage} linkage!")
                    
                elif algorithm == "DIANA (Hierarchical)":
                    model = DIANA(n_clusters=n_clusters)
                    labels = model.fit_predict(X)
                    st.success(f"✅ DIANA completed!")
                
                # Store results
                st.session_state.clustering_labels = labels
                st.session_state.clustering_model = model
                st.session_state.clustering_X = X
                st.session_state.clustering_features = selected_features
                
            except Exception as e:
                st.error(f"❌ Error running algorithm: {str(e)}")
    
    # ==================
    # VISUALIZATION
    # ==================
    if 'clustering_labels' in st.session_state:
        st.divider()
        st.subheader("📊 Clustering Results")
        
        labels = st.session_state.clustering_labels
        X = st.session_state.clustering_X
        features = st.session_state.clustering_features
        
        # 2D Visualization
        if len(features) >= 2:
            st.subheader("🎨 2D Cluster Visualization")
            
            fig = px.scatter(
                x=X[:, 0],
                y=X[:, 1],
                color=labels.astype(str),
                labels={'x': features[0], 'y': features[1], 'color': 'Cluster'},
                title=f"{algorithm} Clustering Results",
                height=600
            )
            
            fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='white')))
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D Visualization
        if len(features) >= 3:
            st.subheader("🎨 3D Cluster Visualization")
            
            fig = px.scatter_3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                color=labels.astype(str),
                labels={'x': features[0], 'y': features[1], 'z': features[2], 'color': 'Cluster'},
                title=f"{algorithm} 3D Clustering Results",
                height=700
            )
            
            fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='white')))
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Statistics
        st.subheader("📊 Cluster Statistics")
        unique_labels = np.unique(labels)
        stats_data = []
        
        for label in unique_labels:
            cluster_size = np.sum(labels == label)
            if label == -1:
                stats_data.append({
                    'Cluster': 'Noise',
                    'Size': cluster_size,
                    'Percentage': f"{(cluster_size/len(labels)*100):.2f}%"
                })
            else:
                stats_data.append({
                    'Cluster': f"Cluster {label}",
                    'Size': cluster_size,
                    'Percentage': f"{(cluster_size/len(labels)*100):.2f}%"
                })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
