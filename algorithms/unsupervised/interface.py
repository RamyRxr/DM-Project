import streamlit as st
import pandas as pd
import numpy as np
import os
import importlib.util
import plotly.express as px


def render_unsupervised_interface():
    """
    Render the unsupervised learning interface for Phase 9
    """
    st.header("🔍 Phase 9: Unsupervised Learning")
    
    # Check if cleaned data exists and get working dataframe
    if st.session_state.get('cleaned_data_for_ml') is not None:
        working_df = st.session_state.cleaned_data_for_ml.copy()
        data_source = "cleaned_data_for_ml"
    elif st.session_state.get('normalized_df') is not None:
        working_df = st.session_state.normalized_df.copy()
        data_source = "normalized_df"
    else:
        st.error("❌ No cleaned data found. Please complete data cleaning first (including normalization).")
        st.info("👈 Go back to Data Cleaning section and complete all phases.")
        return
    
    # ==================
    # DISPLAY CURRENT DATA
    # ==================
    st.subheader("Current Data for Clustering")
    st.dataframe(working_df, use_container_width=True)
    st.caption(f"Data Source: {data_source} | Normalization: {st.session_state.get('normalization_method', 'None')}")
    
    st.divider()
    
    # ==================
    # LOAD AVAILABLE ALGORITHMS
    # ==================
    @st.cache_resource
    def get_algorithms():
        algorithms = {}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        for filename in os.listdir(current_dir):
            if filename.endswith('.py') and not filename.startswith('__') and filename != 'interface.py':
                module_name = filename[:-3]
                try:
                    module_path = os.path.join(current_dir, filename)
                    spec = importlib.util.spec_from_file_location(f"algorithms.unsupervised.{module_name}", module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    alg_name = module_name.upper()
                    if alg_name == "KMEANS": alg_name = "K-Means"
                    if alg_name == "KMEDOIDS": alg_name = "K-Medoids"
                    
                    algorithms[alg_name] = module
                except Exception as e:
                    st.error(f"Could not load algorithm from {filename}: {e}")
        return algorithms
    
    available_algorithms = get_algorithms()
    
    # ==================
    # ALGORITHM CONFIGURATION & EXECUTION
    # ==================
    st.subheader("Algorithm Configuration & Execution")
    
    # Feature Selection FIRST
    st.write("#### Select Features for 2D Visualization")
    col_feat1, col_feat2, col_feat3 = st.columns([2, 2, 1])
    
    df_cols = working_df.columns.tolist()
    if 'Cluster' in df_cols:
        df_cols.remove('Cluster')
    
    with col_feat1:
        col_x = st.selectbox("X-axis feature:", df_cols, index=0, key="axis_x")
    with col_feat2:
        y_index = 1 if len(df_cols) > 1 else 0
        col_y = st.selectbox("Y-axis feature:", df_cols, index=y_index, key="axis_y")
    with col_feat3:
        st.write("")  # Spacing
        apply_features = st.button("Apply", type="primary", use_container_width=True)
    
    if apply_features or 'selected_features' not in st.session_state:
        st.session_state.selected_features = {'col_x': col_x, 'col_y': col_y}
        if apply_features:
            st.success(f"✅ Features applied: {col_x} vs {col_y}")
    
    # Display 2D scatter plot of selected features
    if 'selected_features' in st.session_state:
        selected_col_x = st.session_state.selected_features['col_x']
        selected_col_y = st.session_state.selected_features['col_y']
        
        fig = px.scatter(
            working_df,
            x=selected_col_x,
            y=selected_col_y,
            title=f"2D Visualization: {selected_col_x} vs {selected_col_y}",
            labels={selected_col_x: selected_col_x, selected_col_y: selected_col_y},
            height=400
        )
        fig.update_traces(marker=dict(size=8, opacity=0.6))
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        sorted_algs = sorted(available_algorithms.keys(), key=lambda x: (x != 'K-Means', x))
        algorithm_name = st.selectbox("Choose an algorithm", sorted_algs)

    with col2:
        params = st.session_state.selected_features.copy()
        
        if algorithm_name == "K-Means":
            params['n_clusters'] = st.slider("k", 2, 15, 3)
        elif algorithm_name == "K-Medoids":
            params['n_clusters'] = st.slider("k", 2, 15, 3)
        elif algorithm_name == "DBSCAN":
            params['eps'] = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
            params['min_samples'] = st.slider("Min samples", 1, 20, 5)
        elif algorithm_name == "AGNES":
            params['n_clusters'] = st.slider("k", 2, 15, 3)
            params['linkage'] = st.selectbox("Linkage", ("ward", "complete", "average", "single"))
        elif algorithm_name == "DIANA":
            params['n_clusters'] = st.slider("k", 2, 15, 3)

    with col2:
        if st.button("Run Clustering", type="primary"):
            st.write(f"Running **{algorithm_name}**...")
            run_data = working_df.copy()
            module = available_algorithms[algorithm_name]
            run_function_name = f"run_{algorithm_name.lower().replace('-', '')}"
            run_function = getattr(module, run_function_name, None)

            if run_function:
                results = run_function(run_data, params)
                st.session_state.results = results
                st.session_state.algorithm_name = algorithm_name
                st.rerun()
    
    # ==================
    # DISPLAY RESULTS
    # ==================
    if 'results' in st.session_state and st.session_state.results:
        st.divider()
        st.subheader("Results")
        results = st.session_state.results
        algorithm_name = st.session_state.get('algorithm_name', '')
        
        if algorithm_name in ["AGNES", "DIANA"]:
            unpacked = results
            if len(unpacked) == 5:
                clustered_data, fig_scatter, fig_dendrogram, fig_hist, metrics = unpacked
                st.write("#### Dendrogram")
                if fig_dendrogram:
                    st.plotly_chart(fig_dendrogram, use_container_width=True)
                st.write(f"#### {algorithm_name} Clustering Results")
                if fig_scatter:
                    st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("Algorithm did not return the expected results for display.")
                
        elif algorithm_name in ["K-Means", "K-Medoids"]:
            unpacked = results
            if len(unpacked) == 6:
                clustered_data, fig_scatter_2d, fig_scatter_3d, fig_elbow, fig_hist, metrics = unpacked
                st.write("#### Clustering Results")
                if fig_scatter_2d:
                    st.plotly_chart(fig_scatter_2d, use_container_width=True)
                st.write("#### Elbow Method for Optimal k")
                if fig_elbow:
                    st.plotly_chart(fig_elbow, use_container_width=True)
            else:
                st.warning("Algorithm did not return the expected results for display.")
                
        else:  # DBSCAN
            unpacked = results
            if len(unpacked) == 4:
                clustered_data, fig_scatter, fig_hist, metrics = unpacked
                st.write(f"#### {algorithm_name} Results and Visualization")
                if fig_scatter:
                    st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("Algorithm did not return the expected results for display.")

        if 'clustered_data' in locals() and clustered_data is not None:
            st.write("#### Data with Cluster Assignments")
            st.dataframe(clustered_data, use_container_width=True)
            
        if 'fig_hist' in locals() and fig_hist:
            st.write("#### Cluster Distribution Histogram")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        if 'metrics' in locals() and metrics:
            st.write("#### Evaluation Metrics")
            cols = st.columns(len(metrics))
            for i, (metric_name, metric_value) in enumerate(metrics.items()):
                cols[i].metric(metric_name, f"{metric_value:.3f}")
        elif 'metrics' in locals():
            st.warning("Evaluation metrics could not be computed.")



