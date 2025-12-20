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
        algorithm_options = ["Select all"] + sorted_algs
        algorithm_name = st.selectbox("Choose an algorithm", algorithm_options)

    with col2:
        # Only show parameters if a single algorithm is selected
        if algorithm_name != "Select all":
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
        else:
            st.info("🔄 All algorithms will run with their optimal parameters")
            params = None

    with col2:
        if st.button("Run Clustering", type="primary"):
            if algorithm_name == "Select all":
                # Run all algorithms and compare
                st.write("🔄 **Running all algorithms for comparison...**")
                
                all_results = {}
                all_histograms = {}
                all_metrics = {}
                
                with st.spinner("Processing..."):
                    for alg_name, module in available_algorithms.items():
                        try:
                            run_data = working_df.copy()
                            params_alg = st.session_state.selected_features.copy()
                            
                            # Set optimal parameters for each algorithm
                            if alg_name in ["K-Means", "K-Medoids"]:
                                # Use optimal k = 3 as default (will be detected from results)
                                params_alg['n_clusters'] = 3
                            elif alg_name == "DBSCAN":
                                params_alg['eps'] = 0.5
                                params_alg['min_samples'] = 5
                            elif alg_name in ["AGNES", "DIANA"]:
                                params_alg['n_clusters'] = 3
                                if alg_name == "AGNES":
                                    params_alg['linkage'] = 'ward'
                            
                            run_function_name = f"run_{alg_name.lower().replace('-', '')}"
                            run_function = getattr(module, run_function_name, None)
                            
                            if run_function:
                                results = run_function(run_data, params_alg)
                                all_results[alg_name] = results
                                
                                # Extract histogram and metrics
                                if alg_name in ["K-Means", "K-Medoids"]:
                                    if len(results) == 6:
                                        _, _, _, _, fig_hist, metrics = results
                                        all_histograms[alg_name] = fig_hist
                                        all_metrics[alg_name] = metrics
                                elif alg_name in ["AGNES", "DIANA"]:
                                    if len(results) == 5:
                                        _, _, _, fig_hist, metrics = results
                                        all_histograms[alg_name] = fig_hist
                                        all_metrics[alg_name] = metrics
                                elif alg_name == "DBSCAN":
                                    if len(results) == 4:
                                        _, _, fig_hist, metrics = results
                                        all_histograms[alg_name] = fig_hist
                                        all_metrics[alg_name] = metrics
                        except Exception as e:
                            st.error(f"Error running {alg_name}: {str(e)}")
                
                # Store comparison results
                st.session_state.comparison_results = {
                    'all_results': all_results,
                    'all_histograms': all_histograms,
                    'all_metrics': all_metrics
                }
                st.session_state.algorithm_name = "Select all"
                st.rerun()
            else:
                # Run single algorithm
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
    # Display comparison results if "Select all" was chosen
    if 'comparison_results' in st.session_state and st.session_state.get('algorithm_name') == "Select all":
        st.divider()
        st.header("📊 Algorithm Comparison Results")
        
        comp_data = st.session_state.comparison_results
        all_histograms = comp_data['all_histograms']
        all_metrics = comp_data['all_metrics']
        
        # Display histograms in a grid
        st.subheader("Cluster Distribution Histograms")
        cols_hist = st.columns(3)
        for idx, (alg_name, fig_hist) in enumerate(all_histograms.items()):
            with cols_hist[idx % 3]:
                st.write(f"**{alg_name}**")
                if fig_hist:
                    st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{alg_name}_{idx}")
        
        # Create comparison table
        st.subheader("Evaluation Metrics Comparison Table")
        
        # Build comparison dataframe
        try:
            comparison_df = pd.DataFrame(all_metrics).T
            comparison_df.index.name = 'Algorithm'
            comparison_df = comparison_df.reset_index()
            
            # Display the raw dataframe first
            st.dataframe(comparison_df.style.format({
                col: "{:.4f}" for col in comparison_df.columns 
                if col != 'Algorithm' and pd.api.types.is_numeric_dtype(comparison_df[col])
            }), use_container_width=True)
        except Exception as e:
            st.error(f"Error creating comparison table: {str(e)}")
            st.write("Raw metrics data:")
            st.json(all_metrics)
        
        # Determine best algorithm
        st.subheader("🏆 Best Algorithm Recommendation")
        
        # Calculate scores based on metrics
        # Higher Silhouette Score is better
        # Lower Davies-Bouldin Index is better
        # Higher Calinski-Harabasz Index is better
        
        scores = {}
        for alg_name, metrics in all_metrics.items():
            # Skip if metrics is None or empty
            if not metrics or metrics is None:
                continue
                
            score = 0
            count = 0
            
            if 'Silhouette Score' in metrics:
                score += metrics['Silhouette Score']
                count += 1
            if 'Silhouette Score (excl. noise)' in metrics:
                score += metrics['Silhouette Score (excl. noise)']
                count += 1
            if 'Davies-Bouldin Index' in metrics:
                # Invert since lower is better
                score -= metrics['Davies-Bouldin Index'] * 0.1
                count += 1
            if 'Davies-Bouldin Index (excl. noise)' in metrics:
                score -= metrics['Davies-Bouldin Index (excl. noise)'] * 0.1
                count += 1
            if 'Calinski-Harabasz Index' in metrics:
                # Normalize by dividing by 100
                score += metrics['Calinski-Harabasz Index'] / 100
                count += 1
            if 'Calinski-Harabasz Index (excl. noise)' in metrics:
                score += metrics['Calinski-Harabasz Index (excl. noise)'] / 100
                count += 1
            
            if count > 0:
                scores[alg_name] = score / count
        
        if scores:
            best_algorithm = max(scores.items(), key=lambda x: x[1])
            st.success(f"✨ **Recommended Algorithm: {best_algorithm[0]}**")
            st.write(f"**Reasoning:**")
            st.write(f"- Composite Score: `{best_algorithm[1]:.4f}`")
            
            # Display all scores
            st.write("**All Algorithm Scores:**")
            score_df = pd.DataFrame(list(scores.items()), columns=['Algorithm', 'Composite Score'])
            score_df = score_df.sort_values('Composite Score', ascending=False)
            score_df['Composite Score'] = score_df['Composite Score'].apply(lambda x: f"{x:.4f}")
            st.dataframe(score_df, use_container_width=True, hide_index=True)
            
            # Explain metrics
            with st.expander("ℹ️ Understanding the Metrics"):
                st.markdown("""
                **Silhouette Score** (0 to 1, higher is better):
                - Measures how similar points are to their own cluster vs. other clusters
                - Values near 1 indicate well-separated clusters
                
                **Davies-Bouldin Index** (0 to ∞, lower is better):
                - Measures average similarity between clusters
                - Lower values indicate better separation
                
                **Calinski-Harabasz Index** (0 to ∞, higher is better):
                - Ratio of between-cluster to within-cluster dispersion
                - Higher values indicate denser, well-separated clusters
                
                **Composite Score**:
                - Weighted combination of all metrics
                - Accounts for different metric scales and directions
                """)
    
    elif 'results' in st.session_state and st.session_state.results:
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
                
                # Display optimal k if available
                if metrics and 'Optimal K (Elbow Method)' in metrics:
                    optimal_k = metrics['Optimal K (Elbow Method)']
                    st.success(f"📊 **Recommended Optimal K: {optimal_k}** (based on Elbow Method)")
                
                st.write("#### Elbow Method for Optimal k")
                if fig_elbow:
                    st.plotly_chart(fig_elbow, use_container_width=True)
                
                st.write("#### 2D Clustering Visualization")
                if fig_scatter_2d:
                    st.plotly_chart(fig_scatter_2d, use_container_width=True)
                
                if fig_scatter_3d:
                    st.write("#### 3D Clustering Visualization")
                    st.plotly_chart(fig_scatter_3d, use_container_width=True)
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



