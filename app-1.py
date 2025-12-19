import streamlit as st
import pandas as pd
from sklearn import datasets
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

st.set_page_config(layout="wide")

st.title("Data Mining Project")

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Unsupervised Learning", "Supervised Learning"])

if selection == "Unsupervised Learning":
    st.header("Unsupervised Learning")

    # --- Function to get available algorithms ---
    @st.cache_resource
    def get_algorithms():
        algorithms = {}
        alg_path = 'algorithms'
        if os.path.exists(alg_path) and os.path.isdir(alg_path):
            for filename in os.listdir(alg_path):
                if filename.endswith('.py') and not filename.startswith('__'):
                    module_name = filename[:-3]
                    try:
                        spec = importlib.util.spec_from_file_location(f"algorithms.{module_name}", os.path.join(alg_path, filename))
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        alg_name = module_name.upper()
                        if alg_name == "KMEANS": alg_name = "K-Means"
                        if alg_name == "KMEDOIDS": alg_name = "K-Medoids"
                        algorithms[alg_name] = module
                    except Exception as e:
                        st.error(f"Could not load algorithm from {filename}: {e}")
        return algorithms

    import os
    import importlib.util
    available_algorithms = get_algorithms()
    
    # --- State Management Initialization ---
    for key in ['original_data', 'cleaned_data', 'processed_data', 'last_uploaded_file', 'results']:
        if key not in st.session_state:
            st.session_state[key] = None

    # --- Step 1: Data Loading ---
    st.subheader("1. Data Loading")
    data_source = st.selectbox("Choose a data source", ["Upload a file", "Use a sample dataset"])

    new_data_loaded = False
    if data_source == "Upload a file":
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

        # If user clears the file uploader, reset everything
        if uploaded_file is None and st.session_state.last_uploaded_file is not None:
            for key in ['original_data', 'cleaned_data', 'processed_data', 'last_uploaded_file', 'results']:
                st.session_state[key] = None
            st.rerun()

        if uploaded_file is not None and uploaded_file != st.session_state.last_uploaded_file:
            st.session_state.last_uploaded_file = uploaded_file
            new_data_loaded = True
            
            missing_markers = st.text_input("Specify characters to treat as missing values (comma-separated)", value="")
            na_values = [marker.strip() for marker in missing_markers.split(',') if marker] if missing_markers else None
            
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file, na_values=na_values)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                    if na_values:
                        data.replace(na_values, np.nan, inplace=True)
                st.session_state.original_data = data
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.original_data = None
    
    else: # Sample dataset
        sample_dataset = st.selectbox("Choose a sample dataset", ["Iris", "Wine"])
        if st.button(f"Load {sample_dataset} dataset"):
            new_data_loaded = True
            if sample_dataset == "Iris":
                data = pd.DataFrame(data=datasets.load_iris().data, columns=datasets.load_iris().feature_names)
            elif sample_dataset == "Wine":
                data = pd.DataFrame(data=datasets.load_wine().data, columns=datasets.load_wine().feature_names)
            st.session_state.original_data = data
            st.session_state.last_uploaded_file = None

    if new_data_loaded and st.session_state.original_data is not None:
        # Automatic detection and replacement of non-numeric values
        temp_data = st.session_state.original_data.copy()
        cols_changed = False
        for col in temp_data.select_dtypes(include=['object']).columns:
            numeric_col = pd.to_numeric(temp_data[col], errors='coerce')
            replaced_mask = numeric_col.isna() & temp_data[col].notna()
            if replaced_mask.any():
                cols_changed = True
                replaced_values = temp_data[col][replaced_mask].unique()
                st.info(f"Automatically converted column **'{col}'**: replaced non-numeric values with NaN: `{list(replaced_values)}`")
            temp_data[col] = numeric_col
        
        if not cols_changed and temp_data.select_dtypes(include=['object']).empty: 
            st.success("All columns are already numeric or successfully converted.")
        elif not cols_changed and not temp_data.select_dtypes(include=['object']).empty:
            st.info("No automatic conversion applied as no object columns contained convertible numeric values.")

        st.session_state.cleaned_data = temp_data.copy()
        st.session_state.processed_data = temp_data.copy()
        st.session_state.results = None # Clear previous results
        st.rerun()

    st.divider()

    # --- Step 2: Preprocessing ---
    if st.session_state.original_data is not None:
        st.subheader("2. Preprocessing")

        with st.expander("Show Preprocessing Options"):
            missing_values_option = st.selectbox("Missing Values:", ("None", "Remove rows", "Fill with mean", "Fill with median"))
            scaling_option = st.selectbox("Feature Scaling:", ("None", "Standardization", "Normalization"))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Preprocessing", type="primary"):
                temp_data = st.session_state.cleaned_data.copy()

                if missing_values_option == "Remove rows": temp_data.dropna(inplace=True)
                elif missing_values_option == "Fill with mean": temp_data.fillna(temp_data.mean(numeric_only=True), inplace=True)
                elif missing_values_option == "Fill with median": temp_data.fillna(temp_data.median(numeric_only=True), inplace=True)
                
                if scaling_option != "None":
                    numeric_cols = temp_data.select_dtypes(include=np.number).columns.tolist()
                    if numeric_cols:
                        scaler = StandardScaler() if scaling_option == "Standardization" else MinMaxScaler()
                        temp_data[numeric_cols] = scaler.fit_transform(temp_data[numeric_cols])
                
                st.session_state.processed_data = temp_data
                st.success("Preprocessing applied!")
                st.session_state.results = None # Clear old results

        with col2:
            if st.button("Reset Preprocessing"):
                st.session_state.processed_data = st.session_state.cleaned_data.copy()
                st.session_state.results = None # Clear old results
                st.info("Preprocessing options reset.")

        st.write("Current Data for Clustering:")
        st.dataframe(st.session_state.processed_data)

    st.divider()

    # --- Step 3: Algorithm Configuration & Execution ---
    if st.session_state.processed_data is not None:
        st.subheader("3. Algorithm Configuration & Execution")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            sorted_algs = sorted(available_algorithms.keys(), key=lambda x: (x != 'K-Means', x))
            algorithm_name = st.selectbox("Choose an algorithm", sorted_algs)
            
            # Add feature selection dropdowns
            st.write("#### Select Features for 2D Visualization")
            df_cols = st.session_state.processed_data.columns.tolist()
            # Filter out the 'Cluster' column if it exists
            if 'Cluster' in df_cols:
                df_cols.remove('Cluster')
            
            col_x = st.selectbox("X-axis feature:", df_cols, index=0)
            # Ensure y-axis defaults to a different column if possible
            y_index = 1 if len(df_cols) > 1 else 0
            col_y = st.selectbox("Y-axis feature:", df_cols, index=y_index)

        with col2:
            params = {'col_x': col_x, 'col_y': col_y}
            if algorithm_name == "K-Means": params['n_clusters'] = st.slider("k", 2, 15, 3)
            elif algorithm_name == "DBSCAN":
                params['eps'] = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
                params['min_samples'] = st.slider("Min samples", 1, 20, 5)
            elif algorithm_name == "AGNES":
                params['n_clusters'] = st.slider("k", 2, 15, 3)
                params['linkage'] = st.selectbox("Linkage", ("ward", "complete", "average", "single"))
            elif algorithm_name in ["K-Medoids", "DIANA"]:
                params['n_clusters'] = st.slider("k", 2, 15, 3)

        with col2:
            if st.button("Run Clustering", type="primary"):
                st.write(f"Running **{algorithm_name}**...")
                # Add selected columns to params
                params['col_x'] = col_x
                params['col_y'] = col_y
                run_data = st.session_state.processed_data.copy()
                module = available_algorithms[algorithm_name]
                run_function_name = f"run_{algorithm_name.lower().replace('-', '')}"
                run_function = getattr(module, run_function_name, None)

                if run_function:
                    results = run_function(run_data, params)
                    st.session_state.results = results # Save results to state

        # --- Step 4: Display Results ---
        if 'results' in st.session_state and st.session_state.results:
            st.subheader("4. Results")
            results = st.session_state.results
            
            if algorithm_name in ["AGNES", "DIANA"]:
                unpacked = results
                if len(unpacked) == 5:
                    clustered_data, fig_scatter, fig_dendrogram, fig_hist, metrics = unpacked
                    st.write("#### Dendrogram")
                    if fig_dendrogram: st.plotly_chart(fig_dendrogram, use_container_width=True)
                    st.write(f"#### {algorithm_name} Clustering Results")
                    if fig_scatter: st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("Algorithm did not return the expected results for display.")
            elif algorithm_name in ["K-Means", "K-Medoids"]:
                unpacked = results
                if len(unpacked) == 6:
                    clustered_data, fig_scatter_2d, fig_scatter_3d, fig_elbow, fig_hist, metrics = unpacked
                    st.write("#### Clustering Results (2D & 3D)")
                    if fig_scatter_2d: st.plotly_chart(fig_scatter_2d, use_container_width=True)
                    if fig_scatter_3d: st.plotly_chart(fig_scatter_3d, use_container_width=True)
                    st.write("#### Elbow Method for Optimal k")
                    if fig_elbow: st.plotly_chart(fig_elbow, use_container_width=True)
                else:
                    st.warning("Algorithm did not return the expected results for display.")
            else: # DBSCAN and others
                unpacked = results
                if len(unpacked) == 4:
                    clustered_data, fig_scatter, fig_hist, metrics = unpacked
                    st.write(f"#### {algorithm_name} Results and Visualization")
                    if fig_scatter: st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("Algorithm did not return the expected results for display.")

            if 'clustered_data' in locals() and clustered_data is not None:
                st.write("#### Data with Cluster Assignments")
                st.dataframe(clustered_data)
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
        
        # --- Step 5: Compare All Algorithms ---
        st.subheader("5. Compare All Algorithms")
        # (Comparison logic can be added here later if needed)

elif selection == "Supervised Learning":
    st.header("Supervised Learning")
    st.write("Welcome to the Supervised Learning section.")
    # Placeholder for supervised learning components


