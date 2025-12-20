import streamlit as st
import pandas as pd
import numpy as np
import os
import importlib.util
import plotly.express as px
import plotly.graph_objects as go


def render_supervised_interface():
    """
    Render the supervised learning interface for Phase 11
    """
    st.header("🎯 Phase 11: Supervised Learning")
    
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
    st.subheader("Current Data for Classification")
    st.dataframe(working_df, use_container_width=True)
    st.caption(f"Data Source: {data_source} | Normalization: {st.session_state.get('normalization_method', 'None')}")
    
    st.divider()
    
    # ==================
    # TARGET COLUMN SELECTION
    # ==================
    st.subheader("🎯 Target Column Selection")
    st.caption("Select the column you want to predict (class/label)")
    
    all_columns = working_df.columns.tolist()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_column = st.selectbox(
            "Select target column:",
            all_columns,
            key="target_column",
            help="This is the column your model will try to predict"
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("✅ Confirm Target", type="primary", use_container_width=True):
            st.session_state.supervised_target = target_column
            st.success(f"✅ Target set to: {target_column}")
    
    if 'supervised_target' not in st.session_state:
        st.info("👆 Please select and confirm your target column to continue")
        return
    
    target_col = st.session_state.supervised_target
    
    # Show target distribution
    st.write(f"#### Target Column: `{target_col}`")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        unique_classes = working_df[target_col].nunique()
        st.metric("Unique Classes", unique_classes)
    with col2:
        st.metric("Total Samples", len(working_df))
    with col3:
        missing_target = working_df[target_col].isna().sum()
        st.metric("Missing Values", missing_target)
    
    # Class distribution
    class_counts = working_df[target_col].value_counts()
    fig_dist = px.bar(
        x=class_counts.index.astype(str),
        y=class_counts.values,
        labels={'x': target_col, 'y': 'Count'},
        title=f"Class Distribution - {target_col}",
        height=300
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.divider()
    
    # ==================
    # TRAIN/TEST SPLIT
    # ==================
    st.subheader("✂️ Train/Test Split")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            key="test_size"
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("🔀 Split Data", type="primary", use_container_width=True):
            # Prepare features and target
            X = working_df.drop(columns=[target_col])
            y = working_df[target_col]
            
            # Simple train/test split
            n = len(working_df)
            test_n = int(n * test_size / 100)
            
            # Shuffle indices
            np.random.seed(42)
            indices = np.random.permutation(n)
            
            test_indices = indices[:test_n]
            train_indices = indices[test_n:]
            
            X_train = X.iloc[train_indices].reset_index(drop=True)
            X_test = X.iloc[test_indices].reset_index(drop=True)
            y_train = y.iloc[train_indices].reset_index(drop=True)
            y_test = y.iloc[test_indices].reset_index(drop=True)
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.success(f"✅ Data split: {len(train_indices)} train, {len(test_indices)} test")
    
    if 'X_train' not in st.session_state:
        st.info("👆 Click 'Split Data' to create train/test sets")
        return
    
    # Show split info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", len(st.session_state.X_train))
        st.caption(f"Classes: {st.session_state.y_train.value_counts().to_dict()}")
    with col2:
        st.metric("Test Samples", len(st.session_state.X_test))
        st.caption(f"Classes: {st.session_state.y_test.value_counts().to_dict()}")
    
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
                    spec = importlib.util.spec_from_file_location(f"algorithms.supervised.{module_name}", module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    alg_name = module_name.upper()
                    if alg_name == "KNN": alg_name = "k-NN"
                    if alg_name == "NAIVE_BAYES": alg_name = "Naive Bayes"
                    if alg_name == "C45": alg_name = "C4.5"
                    
                    algorithms[alg_name] = module
                except Exception as e:
                    st.error(f"Could not load algorithm from {filename}: {e}")
        return algorithms
    
    available_algorithms = get_algorithms()
    
    if len(available_algorithms) == 0:
        st.warning("⚠️ No supervised learning algorithms found. Please implement the algorithms first.")
        return
    
    # ==================
    # ALGORITHM CONFIGURATION & EXECUTION
    # ==================
    st.subheader("Algorithm Configuration & Execution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        sorted_algs = sorted(available_algorithms.keys())
        algorithm_options = ["Select all"] + sorted_algs
        algorithm_name = st.selectbox("Choose an algorithm", algorithm_options)
    
    with col2:
        # Only show parameters if a single algorithm is selected
        if algorithm_name != "Select all":
            params = {}
            
            if algorithm_name == "k-NN":
                params['k'] = st.slider("k (number of neighbors)", 1, 20, 5)
            elif algorithm_name == "Naive Bayes":
                st.info("ℹ️ Naive Bayes has no hyperparameters")
            elif algorithm_name == "C4.5":
                st.info("ℹ️ C4.5 decision tree implementation")
        else:
            st.info("🔄 All algorithms will run with default/optimal parameters")
            params = None
    
    with col2:
        if st.button("🚀 Train & Test", type="primary"):
            if algorithm_name == "Select all":
                # Run all algorithms and compare
                st.write("🔄 **Training all algorithms...**")
                
                all_results = {}
                all_metrics = {}
                
                with st.spinner("Processing..."):
                    for alg_name, module in available_algorithms.items():
                        try:
                            params_alg = {}
                            
                            # Set default parameters
                            if alg_name == "k-NN":
                                params_alg['k'] = 5
                            
                            run_function_name = f"run_{alg_name.lower().replace('-', '').replace('.', '').replace(' ', '_')}"
                            run_function = getattr(module, run_function_name, None)
                            
                            if run_function:
                                results = run_function(
                                    st.session_state.X_train,
                                    st.session_state.X_test,
                                    st.session_state.y_train,
                                    st.session_state.y_test,
                                    params_alg
                                )
                                all_results[alg_name] = results
                                
                                # Extract metrics
                                if len(results) >= 2:
                                    metrics = results[1] if len(results) > 1 else {}
                                    all_metrics[alg_name] = metrics
                        except Exception as e:
                            st.error(f"Error running {alg_name}: {str(e)}")
                
                # Store comparison results
                st.session_state.supervised_comparison_results = {
                    'all_results': all_results,
                    'all_metrics': all_metrics
                }
                st.session_state.supervised_algorithm_name = "Select all"
                st.rerun()
            else:
                # Run single algorithm
                st.write(f"Training **{algorithm_name}**...")
                module = available_algorithms[algorithm_name]
                run_function_name = f"run_{algorithm_name.lower().replace('-', '').replace('.', '').replace(' ', '_')}"
                run_function = getattr(module, run_function_name, None)
                
                if run_function:
                    results = run_function(
                        st.session_state.X_train,
                        st.session_state.X_test,
                        st.session_state.y_train,
                        st.session_state.y_test,
                        params
                    )
                    st.session_state.supervised_results = results
                    st.session_state.supervised_algorithm_name = algorithm_name
                    st.rerun()
    
    # ==================
    # DISPLAY RESULTS
    # ==================
    # Display comparison results if "Select all" was chosen
    if 'supervised_comparison_results' in st.session_state and st.session_state.get('supervised_algorithm_name') == "Select all":
        st.divider()
        st.header("📊 Algorithm Comparison Results")
        
        comp_data = st.session_state.supervised_comparison_results
        all_metrics = comp_data['all_metrics']
        
        # Create comparison table
        st.subheader("Evaluation Metrics Comparison Table")
        
        try:
            comparison_df = pd.DataFrame(all_metrics).T
            comparison_df.index.name = 'Algorithm'
            comparison_df = comparison_df.reset_index()
            
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
        
        scores = {}
        for alg_name, metrics in all_metrics.items():
            score = 0
            count = 0
            
            # Higher accuracy is better
            if 'Accuracy' in metrics:
                score += metrics['Accuracy']
                count += 1
            if 'Precision' in metrics:
                score += metrics['Precision']
                count += 1
            if 'Recall' in metrics:
                score += metrics['Recall']
                count += 1
            if 'F-Measure' in metrics:
                score += metrics['F-Measure']
                count += 1
            
            if count > 0:
                scores[alg_name] = score / count
        
        if scores:
            best_algorithm = max(scores.items(), key=lambda x: x[1])
            st.success(f"✨ **Recommended Algorithm: {best_algorithm[0]}**")
            st.write(f"**Composite Score:** `{best_algorithm[1]:.4f}`")
            
            score_df = pd.DataFrame(list(scores.items()), columns=['Algorithm', 'Composite Score'])
            score_df = score_df.sort_values('Composite Score', ascending=False)
            score_df['Composite Score'] = score_df['Composite Score'].apply(lambda x: f"{x:.4f}")
            st.dataframe(score_df, use_container_width=True, hide_index=True)
    
    elif 'supervised_results' in st.session_state and st.session_state.supervised_results:
        st.divider()
        st.subheader("Results")
        results = st.session_state.supervised_results
        algorithm_name = st.session_state.get('supervised_algorithm_name', '')
        
        # Results contain: predictions, metrics, confusion_matrix, figures
        if len(results) >= 2:
            predictions = results[0] if len(results) > 0 else None
            metrics = results[1] if len(results) > 1 else {}
            confusion_matrix = results[2] if len(results) > 2 else None
            figures = results[3] if len(results) > 3 else {}
            
            # Display metrics
            if metrics:
                st.write("#### Evaluation Metrics")
                cols = st.columns(len(metrics))
                for i, (metric_name, metric_value) in enumerate(metrics.items()):
                    if isinstance(metric_value, (int, float)):
                        cols[i].metric(metric_name, f"{metric_value:.4f}")
                    else:
                        cols[i].metric(metric_name, metric_value)
            
            # Display confusion matrix
            if confusion_matrix is not None:
                st.write("#### Confusion Matrix")
                st.dataframe(confusion_matrix, use_container_width=True)
            
            # Display figures
            if figures:
                for fig_name, fig in figures.items():
                    st.write(f"#### {fig_name}")
                    st.plotly_chart(fig, use_container_width=True, key=f"supervised_{algorithm_name}_{fig_name}")
