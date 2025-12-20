import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append('algorithms')
from algorithms.unsupervised.interface import render_unsupervised_interface
from algorithms.supervised.interface import render_supervised_interface

# ===========================
# CONFIGURATION
# ===========================
st.set_page_config(page_title="Data Mining Project", page_icon="📊", layout="wide")

# ===========================
# HELPER FUNCTIONS
# ===========================

def detect_problematic_values(df):
    """
    Detect values that don't fit the expected data type of their column.
    Returns a dictionary of {value: count} for problematic values found.
    """
    problems = {}
    
    for col in df.columns:
        col_data = df[col].dropna()  # Remove existing NaN values
        
        if len(col_data) == 0:
            continue
            
        # Try to convert to numeric
        numeric_test = pd.to_numeric(col_data, errors='coerce')
        numeric_success_rate = numeric_test.notna().sum() / len(col_data)
        
        # If more than 50% can be numeric, this should be a numeric column
        if numeric_success_rate > 0.5:
            # Find the values that failed numeric conversion
            failed_mask = numeric_test.isna()
            failed_values = col_data[failed_mask]
            
            for val in failed_values.unique():
                val_str = str(val)
                problems[val_str] = problems.get(val_str, 0) + (failed_values == val).sum()
    
    return problems


def convert_to_numeric_with_nan(df):
    """
    Convert all columns to numeric where possible.
    Non-numeric values become NaN automatically.
    """
    df_converted = df.copy()
    
    for col in df_converted.columns:
        # Try to convert each column to numeric
        # errors='coerce' will turn non-numeric values into NaN
        numeric_col = pd.to_numeric(df_converted[col], errors='coerce')
        
        # If ANY values can be converted to numeric (not all NaN), use it
        if numeric_col.notna().any():
            df_converted[col] = numeric_col
    
    return df_converted


def replace_values_with_nan(df, values_to_replace):
    """
    Replace specified values with NaN across the entire dataframe.
    """
    df_clean = df.copy()
    
    for col in df_clean.columns:
        # For each value to replace, create a mask and replace
        for val in values_to_replace:
            mask = df_clean[col].astype(str).str.strip().str.lower() == str(val).strip().lower()
            df_clean.loc[mask, col] = np.nan
    
    return df_clean

def detect_outliers_iqr(series):
    """
    Detect outliers using the IQR (Interquartile Range) method.
    Returns a boolean mask where True indicates an outlier.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (series < lower_bound) | (series > upper_bound)

# ===========================
# MAIN APPLICATION
# ===========================

st.title("📊 Data Mining Project Application")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'custom_indicators' not in st.session_state:
    st.session_state.custom_indicators = []
if 'data_cleaned' not in st.session_state:
    st.session_state.data_cleaned = False
if 'uploaded_file_id' not in st.session_state:
    st.session_state.uploaded_file_id = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'section' not in st.session_state:
    st.session_state.section = "📥 Data Cleaning"
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False
if 'normalization_complete' not in st.session_state:
    st.session_state.normalization_complete = False
if 'cleaned_data_for_ml' not in st.session_state:
    st.session_state.cleaned_data_for_ml = None

# ===========================
# SIDEBAR - NAVIGATION
# ===========================
st.sidebar.header("📊 Navigation")

# Section selection
sections = ["📥 Data Cleaning", "🔍 Unsupervised Learning", "🎯 Supervised Learning"]

# Check if data cleaning is complete
data_ready = st.session_state.get('normalization_complete', False)

# Show all sections with button interface
st.sidebar.write("**Select Section:**")

# Data Cleaning button (always enabled)
if st.sidebar.button("📥 Data Cleaning", use_container_width=True, 
                     type="primary" if st.session_state.section == "📥 Data Cleaning" else "secondary",
                     key="btn_data_cleaning"):
    st.session_state.section = "📥 Data Cleaning"
    st.rerun()

# Unsupervised Learning button (enabled after normalization)
if st.sidebar.button("🔍 Unsupervised Learning", use_container_width=True, 
                     type="primary" if st.session_state.section == "🔍 Unsupervised Learning" else "secondary",
                     disabled=not data_ready,
                     key="btn_unsupervised"):
    st.session_state.section = "🔍 Unsupervised Learning"
    st.rerun()

# Supervised Learning button (enabled after normalization)
if st.sidebar.button("🎯 Supervised Learning", use_container_width=True, 
                     type="primary" if st.session_state.section == "🎯 Supervised Learning" else "secondary",
                     disabled=not data_ready,
                     key="btn_supervised"):
    st.session_state.section = "🎯 Supervised Learning"
    st.rerun()

if not data_ready:
    st.sidebar.warning("⚠️ Complete data cleaning (including normalization) to unlock ML sections")

st.sidebar.divider()
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    """
    **Data Mining Project Application**
    
    **Current Section:**
    """ + st.session_state.section + """
    
    **Progress:**
    - ✅ Data Loading
    - ✅ Data Cleaning  
    - ✅ Outlier Detection
    - ✅ Data Exploration
    - ✅ Data Typing
    - """ + ("✅" if st.session_state.get('normalization_complete', False) else "⏳") + """ Normalization
    """
)

if st.session_state.df is not None:
    st.sidebar.divider()
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.write(f"**Rows:** {st.session_state.df.shape[0]}")
    st.sidebar.write(f"**Columns:** {st.session_state.df.shape[1]}")
    st.sidebar.write(f"**Missing Values:** {st.session_state.df.isnull().sum().sum()}")

# ===========================
# SECTION: DATA CLEANING
# ===========================

if st.session_state.section == "📥 Data Cleaning":

    # ===========================
    # PHASE 1: DATA LOADING
    # ===========================

    st.header("📂 Phase 1: Data Loading & Handling")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Only load the file if it's a new file (prevent reloading on every rerun)
        file_id = uploaded_file.file_id
        
        if st.session_state.uploaded_file_id != file_id:
            try:
                # Load the data as-is (don't convert yet)
                df = pd.read_csv(uploaded_file)
                
                st.session_state.df = df.copy()
                st.session_state.original_df = df.copy()
                st.session_state.data_cleaned = False  # Reset cleaning flag on new upload
                st.session_state.uploaded_file_id = file_id
                
                st.success(f"✅ File uploaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    # Display data if available
    if st.session_state.df is not None:
        # Display metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", st.session_state.df.shape[0])
        with col2:
            st.metric("Columns", st.session_state.df.shape[1])
        with col3:
            st.metric("Total Cells", st.session_state.df.shape[0] * st.session_state.df.shape[1])
        
        # Show data types
        with st.expander("📋 View Column Data Types"):
            types_df = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Data Type': st.session_state.df.dtypes.values
            })
            st.dataframe(types_df, use_container_width=True)
        
        # Display the dataset
        st.subheader("Complete Dataset")
        st.dataframe(st.session_state.df, use_container_width=True, height=400)
    else:
        st.info("👆 Please upload a CSV file to begin.")

    # ===========================
    # PHASE 2: DATA CLEANING
    # ===========================

    if st.session_state.df is not None:
        st.header("🧹 Phase 2: Data Cleaning & Quality Control")
    
        # ==================
        # DUPLICATE HANDLING
        # ==================
        st.subheader("🔄 Duplicate Rows")
        
        dup_count = st.session_state.df.duplicated().sum()
        
        if dup_count > 0:
            st.warning(f"⚠️ Found **{dup_count}** duplicate rows")
            if st.button("Remove Duplicates", type="primary"):
                st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
                st.success(f"✅ Removed {dup_count} duplicate rows!")
        else:
            st.success("✅ No duplicate rows found")
        
        st.divider()
        
        # ==================
        # MISSING VALUES HANDLING
        # ==================
        st.subheader("🔍 Missing Values Handling")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        # --- COLUMN 1: Auto-Detect Problematic Values ---
        with col1:
            st.markdown("### 🤖 Auto-Detection & Cleanup")
            st.caption("Convert columns to numeric type - non-numeric values become NaN")
            
            if st.button("🔎 Scan for Problematic Values", type="primary", use_container_width=True):
                problems = detect_problematic_values(st.session_state.df)
                st.session_state.detected_problems = problems
            
            if hasattr(st.session_state, 'detected_problems'):
                problems = st.session_state.detected_problems
                
                if problems:
                    st.write(f"**Found {len(problems)} problematic value(s):**")
                    
                    problems_df = pd.DataFrame([
                        {"Value": k, "Count": v} 
                        for k, v in sorted(problems.items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(problems_df, use_container_width=True, hide_index=True)
                    
                    if st.button("🧹 Auto Clean (Convert to Numeric)", type="primary", use_container_width=True):
                        # Convert object columns to numeric
                        object_cols = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
                        converted_count = len(object_cols)
                        
                        for col in object_cols:
                            st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors='coerce')
                        
                        st.session_state.data_cleaned = True
                        st.session_state.cleaned_df = st.session_state.df.copy()  # Save cleaned state
                        if hasattr(st.session_state, 'detected_problems'):
                            delattr(st.session_state, 'detected_problems')
                        st.success(f"✅ Converted {converted_count} columns to numeric - problematic values are now NaN!")
                else:
                    st.success("✅ No problematic values detected!")
        
        # --- COLUMN 2: Manual Custom Indicators ---
        with col2:
            st.markdown("### ✏️ Manual Replacement")
            st.caption("Specify exact values to convert to NaN (e.g., '?', 'NA', 'missing')")
            
            # Form for adding indicators
            with st.form("custom_indicator_form", clear_on_submit=True):
                new_indicator = st.text_input(
                    "Enter value to mark as missing",
                    placeholder="e.g., ?, NA, unknown"
                )
                submitted = st.form_submit_button("➕ Add", use_container_width=True, type="primary")
                
                if submitted and new_indicator.strip():
                    indicator = new_indicator.strip()
                    if indicator not in st.session_state.custom_indicators:
                        st.session_state.custom_indicators.append(indicator)
            
            # Display current indicators
            if st.session_state.custom_indicators:
                st.write("**Custom Missing Indicators:**")
                
                for idx, indicator in enumerate(st.session_state.custom_indicators):
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.code(indicator, language=None)
                    with col_b:
                        if st.button("❌", key=f"del_{idx}"):
                            st.session_state.custom_indicators.remove(indicator)
                
                st.markdown("---")
                
                if st.button("✅ Replace with NaN", type="primary", use_container_width=True):
                    # Replace custom indicators with NaN
                    for col in st.session_state.df.columns:
                        for val in st.session_state.custom_indicators:
                            mask = st.session_state.df[col].astype(str).str.strip().str.lower() == str(val).strip().lower()
                            st.session_state.df.loc[mask, col] = np.nan
                    
                    # Convert object columns to numeric
                    for col in st.session_state.df.select_dtypes(include=['object']).columns:
                        st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors='coerce')
                    
                    count = len(st.session_state.custom_indicators)
                    st.session_state.data_cleaned = True
                    st.session_state.cleaned_df = st.session_state.df.copy()  # Save cleaned state
                    st.session_state.custom_indicators = []
                    st.success(f"✅ Replaced {count} custom value(s) with NaN and converted to numeric!")
            else:
                st.info("No custom indicators added yet")
        
        st.divider()
        
        # ==================
        # NaN HANDLING STRATEGIES
        # ==================
        
        st.subheader("🔧 Handle NaN Values")
        
        # Count NaN values
        nan_counts = st.session_state.df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            st.warning(f"⚠️ Dataset contains **{total_nans}** NaN values across **{len(nan_cols)}** column(s)")
            
            # Show NaN distribution
            with st.expander("📊 View NaN Distribution by Column"):
                nan_df = pd.DataFrame({
                    'Column': nan_cols.index,
                    'NaN Count': nan_cols.values,
                    'Percentage': (nan_cols.values / len(st.session_state.df) * 100).round(2)
                })
                st.dataframe(nan_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No NaN values found in the dataset!")
        
        # Strategy selection (always visible)
        col_select, col_reset = st.columns([3, 1])
        
        with col_select:
            strategy = st.selectbox(
                "Select NaN handling strategy:",
                [
                    "-- Select Strategy --",
                    "Drop rows with NaN",
                    "Drop columns with NaN",
                    "Replace with 0",
                    "Replace with mean",
                    "Replace with median",
                    "Forward fill",
                    "Backward fill"
                ],
                key="nan_strategy"
            )
        
        with col_reset:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("🔄 Reset", type="secondary", use_container_width=True):
                if st.session_state.cleaned_df is not None:
                    st.session_state.df = st.session_state.cleaned_df.copy()
                    st.success("✅ Reset to cleaned data with NaN!")
                else:
                    st.warning("⚠️ No cleaned data to reset to!")
        
        if strategy != "-- Select Strategy --":
            if st.button("🔄 Apply Strategy", type="primary"):
                    
                    if strategy == "Drop rows with NaN":
                        before = len(st.session_state.df)
                        st.session_state.df = st.session_state.df.dropna(axis=0).reset_index(drop=True)
                        st.success(f"✅ Dropped {before - len(st.session_state.df)} rows")
                    
                    elif strategy == "Drop columns with NaN":
                        before = len(st.session_state.df.columns)
                        st.session_state.df = st.session_state.df.dropna(axis=1)
                        st.success(f"✅ Dropped {before - len(st.session_state.df.columns)} columns")
                    
                    elif strategy == "Replace with 0":
                        st.session_state.df = st.session_state.df.fillna(0)
                        st.success("✅ Replaced all NaN with 0")
                    
                    elif strategy == "Replace with mean":
                        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].mean())
                        st.success("✅ Replaced NaN with column mean (numeric columns only)")
                    
                    elif strategy == "Replace with median":
                        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].median())
                        st.success("✅ Replaced NaN with column median (numeric columns only)")
                    
                    elif strategy == "Forward fill":
                        st.session_state.df = st.session_state.df.ffill()
                        st.success("✅ Applied forward fill")
                    
                    elif strategy == "Backward fill":
                        st.session_state.df = st.session_state.df.bfill()
                        st.success("✅ Applied backward fill")
        
        st.divider()
        
        # ==================
        # CURRENT DATASET
        # ==================
        st.subheader("📋 Current Dataset")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", st.session_state.df.shape[0])
        with col2:
            st.metric("Columns", st.session_state.df.shape[1])
        with col3:
            st.metric("NaN Values", st.session_state.df.isnull().sum().sum())
        
        st.dataframe(st.session_state.df, use_container_width=True, height=400)
        
        # Download button
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned Data",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv",
            type="primary"
        )

        # ===========================
        # PHASE 3: OUTLIER DETECTION
        # ===========================

        if st.session_state.df is not None:
            st.header("📉 Phase 3: Outlier Detection & Handling")
        
        # Get numeric columns only
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.warning("⚠️ No numeric columns found in the dataset. Please clean your data first.")
        else:
            st.info(f"Found **{len(numeric_cols)}** numeric column(s) for outlier analysis")
            
            # ==================
            # OUTLIER VISUALIZATION
            # ==================
            st.subheader("📊 Boxplot Visualization")
            
            # Column selector for boxplot
            selected_col = st.selectbox(
                "Select a column to visualize:",
                numeric_cols,
                key="outlier_col_select"
            )
            
            if selected_col:
                # Create boxplot
                fig = px.box(
                    st.session_state.df,
                    y=selected_col,
                    title=f"Boxplot for {selected_col}",
                    labels={selected_col: "Value"},
                    height=500
                )
                
                fig.update_layout(
                    showlegend=False,
                    yaxis_title=selected_col,
                    xaxis_title=""
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate outlier statistics for selected column
                col_data = st.session_state.df[selected_col].dropna()
                outlier_mask = detect_outliers_iqr(col_data)
                n_outliers = outlier_mask.sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Values", len(col_data))
                with col2:
                    st.metric("Outliers", n_outliers)
                with col3:
                    st.metric("Outlier %", f"{(n_outliers/len(col_data)*100):.2f}%")
                with col4:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    st.metric("IQR", f"{IQR:.2f}")
            
            st.divider()
            
            # ==================
            # OUTLIER DETECTION FOR ALL COLUMNS
            # ==================
            st.subheader("🔍 Outlier Detection Summary")
            
            # Calculate outliers for all numeric columns
            outlier_summary = []
            total_outliers = 0
            
            for col in numeric_cols:
                col_data = st.session_state.df[col].dropna()
                if len(col_data) > 0:
                    outlier_mask = detect_outliers_iqr(col_data)
                    n_outliers = outlier_mask.sum()
                    total_outliers += n_outliers
                    
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_summary.append({
                        'Column': col,
                        'Total Values': len(col_data),
                        'Outliers': n_outliers,
                        'Outlier %': f"{(n_outliers/len(col_data)*100):.2f}%",
                        'Lower Bound': f"{lower_bound:.2f}",
                        'Upper Bound': f"{upper_bound:.2f}",
                        'IQR': f"{IQR:.2f}"
                    })
            
            if outlier_summary:
                summary_df = pd.DataFrame(outlier_summary)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                st.info(f"**Total outliers detected across all columns: {total_outliers}**")
            
            st.divider()
            
            # ==================
            # OUTLIER HANDLING
            # ==================
            st.subheader("⚙️ Outlier Handling")
            
            if total_outliers > 0:
                st.warning(f"⚠️ Found **{total_outliers}** outlier(s) in the dataset")
                
                # Create tabs for different handling options
                tab1, tab2, tab3, tab4 = st.tabs(["🎯 Select Columns", "🗑️ Remove All", "🔄 Transform", "✅ Keep All"])
                
                with tab1:
                    st.markdown("**Select specific columns to remove outliers from:**")
                    
                    # Find columns that have outliers
                    cols_with_outliers = []
                    for col in numeric_cols:
                        col_data = st.session_state.df[col].dropna()
                        if len(col_data) > 0:
                            outlier_mask = detect_outliers_iqr(col_data)
                            if outlier_mask.sum() > 0:
                                cols_with_outliers.append(col)
                    
                    if cols_with_outliers:
                        # Multi-select for columns
                        selected_cols = st.multiselect(
                            "Choose columns:",
                            cols_with_outliers,
                            default=[],
                            help="Select one or more columns to remove outliers from"
                        )
                        
                        if selected_cols:
                            # Show how many rows will be affected
                            mask = pd.Series([True] * len(st.session_state.df))
                            total_to_remove = 0
                            
                            for col in selected_cols:
                                col_data = st.session_state.df[col]
                                outlier_mask = detect_outliers_iqr(col_data.dropna())
                                
                                # Create full mask for the column (including NaN positions)
                                full_mask = pd.Series([False] * len(st.session_state.df))
                                full_mask.loc[col_data.dropna().index] = outlier_mask.values
                                
                                mask = mask & ~full_mask
                            
                            rows_to_remove = len(st.session_state.df) - mask.sum()
                            
                            st.info(f"📊 This will remove **{rows_to_remove}** row(s) containing outliers in the selected column(s)")
                            
                            if st.button("🗑️ Remove Outliers from Selected Columns", type="primary", use_container_width=True):
                                original_count = len(st.session_state.df)
                                st.session_state.df = st.session_state.df[mask].reset_index(drop=True)
                                removed_count = original_count - len(st.session_state.df)
                                
                                st.success(f"✅ Removed {removed_count} row(s) containing outliers from {len(selected_cols)} column(s)!")
                        else:
                            st.info("👆 Select at least one column to remove outliers")
                    else:
                        st.info("No columns with outliers found")
                
                with tab2:
                    st.markdown("**Remove outliers from ALL numeric columns:**")
                    st.warning(f"⚠️ This will remove rows that contain outliers in ANY numeric column")
                    
                    if st.button("🗑️ Remove All Outliers", type="primary", use_container_width=True):
                        # Iteratively remove outliers until no more outliers exist
                        original_count = len(st.session_state.df)
                        iteration = 0
                        max_iterations = 10  # Safety limit
                        
                        while iteration < max_iterations:
                            # Check for outliers in current dataset
                            mask = pd.Series([True] * len(st.session_state.df))
                            found_outliers = False
                            
                            for col in numeric_cols:
                                if col in st.session_state.df.columns:
                                    col_data = st.session_state.df[col]
                                    outlier_mask = detect_outliers_iqr(col_data.dropna())
                                    
                                    if outlier_mask.sum() > 0:
                                        found_outliers = True
                                    
                                    # Create full mask for the column (including NaN positions)
                                    full_mask = pd.Series([False] * len(st.session_state.df))
                                    full_mask.loc[col_data.dropna().index] = outlier_mask.values
                                    
                                    mask = mask & ~full_mask
                            
                            # If no outliers found, break the loop
                            if not found_outliers:
                                break
                            
                            # Remove outliers
                            st.session_state.df = st.session_state.df[mask].reset_index(drop=True)
                            iteration += 1
                        
                        removed_count = original_count - len(st.session_state.df)
                        st.success(f"✅ Removed {removed_count} row(s) containing outliers in {iteration} iteration(s)!")
                
                with tab3:
                    st.markdown("**Transform data to reduce outlier impact:**")
                    st.info("ℹ️ Transformations change the distribution without deleting data")
                    
                    # Log Transformation
                    st.markdown("### 📊 Log Transformation")
                    st.caption("📐 Formula: log(x) - Reduces right skewness")
                    
                    if st.button("🔄 Apply Log Transformation", type="primary", use_container_width=True):
                        transformed_count = 0
                        for col in numeric_cols:
                            col_data = st.session_state.df[col]
                            # Shift to ensure all values are positive
                            min_val = col_data.min()
                            if min_val <= 0:
                                shift = abs(min_val) + 1
                                st.session_state.df[col] = np.log(col_data + shift)
                            else:
                                st.session_state.df[col] = np.log(col_data)
                            transformed_count += 1
                        
                        st.success(f"✅ Applied log transformation to {transformed_count} column(s)!")
                    
                    st.divider()
                    
                    # Square Root Transformation
                    st.markdown("### 📊 Square Root Transformation")
                    st.caption("📐 Formula: sqrt(x) - Reduces impact of large values")
                    
                    if st.button("🔄 Apply Square Root Transformation", type="primary", use_container_width=True):
                        transformed_count = 0
                        for col in numeric_cols:
                            col_data = st.session_state.df[col]
                            # Shift to ensure all values are non-negative
                            min_val = col_data.min()
                            if min_val < 0:
                                shift = abs(min_val)
                                st.session_state.df[col] = np.sqrt(col_data + shift)
                            else:
                                st.session_state.df[col] = np.sqrt(col_data)
                            transformed_count += 1
                        
                        st.success(f"✅ Applied square root transformation to {transformed_count} column(s)!")
                
                with tab4:
                    st.markdown("**Keep all outliers in the dataset:**")
                    st.info("ℹ️ Outliers will remain in the dataset. You can proceed to the next phase.")
                    
                    if st.button("✅ Confirm - Keep All Outliers", type="secondary", use_container_width=True):
                        st.success("✅ Outliers kept. You can proceed to the next phase!")
            else:
                st.success("✅ No outliers detected in the dataset!")
            
            st.divider()
            
            # ==================
            # CURRENT DATASET AFTER OUTLIER HANDLING
            # ==================
            st.subheader("📋 Current Dataset")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", st.session_state.df.shape[0])
            with col2:
                st.metric("Columns", st.session_state.df.shape[1])
            with col3:
                # Recalculate outliers for current dataset
                current_outliers = 0
                for col in numeric_cols:
                    if col in st.session_state.df.columns:
                        col_data = st.session_state.df[col].dropna()
                        if len(col_data) > 0:
                            outlier_mask = detect_outliers_iqr(col_data)
                            current_outliers += outlier_mask.sum()
                st.metric("Outliers", current_outliers)
            
            st.dataframe(st.session_state.df, use_container_width=True, height=400)

        # ===========================
        # PHASE 4: DATA EXPLORATION & DESCRIPTIVE ANALYSIS
        # ===========================

        if st.session_state.df is not None:
            st.header("📊 Phase 4: Data Exploration & Descriptive Analysis")
        
        # Get numeric columns only
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.warning("⚠️ No numeric columns found in the dataset. Please ensure your data is properly cleaned.")
        else:
            # ==================
            # DESCRIPTIVE STATISTICS
            # ==================
            st.subheader("📈 Descriptive Statistics")
            
            # Column selector for statistics
            selected_stat_col = st.selectbox(
                "Select a column for detailed statistics:",
                numeric_cols,
                key="stats_col_select"
            )
            
            if selected_stat_col:
                col_data = st.session_state.df[selected_stat_col].dropna()
                
                if len(col_data) == 0:
                    st.warning(f"⚠️ Column '{selected_stat_col}' has no valid numeric values")
                else:
                    # Calculate all statistics
                    min_val = col_data.min()
                    max_val = col_data.max()
                    mean_val = col_data.mean()
                    median_val = col_data.median()
                    mode_result = col_data.mode()
                    mode_val = mode_result.iloc[0] if len(mode_result) > 0 else "N/A"
                    q1_val = col_data.quantile(0.25)
                    q3_val = col_data.quantile(0.75)
                    
                    # Display in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Minimum", f"{min_val:.4f}")
                        st.metric("Q1 (25%)", f"{q1_val:.4f}")
                    
                    with col2:
                        st.metric("Mean", f"{mean_val:.4f}")
                        st.metric("Median", f"{median_val:.4f}")
                    
                    with col3:
                        st.metric("Maximum", f"{max_val:.4f}")
                        st.metric("Q3 (75%)", f"{q3_val:.4f}")
                    
                    with col4:
                        st.metric("Mode", f"{mode_val:.4f}" if mode_val != "N/A" else "N/A")
                        st.metric("Range", f"{(max_val - min_val):.4f}")
                    
                    st.divider()
                    
                    # Summary table for all numeric columns
                    st.subheader("📋 Summary Statistics for All Numeric Columns")
                    
                    summary_data = []
                    for col in numeric_cols:
                        col_data = st.session_state.df[col].dropna()
                        
                        if len(col_data) > 0:
                            mode_result = col_data.mode()
                            mode_val = mode_result.iloc[0] if len(mode_result) > 0 else np.nan
                            
                            summary_data.append({
                                'Column': col,
                                'Min': f"{col_data.min():.4f}",
                                'Q1': f"{col_data.quantile(0.25):.4f}",
                                'Mean': f"{col_data.mean():.4f}",
                                'Median': f"{col_data.median():.4f}",
                                'Q3': f"{col_data.quantile(0.75):.4f}",
                                'Max': f"{col_data.max():.4f}",
                                'Mode': f"{mode_val:.4f}" if not pd.isna(mode_val) else "N/A"
                            })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # ==================
            # SCATTER PLOT VISUALIZATION
            # ==================
            st.subheader("🔵 Scatter Plot Visualization")
            st.caption("Visualize relationships between two numeric attributes")
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox(
                        "Select X-axis:",
                        numeric_cols,
                        key="scatter_x"
                    )
                
                with col2:
                    # Filter out the selected x-axis from y-axis options
                    y_options = [col for col in numeric_cols if col != x_axis]
                    if len(y_options) > 0:
                        y_axis = st.selectbox(
                            "Select Y-axis:",
                            y_options,
                            key="scatter_y"
                        )
                    else:
                        y_axis = numeric_cols[0]
                        st.warning("⚠️ Only one numeric column available")
                
                if x_axis and y_axis:
                    # Create scatter plot
                    fig = px.scatter(
                        st.session_state.df,
                        x=x_axis,
                        y=y_axis,
                        title=f"Scatter Plot: {x_axis} vs {y_axis}",
                        labels={x_axis: x_axis, y_axis: y_axis},
                        height=500,
                        opacity=0.6
                    )
                    
                    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
                    fig.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show correlation coefficient
                    correlation = st.session_state.df[[x_axis, y_axis]].corr().iloc[0, 1]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Correlation", f"{correlation:.4f}")
                    with col2:
                        # Count valid pairs
                        valid_pairs = st.session_state.df[[x_axis, y_axis]].dropna().shape[0]
                        st.metric("Valid Data Points", valid_pairs)
                    with col3:
                        # Correlation interpretation
                        if abs(correlation) > 0.7:
                            strength = "Strong"
                        elif abs(correlation) > 0.4:
                            strength = "Moderate"
                        else:
                            strength = "Weak"
                        st.metric("Correlation Strength", strength)
            else:
                st.warning("⚠️ Need at least 2 numeric columns to create scatter plots")

        # ===========================
        # PHASE 5: DATA TYPING & OPTIMIZATION
        # ===========================

        if st.session_state.df is not None:
            st.header("🧠 Phase 5: Data Typing & Optimization")
        st.caption("Optimize data representation by converting floats to integers where appropriate")
        
        # Get all numeric columns
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.warning("⚠️ No numeric columns found in the dataset.")
        else:
            # ==================
            # DETECT FLOAT COLUMNS THAT CAN BE INTEGERS
            # ==================
            st.subheader("🔍 Float to Integer Detection")
            
            # Analyze each float column
            float_cols = st.session_state.df.select_dtypes(include=['float64', 'float32']).columns.tolist()
            
            if len(float_cols) == 0:
                st.success("✅ No float columns found. All numeric columns are already optimally typed.")
            else:
                st.info(f"Found **{len(float_cols)}** float column(s) to analyze")
                
                # Check which float columns can be converted to integers
                convertible_cols = []
                conversion_info = []
                
                for col in float_cols:
                    col_data = st.session_state.df[col].dropna()
                    
                    if len(col_data) > 0:
                        # Check if all values are integers (no decimal part)
                        is_integer = (col_data == col_data.astype(int)).all()
                        
                        if is_integer:
                            convertible_cols.append(col)
                            conversion_info.append({
                                'Column': col,
                                'Current Type': 'float64',
                                'Target Type': 'int64',
                                'Sample Values': f"{col_data.head(3).tolist()}",
                                'Non-null Count': len(col_data),
                                'Status': '✅ Convertible'
                            })
                        else:
                            conversion_info.append({
                                'Column': col,
                                'Current Type': 'float64',
                                'Target Type': '-',
                                'Sample Values': f"{col_data.head(3).tolist()}",
                                'Non-null Count': len(col_data),
                                'Status': '❌ Has decimals'
                            })
                
                # Display analysis table
                if conversion_info:
                    st.subheader("📊 Type Conversion Analysis")
                    analysis_df = pd.DataFrame(conversion_info)
                    st.dataframe(analysis_df, use_container_width=True, hide_index=True)
                
                st.divider()
                
                # ==================
                # CONVERSION CONTROLS
                # ==================
                if len(convertible_cols) > 0:
                    st.subheader("⚙️ Convert Float to Integer")
                    st.success(f"✅ Found **{len(convertible_cols)}** column(s) that can be safely converted to integers")
                    
                    # Show which columns will be converted
                    with st.expander("📋 Columns to be converted", expanded=True):
                        for col in convertible_cols:
                            st.write(f"- **{col}**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🔄 Convert to Integer", type="primary", use_container_width=True):
                            # Perform conversion
                            converted_count = 0
                            for col in convertible_cols:
                                try:
                                    # Convert to integer, keeping NaN as NaN
                                    st.session_state.df[col] = st.session_state.df[col].astype('Int64')
                                    converted_count += 1
                                except Exception as e:
                                    st.error(f"Error converting {col}: {e}")
                            
                            if converted_count > 0:
                                st.success(f"✅ Successfully converted {converted_count} column(s) to integer type!")
                            else:
                                st.warning("⚠️ No columns were converted")
                    
                    with col2:
                        if st.button("⏭️ Skip Conversion", type="secondary", use_container_width=True):
                            st.info("ℹ️ Skipped type conversion. Columns will remain as float type.")
                    
                    # Show validation info
                    st.divider()
                    st.subheader("✅ Conversion Validation")
                    st.info("""
                    **Validation Process:**
                    - ✓ Checks if all non-null values are whole numbers
                    - ✓ Preserves NaN/null values
                    - ✓ Uses Int64 type (pandas nullable integer)
                    - ✓ No data loss during conversion
                    """)
                else:
                    st.warning("⚠️ No float columns can be converted to integers")
                    st.info("All float columns contain decimal values and cannot be safely converted.")
            
            st.divider()
            
            # ==================
            # CURRENT DATA TYPES
            # ==================
            st.subheader("📋 Current Data Types")
            
            types_summary = []
            for col in st.session_state.df.columns:
                dtype = st.session_state.df[col].dtype
                non_null = st.session_state.df[col].notna().sum()
                null_count = st.session_state.df[col].isna().sum()
                
                types_summary.append({
                    'Column': col,
                    'Data Type': str(dtype),
                    'Non-null Values': non_null,
                    'Null Values': null_count
                })
            
            types_df = pd.DataFrame(types_summary)
            st.dataframe(types_df, use_container_width=True, hide_index=True)

        # ===========================
        # PHASE 6: DATA NORMALIZATION
        # ===========================

        if st.session_state.df is not None:
            st.header("⚙️ Phase 6: Data Normalization")
        st.caption("Prepare numerical data for machine learning algorithms")
        
        # Initialize normalized data in session state
        if 'normalized_df' not in st.session_state:
            st.session_state.normalized_df = None
        if 'normalization_method' not in st.session_state:
            st.session_state.normalization_method = None
        
        # Get numeric columns only
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.warning("⚠️ No numeric columns found in the dataset.")
        else:
            st.info(f"Found **{len(numeric_cols)}** numeric column(s) for normalization")
            
            # ==================
            # NORMALIZATION STRATEGY SELECTION
            # ==================
            st.subheader("📊 Select Normalization Strategy")
            
            normalization_option = st.radio(
                "Choose a normalization method:",
                [
                    "None (Skip Normalization)",
                    "Min-Max Normalization",
                    "Z-Score Standardization"
                ],
                key="norm_method"
            )
            
            # Show formulas and explanations
            with st.expander("ℹ️ Learn About Normalization Methods"):
                st.markdown("""
                ### Min-Max Normalization
                Scales values to a range [0, 1]
                
                **Formula:** $X_{norm} = \\frac{X - X_{min}}{X_{max} - X_{min}}$
                
                - **Use when:** You need values in a specific range
                - **Preserves:** Original distribution shape
                - **Sensitive to:** Outliers
                
                ---
                
                ### Z-Score Standardization
                Transforms data to have mean=0 and std=1
                
                **Formula:** $Z = \\frac{X - \\mu}{\\sigma}$
                
                Where:
                - $\\mu$ = mean
                - $\\sigma$ = standard deviation
                
                - **Use when:** Features have different scales
                - **Preserves:** Outlier information
                - **Assumes:** Approximately normal distribution
                """)
            
            st.divider()
            
            # ==================
            # PREVIEW BEFORE NORMALIZATION
            # ==================
            if normalization_option != "None (Skip Normalization)":
                st.subheader("👀 Preview: Before Normalization")
                
                # Show statistics before normalization
                preview_col = st.selectbox(
                    "Select a column to preview statistics:",
                    numeric_cols,
                    key="preview_norm_col"
                )
                
                if preview_col:
                    col_data = st.session_state.df[preview_col].dropna()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Min", f"{col_data.min():.4f}")
                    with col2:
                        st.metric("Max", f"{col_data.max():.4f}")
                    with col3:
                        st.metric("Mean", f"{col_data.mean():.4f}")
                    with col4:
                        st.metric("Std Dev", f"{col_data.std():.4f}")
            
            st.divider()
            
            # ==================
            # APPLY NORMALIZATION
            # ==================
            st.subheader("🔄 Apply Normalization")
            
            if normalization_option == "None (Skip Normalization)":
                st.info("ℹ️ No normalization will be applied. Original data will be used.")
                
                if st.button("✅ Confirm - Skip Normalization", type="secondary"):
                    st.session_state.normalized_df = st.session_state.df.copy()
                    st.session_state.normalization_method = "None"
                    st.session_state.normalization_complete = True
                    st.session_state.cleaned_data_for_ml = st.session_state.df.copy()
                    st.success("✅ Skipped normalization. Using original data.")
                    st.info("🎉 Data cleaning complete! You can now access Unsupervised and Supervised Learning sections from the sidebar.")
                    st.rerun()
            
            elif normalization_option == "Min-Max Normalization":
                st.info("📊 Min-Max will scale all numeric columns to range [0, 1]")
                
                if st.button("🔄 Apply Min-Max Normalization", type="primary", use_container_width=True):
                    # Create a copy of the dataframe
                    normalized_df = st.session_state.df.copy()
                    
                    # Apply Min-Max normalization to each numeric column
                    for col in numeric_cols:
                        col_data = normalized_df[col]
                        min_val = col_data.min()
                        max_val = col_data.max()
                        
                        # Avoid division by zero
                        if max_val - min_val != 0:
                            normalized_df[col] = (col_data - min_val) / (max_val - min_val)
                        else:
                            st.warning(f"⚠️ Column '{col}' has constant values, skipping normalization")
                    
                    st.session_state.normalized_df = normalized_df
                    st.session_state.normalization_method = "Min-Max"
                    st.session_state.normalization_complete = True
                    st.session_state.cleaned_data_for_ml = normalized_df.copy()
                    st.success(f"✅ Successfully applied Min-Max normalization to {len(numeric_cols)} column(s)!")
                    st.info("🎉 Data cleaning complete! You can now access Unsupervised and Supervised Learning sections from the sidebar.")
                    st.rerun()
            
            elif normalization_option == "Z-Score Standardization":
                st.info("📊 Z-Score will transform data to have mean=0 and std=1")
                
                if st.button("🔄 Apply Z-Score Standardization", type="primary", use_container_width=True):
                    # Create a copy of the dataframe
                    normalized_df = st.session_state.df.copy()
                    
                    # Apply Z-Score standardization to each numeric column
                    for col in numeric_cols:
                        col_data = normalized_df[col]
                        mean_val = col_data.mean()
                        std_val = col_data.std()
                        
                        # Avoid division by zero
                        if std_val != 0:
                            normalized_df[col] = (col_data - mean_val) / std_val
                        else:
                            st.warning(f"⚠️ Column '{col}' has zero std deviation, skipping normalization")
                    
                    st.session_state.normalized_df = normalized_df
                    st.session_state.normalization_method = "Z-Score"
                    st.session_state.normalization_complete = True
                    st.session_state.cleaned_data_for_ml = normalized_df.copy()
                    st.success(f"✅ Successfully applied Z-Score standardization to {len(numeric_cols)} column(s)!")
                    st.info("🎉 Data cleaning complete! You can now access Unsupervised and Supervised Learning sections from the sidebar.")
                    st.rerun()
            
            # ==================
            # SHOW NORMALIZED DATA
            # ==================
            if st.session_state.normalized_df is not None:
                st.divider()
                st.subheader("📊 After Normalization")
                st.caption(f"Method Applied: **{st.session_state.normalization_method}**")
                
                # Show statistics after normalization
                if st.session_state.normalization_method != "None":
                    preview_col = st.selectbox(
                        "Select a column to view normalized statistics:",
                        numeric_cols,
                        key="preview_norm_col_after"
                    )
                    
                    if preview_col:
                        col_data = st.session_state.normalized_df[preview_col].dropna()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Min", f"{col_data.min():.4f}")
                        with col2:
                            st.metric("Max", f"{col_data.max():.4f}")
                        with col3:
                            st.metric("Mean", f"{col_data.mean():.4f}")
                        with col4:
                            st.metric("Std Dev", f"{col_data.std():.4f}")
                
                # Display normalized dataset
                st.subheader("📋 Normalized Dataset")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", st.session_state.normalized_df.shape[0])
                with col2:
                    st.metric("Columns", st.session_state.normalized_df.shape[1])
                
                st.dataframe(st.session_state.normalized_df, use_container_width=True, height=400)
                
                # Download normalized data
                csv = st.session_state.normalized_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Normalized Data",
                    data=csv,
                    file_name="normalized_data.csv",
                    mime="text/csv",
                    type="primary"
                )

# ===========================
# SECTION: UNSUPERVISED LEARNING
# ===========================

elif st.session_state.section == "🔍 Unsupervised Learning":
    render_unsupervised_interface()

# ===========================
# SECTION: SUPERVISED LEARNING
# ===========================

elif st.session_state.section == "🎯 Supervised Learning":
    render_supervised_interface()