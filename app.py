import streamlit as st
import pandas as pd
import numpy as np

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
# SIDEBAR
# ===========================
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    """
    **Data Mining Project Application**
    
    This application guides you through:
    1. ✅ Data Loading
    2. ✅ Data Cleaning (Current Phase)
    3. ⏳ Outlier Detection
    4. ⏳ Data Exploration
    5. ⏳ Normalization
    6. ⏳ Unsupervised Learning
    7. ⏳ Supervised Learning
    """
)

if st.session_state.df is not None:
    st.sidebar.divider()
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.write(f"**Rows:** {st.session_state.df.shape[0]}")
    st.sidebar.write(f"**Columns:** {st.session_state.df.shape[1]}")
    st.sidebar.write(f"**Missing Values:** {st.session_state.df.isnull().sum().sum()}")
