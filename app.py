import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="DM Project App", page_icon=":rocket:", layout="wide")

st.title("📊 Data Mining Project Application")

# --- Phase 1: Data Loading & Handling ---
st.header("📂 Phase 1: Data Loading & Handling")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Validate file type (Streamlit's type param already does this for basic check,
    # but we can add more specific checks if needed)
    if not uploaded_file.name.endswith('.csv'):
        st.error("Invalid file type. Please upload a CSV file.")
    else:
        st.success("CSV file uploaded successfully!")
        # Load dataset into Pandas DataFrame
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['dataframe'] = df # Store DataFrame in session state

            st.subheader("Dataset Metadata")

            # Initialize session state for data types display if not already present
            if 'show_dtypes' not in st.session_state:
                st.session_state.show_dtypes = False

            # Button to toggle data types display
            if st.button("Toggle Data Types Display"):
                st.session_state.show_dtypes = not st.session_state.show_dtypes

            # Conditional display of data types
            if st.session_state.show_dtypes:
                st.write("**Data Types:**")
                # Display data types in a vertical format
                for col_name, dtype in df.dtypes.items():
                    st.markdown(f"- **{col_name}**: `{dtype}`")
                st.markdown("---") # Separator for clarity

            # Display dataset dimensions
            st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

            st.subheader("Complete Dataset")
            # Display full dataset with scrolling support
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.info("Please ensure your CSV file is correctly formatted.")
else:
    st.info("Please upload a CSV file to begin data analysis.")

st.sidebar.header("About")
st.sidebar.info(
    "This is your Data Mining project application. "
    "Start by uploading a CSV file in Phase 1 to proceed with analysis."
)