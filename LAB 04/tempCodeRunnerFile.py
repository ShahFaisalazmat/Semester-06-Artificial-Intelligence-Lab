import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: bold;
            border-bottom: 2px solid #1E3A8A;
            padding-bottom: 0.5rem;
        }
        .metric-card {
            background-color: #F3F4F6;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #1E3A8A;
                margin-bottom: 1rem;
            }
            .stDataFrame {
                border: 1px solid #E5E7EB;
                border-radius: 0.5rem;
            }
            .upload-section {
                background-color: #F9FAFB;
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    border: 2px dashed #D1D5DB;
                    text-align: center;
                    margin-bottom: 1.5rem;
                }
                </style>
                """, unsafe_allow_html=True)

                # App title
                st.markdown('<h1 class="main-header">📊 Exploratory Data Analysis Dashboard</h1>', unsafe_allow_html=True)

                # ====================
                # SIDEBAR CONTROLS
                # ====================
                with st.sidebar:
                    st.markdown("### ⚙️ Controls")
                    st.markdown("---")

                    # File upload section
                    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
                    st.markdown("#### 📁 Upload Dataset")
                    uploaded_file = st.file_uploader(
                        "Choose a CSV file",
                        type=['csv'],
                        help="Upload your dataset in CSV format (max 200MB)"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if uploaded_file is not None:
                        # Load the dataset
                        try:
                            df = pd.read_csv(uploaded_file)
                            st.success(f"✅ File loaded successfully!")

                            # Dataset information in sidebar
                            st.markdown("---")
                            st.markdown("### 📋 Dataset Info")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Rows", f"{df.shape[0]:,}")
                                with col2:
                                    st.metric("Columns", f"{df.shape[1]}")

                                    # Attribute selection
                                    st.markdown("---")
                                    st.markdown("### 🔍 Attribute Selection")
                                    selected_column = st.selectbox(
                                        "Select column for analysis:",
                                        options=df.columns.tolist(),
                                        help="Select a column to visualize its distribution"
                                    )

                                    # Determine column type
                                    if selected_column:
                                        if pd.api.types.is_numeric_dtype(df[selected_column]):
                                            col_type = "Numerical"
                                        else:
                                            col_type = "Categorical"

                                            st.markdown(f"**Type:** `{col_type}`")
                                            if col_type == "Numerical":
                                                unique_vals = df[selected_column].nunique()
                                                st.markdown(f"**Unique values:** `{unique_vals:,}`")
                                            else:
                                                unique_cats = df[selected_column].nunique()
                                                st.markdown(f"**Unique categories:** `{unique_cats:,}`")

                        except Exception as e:
                            st.error(f"Error loading file: {str(e)}")
                            df = None
                            selected_column = None
                                    else:
                                        st.info("👆 Please upload a CSV file to begin analysis")
                                        df = None
                                        selected_column = None

                                        # Instructions
                                        st.markdown("---")
                                        with st.expander("📖 How to use"):
                                            st.markdown("""
                                            1. **Upload** a CSV file using the uploader above
                                            2. **Select** a column from the dropdown menu
                                            3. **Explore** the dataset metadata and visualizations
                                            4. **Analyze** numerical and categorical distributions

                                            **Test Dataset:** You can use `titanic.csv` for testing
                                            """)

                                            # ====================
                                            # MAIN CONTENT AREA
                                            # ====================
                                            if uploaded_file is not None and df is not None:
                                                # Create tabs for better organization
                                                tab1, tab2, tab3 = st.tabs(["📈 Dataset Overview", "🔢 Data Statistics", "📊 Visualizations"])

                                                # Tab 1: Dataset Overview
                                                with tab1:
                                                    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)

                                                    # Dataset preview
                                                    st.markdown("#### 📄 First 5 Rows")
                                                    st.dataframe(df.head(), use_container_width=True)

                                                    # Column information
                                                    col1, col2 = st.columns(2)

                                                    with col1:
                                                        st.markdown("#### 🏷️ Column Data Types")
                                                        dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
                                                        dtype_df.reset_index(inplace=True)
                                                        dtype_df.columns = ['Column', 'Data Type']
                                                        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

                                                        with col2:
                                                            st.markdown("#### ⚠️ Missing Values")
                                                            missing_df = pd.DataFrame({
                                                                'Column': df.columns,
                                                                'Missing Values': df.isnull().sum().values,
                                                                'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
                                                            })
                                                            st.dataframe(missing_df, use_container_width=True, hide_index=True)

                                                            # Tab 2: Data Statistics
                                                            with tab2:
                                                                st.markdown('<h2 class="sub-header">Data Statistics</h2>', unsafe_allow_html=True)

                                                                # Numerical statistics
                                                                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                                                                if numerical_cols:
                                                                    st.markdown("#### 🔢 Numerical Statistics")

                                                                    # Summary statistics for all numerical columns
                                                                    summary_stats = df[numerical_cols].describe().T
                                                                    summary_stats = summary_stats[['count', 'mean', 'std', 'min', '50%', 'max']]
                                                                    summary_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Median', 'Max']
                                                                    summary_stats = summary_stats.round(2)

                                                                    st.dataframe(summary_stats, use_container_width=True)

                                                                    # Statistics for selected column (if numerical)
                                                                    if selected_column and selected_column in numerical_cols:
                                                                        st.markdown(f"#### 📊 Statistics for `{selected_column}`")

                                                                        col1, col2, col3, col4 = st.columns(4)
                                                                        with col1:
                                                                            st.metric("Mean", f"{df[selected_column].mean():.2f}")
                                                                            with col2:
                                                                                st.metric("Median", f"{df[selected_column].median():.2f}")
                                                                                with col3:
                                                                                    st.metric("Std Dev", f"{df[selected_column].std():.2f}")
                                                                                    with col4:
                                                                                        st.metric("Missing", f"{df[selected_column].isnull().sum()}")
                                                                    else:
                                                                        st.info("No numerical columns found in the dataset")

                                                                        # Categorical statistics
                                                                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                                                                        if categorical_cols:
                                                                            st.markdown("#### 🏷️ Categorical Statistics")

                                                                            # Select a categorical column to show details
                                                                            cat_col = st.selectbox(
                                                                                "Select categorical column for detailed statistics:",
                                                                                options=categorical_cols,
                                                                                key="cat_stats"
                                                                            )

                                                                            if cat_col:
                                                                                value_counts = df[cat_col].value_counts()
                                                                                value_percentage = (df[cat_col].value_counts(normalize=True) * 100).round(2)

                                                                                cat_stats_df = pd.DataFrame({
                                                                                    'Value': value_counts.index,
                                                                                    'Count': value_counts.values,
                                                                                    'Percentage': value_percentage.values
                                                                                })

                                                                                st.dataframe(cat_stats_df, use_container_width=True, hide_index=True)

                                                                                # Tab 3: Visualizations
                                                                                with tab3:
                                                                                    if selected_column:
                                                                                        st.markdown(f'<h2 class="sub-header">Visualization: {selected_column}</h2>', unsafe_allow_html=True)

                                                                                        # Check if column is numerical or categorical
                                                                                        is_numerical = pd.api.types.is_numeric_dtype(df[selected_column])

                                                                                        if is_numerical:
                                                                                            # Handle missing values for numerical columns
                                                                                            data_to_plot = df[selected_column].dropna()

                                                                                            if len(data_to_plot) > 0:
                                                                                                st.markdown(f"#### 📈 Histogram of `{selected_column}`")

                                                                                                fig, ax = plt.subplots(figsize=(10, 6))

                                                                                                # Create histogram with better styling
                                                                                                n_bins = min(30, len(data_to_plot.unique()))
                                                                                                hist_color = '#1E3A8A'

                                                                                                ax.hist(data_to_plot, bins=n_bins, edgecolor='black',
                                                                                                alpha=0.7, color=hist_color, linewidth=1.2)

                                                                                                # Add mean and median lines
                                                                                                mean_val = data_to_plot.mean()
                                                                                                median_val = data_to_plot.median()

                                                                                                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                                                                                                label=f'Mean: {mean_val:.2f}')
                                                                                                ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                                                                                                label=f'Median: {median_val:.2f}')

                                                                                                # Customize the plot
                                                                                                ax.set_xlabel(selected_column, fontsize=12, fontweight='bold')
                                                                                                ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                                                                                                ax.set_title(f'Distribution of {selected_column}', fontsize=14, fontweight='bold', pad=20)

                                                                                                # Add grid
                                                                                                ax.grid(True, alpha=0.3, linestyle='--')

                                                                                                # Add legend
                                                                                                ax.legend()

                                                                                                # Add annotation for statistics
                                                                                                stats_text = f"Count: {len(data_to_plot):,}\nStd Dev: {data_to_plot.std():.2f}"
                                                                                                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                                                                                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                                                                                                st.pyplot(fig)

                                                                                                # Additional statistics
                                                                                                col1, col2, col3 = st.columns(3)
                                                                                                with col1:
                                                                                                    st.metric("Skewness", f"{data_to_plot.skew():.3f}")
                                                                                                    with col2:
                                                                                                        st.metric("Kurtosis", f"{data_to_plot.kurtosis():.3f}")
                                                                                                        with col3:
                                                                                                            st.metric("Range", f"{data_to_plot.max() - data_to_plot.min():.2f}")
                                                                                            else:
                                                                                                st.warning(f"Column `{selected_column}` has no valid numerical data to plot.")

                                                                                        else:  # Categorical column
                                                                                            st.markdown(f"#### 📊 Bar Chart of `{selected_column}`")

                                                                                            # Calculate frequencies
                                                                                            value_counts = df[selected_column].value_counts()
                                                                                            value_percentage = (df[selected_column].value_counts(normalize=True) * 100).round(2)

                                                                                            # Create bar chart
                                                                                            fig, ax = plt.subplots(figsize=(10, 6))

                                                                                            bars = ax.bar(range(len(value_counts)), value_counts.values,
                                                                                            color=sns.color_palette("husl", len(value_counts)),
                                                                                            edgecolor='black', linewidth=1.2)

                                                                                            # Add value labels on top of bars
                                                                                            for i, (bar, count, perc) in enumerate(zip(bars, value_counts.values, value_percentage.values)):
                                                                                                height = bar.get_height()
                                                                                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                                                                                f'{count}\n({perc}%)', ha='center', va='bottom', fontsize=9)

                                                                                                # Customize the plot
                                                                                                ax.set_xlabel(selected_column, fontsize=12, fontweight='bold')
                                                                                                ax.set_ylabel('Count', fontsize=12, fontweight='bold')
                                                                                                ax.set_title(f'Frequency Distribution of {selected_column}',
                                                                                                fontsize=14, fontweight='bold', pad=20)

                                                                                                # Set x-ticks
                                                                                                ax.set_xticks(range(len(value_counts)))
                                                                                                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')

                                                                                                # Add grid
                                                                                                ax.grid(True, alpha=0.3, linestyle='--', axis='y')

                                                                                                st.pyplot(fig)

                                                                                                # Display data table
                                                                                                st.markdown("#### 📋 Frequency Table")
                                                                                                freq_df = pd.DataFrame({
                                                                                                    'Category': value_counts.index,
                                                                                                    'Count': value_counts.values,
                                                                                                    'Percentage': value_percentage.values
                                                                                                })
                                                                                                st.dataframe(freq_df, use_container_width=True, hide_index=True)
                                                                                    else:
                                                                                        st.info("Please select a column from the sidebar to visualize")

                                                                                        # Additional information at the bottom
                                                                                        st.markdown("---")

                                                                                        col1, col2, col3 = st.columns(3)
                                                                                        with col1:
                                                                                            st.metric("Total Cells", f"{df.size:,}")
                                                                                            with col2:
                                                                                                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                                                                                                with col3:
                                                                                                    duplicate_rows = df.duplicated().sum()
                                                                                                    st.metric("Duplicate Rows", duplicate_rows)

                                                                            else:
                                                                                # Welcome screen when no file is uploaded
                                                                                st.markdown("""
                                                                                <div style='text-align: center; padding: 3rem; background-color: #F3F4F6; border-radius: 1rem;'>
                                                                                    <h2 style='color: #1E3A8A;'>🚀 Welcome to EDA Dashboard</h2>
                                                                                        <p style='font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem;'>
                                                                                            Upload a CSV file to start exploring your data
                                                                                            </p>
                                                                                            <div style='display: inline-block; background-color: white; padding: 2rem; border-radius: 0.5rem; border: 2px dashed #D1D5DB;'>
                                                                                            <h3 style='color: #374151;'>📁 Expected CSV Format:</h3>
                                                                                                <ul style='text-align: left; color: #6B7280;'>
                                                                                                    <li>First row should contain column headers</li>
                                                                                                    <li>Comma-separated values (CSV format)</li>
                                                                                                    <li>Supported encodings: UTF-8, ASCII</li>
                                                                                                    <li>Maximum file size: 200MB</li>
                                                                                                    </ul>
                                                                                                    </div>
                                                                                                    <div style='margin-top: 2rem;'>
                                                                                                    <h4 style='color: #374151;'>💡 Try with Sample Dataset:</h4>
                                                                                                        <p style='color: #6B7280;'>
                                                                                                            You can use <code>titanic.csv</code> to test all features
                                                                                                            </p>
                                                                                                            </div>
                                                                                                            </div>
                                                                                                            """, unsafe_allow_html=True)

                                                                                                            # Show example of what the app can do
                                                                                                            with st.expander("🎯 What you can analyze with this dashboard"):
                                                                                                                col1, col2, col3 = st.columns(3)

                                                                                                                with col1:
                                                                                                                    st.markdown("#### 📊 Data Overview")
                                                                                                                    st.markdown("""
                                                                                                                    - Dataset dimensions
                                                                                                                    - Column data types
                                                                                                                    - Missing values analysis
                                                                                                                    - First/last rows preview
                                                                                                                    """)

                                                                                                                    with col2:
                                                                                                                        st.markdown("#### 📈 Statistics")
                                                                                                                        st.markdown("""
                                                                                                                        - Summary statistics (mean, median, std)
                                                                                                                        - Numerical distributions
                                                                                                                        - Categorical frequencies
                                                                                                                        - Custom column analysis
                                                                                                                        """)

                                                                                                                        with col3:
                                                                                                                            st.markdown("#### 🎨 Visualizations")
                                                                                                                            st.markdown("""
                                                                                                                            - Histograms for numerical data
                                                                                                                            - Bar charts for categorical data
                                                                                                                            - Dynamic updates on selection
                                                                                                                            - Export-ready plots
                                                                                                                            """)

                                                                                                                            # Footer
                                                                                                                            st.markdown("---")
                                                                                                                            st.markdown(
                                                                                                                                "<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>"
                                                                                                                                    "EDA Dashboard v1.0 | Built with Streamlit | "
                                                                                                                                    "Upload a CSV file to begin analysis"
                                                                                                                                    "</div>",
                                                                                                                                    unsafe_allow_html=True
                                                                                                                                )