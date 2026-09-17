import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="EDA Interface", layout="wide")
st.title("Exploratory Data Analysis (EDA)")
st.write("Upload a csv file to start analysis")

st.sidebar.header("User Controls")
uploaded_file = st.sidebar.file_uploader("Upload csv file", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Review")
    st.write(df.head())

    rows, columns = df.shape
    st.write("Rows:", rows)
    st.write("Columns:", columns)

    st.write("Columns DataTypes:")
    st.write(df.dtypes)

    st.write("Missing Values:")
    st.write(df.isnull().sum())

    st.write("Statistical Summary:")
    st.write(df.describe())

    column = st.selectbox("Select Column for Visualization", df.columns)

    st.subheader("Visualization")

    if pd.api.types.is_numeric_dtype(df[column]):
        fig, ax = plt.subplots()
        ax.hist(df[column].dropna())
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    else:
        counts = df[column].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind='bar', ax=ax)
        ax.set_title(f"Bar Chart of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        st.pyplot(fig)