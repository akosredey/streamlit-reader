import streamlit as st
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 100:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

def main():
    st.title("Hello and welcome to the Data Profiling App")

    st.header("Upload your excel/csv data file")
    data_file = st.sidebar.file_uploader("Upload only csv/excel file")

    if data_file is not None:
        file_extension = data_file.name.split(".")[-1].lower()
        if file_extension == "csv":
            data = pd.read_csv(data_file)
        elif file_extension in ["xls", "xlsx"]:
            data = pd.read_excel(data_file)
        else:
            st.write("Error: Unsupported file format")
            data = None
        filtered_data = filter_dataframe(data)
        st.header('**Input DataFrame**')
        st.write("Data overview:")
        st.write(filtered_data.head())
        st.sidebar.header("Visualizations")
        plot_options = ["Bar plot", "Line chart", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)

        if selected_plot == "Bar plot":
            x_axis = st.sidebar.selectbox("Select x-axis", filtered_data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", filtered_data.columns)
            st.write("Bar plot:")
            fig, ax = plt.subplots()
            sns.barplot(x=filtered_data[x_axis], y=filtered_data[y_axis], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Scatter plot":
            x_axis = st.sidebar.selectbox("Select x-axis", filtered_data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", filtered_data.columns)
            st.write("Scatter plot:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=filtered_data[x_axis], y=filtered_data[y_axis], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Histogram":
            column = st.sidebar.selectbox("Select a column", filtered_data.columns)
            bins = st.sidebar.slider("Number of bins", 5, 100, 20)
            st.write("Histogram:")
            fig, ax = plt.subplots()
            sns.histplot(filtered_data[column], bins=bins, ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            column = st.sidebar.selectbox("Select a column", filtered_data.columns)
            st.write("Box plot:")
            fig, ax = plt.subplots()
            sns.boxplot(filtered_data[column], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Line chart":
            x_axis = st.sidebar.selectbox("Select x-axis", filtered_data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", filtered_data.columns)
            st.write("Line chart:")
            fig, ax = plt.subplots()
            sns.lineplot(x=filtered_data[x_axis], y=filtered_data[y_axis], ax=ax)
            st.pyplot(fig)

        if data is not None and not data.empty:
            ok = st.button("Generate Report")

            if ok:
                #Data Profiling
                st.write("Profiling Report")
                profile = ProfileReport(data, title="Pandas Profiling Report")
                with st.spinner("Generating Report....\nplease wait...."):
                    st.write("##Report")
                    st.components.v1.html(profile.to_html(), width=1000, height=1200, scrolling=True)
            else:
                st.write("Please upload excel or CSV format file")

if __name__ == "__main__":
    main()