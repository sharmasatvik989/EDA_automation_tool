import streamlit as st
import pandas as pd
from transformer_model import Column_Recommendar
from eda import VisualizationFuntions

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

def main():

    column_list = []
    column_list_dtype = {}

    st.title("EDA Automation Tool")
    
    # Sidebar inputs
    st.header("ML Task Setup")
    task = st.radio("Choose your ML task", ["Prediction", "Classification"])
    st.write(f"You selected: {task}")

    #File upload - Only CSV files allowed
    st.header("Dataset Upload")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

    if uploaded_file is not None:
        
        # Read dataset
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.write(df.head())
        
        # Dataset columns 
        st.subheader("Columns : Data Types ")
        for column in df.columns:
            column_list.append(column)
            column_list_dtype[column]=df[column].dtype
        st.write(column_list_dtype)

        #Logging the Columns 
        print(column_list)
        print(column_list_dtype)

        # Data information 
        st.subheader(" Data Information")
        print(df.describe())
        st.write(df.describe(include='all'))

        st.write(df.nunique())

        # Clean the data (e.g., remove null values)
        df_clean = VisualizationFuntions.clean_data(df)
        st.subheader("Cleaned Data Preview")
        st.write(df_clean.head())



    #     # Instantiate our dummy transformer model for column recommendation
    #     recommender = Column_Recommendar(df_clean.columns.tolist(), df_clean)

    #     # Visualizations Section
    #     st.header("Data Visualizations")

    #     # 1. Distribution Plot
    #     st.subheader("Distribution Plot")
    #     col_dist, _ = recommender.recommend_for_distribution()
    #     st.write(f"Recommended column for distribution: **{col_dist}**")
    #     plot_distribution(df_clean, col_dist)

    #     # 2. Bar Chart
    #     st.subheader("Bar Chart")
    #     col_bar, _ = recommender.recommend_for_bar_chart()
    #     st.write(f"Recommended column for bar chart: **{col_bar}**")
    #     plot_bar_chart(df_clean, col_bar)

    #     # 3. Pie Chart
    #     st.subheader("Pie Chart")
    #     col_pie, _ = recommender.recommend_for_pie_chart()
    #     st.write(f"Recommended column for pie chart: **{col_pie}**")
    #     plot_pie_chart(df_clean, col_pie)

    #     # Optionally: You could also offer a download of an ipynb version of the analysis.
    #     st.info("If you prefer, you can export this analysis as a Jupyter Notebook file (ipynb) with the output.")

if __name__ == "__main__":
    main()