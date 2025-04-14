import streamlit as st
import pandas as pd
from transformer_model import Column_Recommendar
from eda import VisualizationFuntions
from dotenv import load_dotenv 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from openai import OpenAI


load_dotenv()

OPEN_AI_KEY = os.environ['OPENAI_KEY']
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

def open_ai_llm_call():
    
    client = OpenAI(api_key=OPEN_AI_KEY)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "Write a one-sentence bedtime story about a unicorn."
            }
        ]
    )
    return completion.choices[0].message.content


def main():

    print(OPEN_AI_KEY)
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

        # Count no of unique and distinct elements in the data columns
        st.subheader("Data unique values")
        st.write(df.nunique())

        calculate_null_value_sum = df.isnull().sum()
        # Data before the null values 
        st.subheader("Rows with null values ")
        st.write(f"The value of every row with null values :\n",calculate_null_value_sum)
        st.write(df[df.isnull().any(axis=1)])

        int_columns = ["total_bill","tip","size"]


        # The picture representation of the data 
        with st.container(border =True):
            data_columns = st.multiselect("Column names",int_columns,default=int_columns)
            rolling_average = st.toggle(" Dynamic Content")

        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(20, len(data_columns)), columns=data_columns)
        if rolling_average:
            data = data.rolling(7).mean().dropna()

        tab1, tab2 = st.tabs(["Chart", "Dataframe"])
        tab1.line_chart(data, height=250)
        tab2.dataframe(data, height=250, use_container_width=True)




        # # Clean the data (e.g., remove null values)
        # df_clean = VisualizationFuntions.clean_data(df)
        # st.subheader("Cleaned Data Preview")
        # st.write(df_clean.head())

        func_call = open_ai_llm_call()
        print(func_call)


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