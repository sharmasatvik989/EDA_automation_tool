import streamlit as st
import pandas as pd
import random
from transformer_model import Column_Recommendar
from eda import VisualizationFuntions
from dotenv import load_dotenv 
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import altair as alt


load_dotenv()

OPEN_AI_KEY = os.environ['OPENAI_KEY']
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# def open_ai_llm_call():
    
#     client = OpenAI(api_key=OPEN_AI_KEY)
#     completion = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {
#                 "role": "user",
#                 "content": "Write a one-sentence bedtime story about a unicorn."
#             }
#         ]
#     )
#     return completion.choices[0].message.content

project_dir = Path(__file__).parent

# Find all CSVs in that directory
file_paths = sorted(glob.glob(str(project_dir / "*.csv")))


def main():

    # print(OPEN_AI_KEY)
    column_list = []
    column_list_dtype = {}

    st.title("EDA Automation Tool")
    
    # Sidebar inputs
    st.sidebar.header("ML Task Setup")
    task = st.sidebar.radio("Choose your ML task", ["Prediction", "Classification"])
    st.sidebar.write(f"You selected: {task}")

    #File upload - Only CSV files allowed
    # st.sidebar.header("Dataset Upload")
    # uploaded_file = st.file_uploader("The data Set is presented with the EDGAR")

    if not file_paths:
        st.error(f"No CSV files found in `{project_dir}`")
    else:
        for paths in file_paths:
            # Read dataset

            df = pd.read_csv(paths)
            df = df.drop(columns=["Substance"])
            print(df.shape)            # e.g. (210, 36)
            print(df.columns.tolist())
            print("Columns in df:", df.columns.tolist())
            print("Count:", len(df.columns))   # should print 36

            # 2. Build your names so they exactly match count=36
            #   – two id columns, then one per year 1990–2023 inclusive (34 years)
            years = list(range(1990, 2024))     # range end is exclusive, so use 2024
            print("Years list length:", len(years))  # should print 34

            # If your existing column headers for years are strings, convert these to strings, too:
            year_strs = [str(y) for y in years]

            new_cols = ["EDGAR Country Code", "Country"] + year_strs
            print("New names count:", len(new_cols))   # should print 36

            # 3. Sanity‐check before assigning
            # assert len(new_cols) == len(df.columns), (
            #     f"Mismatch: df has {len(df.columns)} cols but new_cols has {len(new_cols)}"
        
            st.subheader("Dataset Preview")
            st.write(df.head())

            # Dataset columns 
            for column in df.columns:
                column_list.append(column)

            #Logging the Columns 
            print(column_list)
            # print(column_list_dtype)

            # Data information 
            st.subheader(" Data Information")
            # print(df.describe())
            st.write(df.describe(include='all'))

            
            df_long = df.melt(
                                id_vars=["EDGAR Country Code", "Country"],
                                var_name="Year",
                                value_name="Value"
                            )
            wide = df_long.pivot(index="Year", columns="Country", values="Value")
            wide.index = wide.index.astype(int)
            all_countries = wide.columns.tolist()
            decade_years = list(range(1990, 2024, 10)) 


            st.title("GDP Explorer (1990–2023)")

            # — Part A: random “top 10” at decade intervals —
            if "sample10" not in st.session_state:
                st.session_state.sample10 = random.sample(all_countries, 10)

            if st.button("Shuffle 10 Countries"):
                st.session_state.sample10 = random.sample(all_countries, 10)

            st.subheader("Random 10 Countries at 10-Year Intervals")
            subset = wide.loc[decade_years, st.session_state.sample10]
            st.dataframe(subset)       # table view
            st.line_chart(subset)       # bar chart view
            st.area_chart(subset)  

            # — Part B: lookup any country & year —
            st.subheader("Lookup Specific Country & Year")
            col1, col2 = st.columns(2)
            with col1:
                chosen_country = st.selectbox("Country", all_countries)
            with col2:
                chosen_year = st.selectbox("Year", wide.index.tolist())

            val = wide.at[chosen_year, chosen_country]
            st.markdown(f"**{chosen_country}** in **{chosen_year}** → {val:.3f}")


            df_r = wide.reset_index().melt(
                id_vars="Year",
                var_name="Country",
                value_name="Value"
            )

            # 3. Define the decade ticks
            decade_years = list(range(1990, 2024, 10))  # [1990, 2000, 2010, 2020]

            # 4. Build the heatmap
            heatmap = (
                alt.Chart(df_r)
                .mark_rect()
                .encode(
                    x=alt.X(
                        "Country:N",
                        sort=wide.columns.tolist(),
                        axis=alt.Axis(labelAngle=-90, title="Country")
                    ),
                    y=alt.Y(
                        "Year:O",
                        axis=alt.Axis(values=decade_years, title="Year")
                    ),
                    color=alt.Color("Value:Q", title="CO₂"),
                    tooltip=["Country","Year","Value"]
                )
                .properties(width=800, height=400)
            )

            st.altair_chart(heatmap, use_container_width=True)
            df_r = wide.reset_index().melt(
    id_vars="Year", var_name="Country", value_name="Value"
)

# 2) filter to just those 10
            df_sel = df_r[df_r["Country"].isin(st.session_state.sample10)]

            # 3) build the chart
            five_years = list(range(1990, 2024, 5))  # [1990, 1995, …, 2020]
            chart = (
                alt.Chart(df_sel)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "Year:O",
                        axis=alt.Axis(values=five_years, title="Year")
                    ),
                    y=alt.Y("Value:Q", title="GDP"),
                    color="Country:N",
                    tooltip=["Country","Year","Value"]
                )
                .properties(width=800, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
            # # Count no of unique and distinct elements in the data columns
            # st.subheader("Data unique values")
            # st.write(df.nunique())

            # calculate_null_value_sum = df.isnull().sum()
            # Data before the null values 
            # st.subheader("Rows with null values ")
            # st.write(f"The value of every row with null values :\n",calculate_null_value_sum)
            # st.write(df[df.isnull().any(axis=1)])

            # int_columns = ["total_bill","tip","size"]


            # # The picture representation of the data 
            # with st.container(border =True):
            #     data_columns = st.multiselect("Column names",int_columns,default=int_columns)
            #     rolling_average = st.toggle(" Dynamic Content")

            # np.random.seed(42)
            # data = pd.DataFrame(np.random.randn(20, len(data_columns)), columns=data_columns)
            # if rolling_average:
            #     data = data.rolling(7).mean().dropna()

            # tab1, tab2 = st.tabs(["Chart", "Dataframe"])
            # tab1.line_chart(data, height=250)
            # tab2.dataframe(data, height=250, use_container_width=True)




            # # Clean the data (e.g., remove null values)
            # df_clean = VisualizationFuntions.clean_data(df)
            # st.subheader("Cleaned Data Preview")
            # st.write(df_clean.head())

            # func_call = open_ai_llm_call()
            # print(func_call)


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