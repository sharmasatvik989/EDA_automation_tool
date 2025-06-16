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

#To load environment variables
load_dotenv()

#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Parent Directory Path
project_dir = Path(__file__).parent

# Find all CSVs in that directory
file_paths = sorted(glob.glob(str(project_dir / "*.csv")))

#
def function_one():

    column_list = []
    column_list_dtype = {}

    st.title("Renewable Energy data transformation")
    
    # Sidebar inputs
    st.sidebar.header("Energy Sector")
    task = st.sidebar.radio("Choose your ML task", ["Prediction", "Classification"])
    st.sidebar.write(f"You selected: {task}")

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

            # Count no of unique and distinct elements in the data columns
            st.subheader("Data unique values")
            st.write(df.nunique())

            # Dataset columns 
            for column in df.columns:
                column_list.append(column)

            #Logging the Columns 
            print(column_list)

            # Data information 
            st.subheader(" Data Information")
            st.write(df.describe(include='all'))

            #To convert the wide format to long format since we have mulitple year for same country 
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

            # table view
            st.dataframe(subset)   
            # bar chart view    
            st.line_chart(subset)
               
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


if __name__ == "__main__":
    function_one()