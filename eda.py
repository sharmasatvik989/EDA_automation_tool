import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class VisualizationFuntions:

    def clean_data(df:pd.DataFrame)->pd.DataFrame:
        """
        1. Remove nulls ( drop rows with missing values)
        """
        df = df.dropna()
        df = df.drop_duplicates()
        
        return df.dropna()

    def plot_distribution(df:pd.DataFrame, col:str):
        """
        The following function will plot the distribution of 
        a particular column.
        """
        plt.figure(figsize=(8,4))
        sns.histplot(df[col], kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        st.pyplot(plt)
        plt.clf()

    def plot_bar_chart(df: pd.DataFrame, col: str):
        """Plot a bar chart for the counts of a categorical column."""
        plt.figure(figsize=(8, 4))
        data = df[col].value_counts()
        sns.barplot(x=data.index, y=data.values, palette='viridis')
        plt.title(f'Bar Chart of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(plt)
        plt.clf()

    def plot_pie_chart(df: pd.DataFrame, col: str):
        """Plot a pie chart for the proportions of a categorical column."""
        plt.figure(figsize=(6, 6))
        data = df[col].value_counts()
        plt.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title(f'Pie Chart of {col}')
        st.pyplot(plt)
        plt.clf()