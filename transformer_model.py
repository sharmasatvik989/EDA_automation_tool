"""
Basic Class that acts as a transformer that will identify which columns to  
select for the visualization.
"""

class Column_Recommendar:
    def __init__(self,columns,df):
        self.columns = columns
        self.df = df
    
    def recommend_for_distribution(self):
        pass
        