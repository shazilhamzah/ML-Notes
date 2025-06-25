import pandas as pd
import numpy as np


"""
    INTRO TO DATAFRAMES
"""
df = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],columns=["A","B","C"],index=["x","y","z"])
print("COMPLETE DF",df,sep="\n",end="\n\n")
print("FIRST TWO ROWS",df.head(2),sep="\n",end="\n\n")          # DISPLAY FIRST TWO ROWS
print("LAST TWO ROWS",df.tail(2),sep="\n",end="\n\n")           # DISPLAY LAST TWO ROWS
print("VIEW HEADERS",df.columns,sep="\n",end="\n\n")            # VIEW HEADERS
print("VIEW INDEXES",df.index.to_list,sep="\n",end="\n\n")      # VIEW INDEXES
print("VIEW INFO",df.info(),sep="\n",end="\n\n")                # VIEW INFO OF DF
print("DESCRIBE DF",df.describe(),sep="\n",end="\n\n")          # DESCRIBE - STATISTICAL SUMMARY OF DF + 5 POINT SUMMARY
print("VIEW UNIQUE VALUES OF DF",df['A'].nunique,sep="\n",end="\n\n")          # VIEW UNIQUE VALUES
print("SHAPE OF DF",df.shape,sep="\n",end="\n\n")               # VIEW SHAPE OF DF
print("TOTAL ELEMENTS OF DF",df.size,sep="\n",end="\n\n")       # VIEW SIZE OF DF (TOTAL ELEMENTS)

"""
    LOADING DATAFRAMES FROM FILES
"""
coffee = pd.read_csv('./warmup-data/coffee.csv')                # READ CSV
print("COFFEE CSV FILE: ",coffee.head(),sep="\n",end="\n\n")       
results = pd.read_parquet('./data/results.parquet')             # READ PARQUET
print("RESULTS PARQUET FILE: ",results.head(),sep="\n",end="\n\n")
# olympics_data = pd.read_excel('./data/olympics-data.xlsx')             # READ EXCEL
# print("RESULTS EXCEL FILE: ",olympics_data.head(),sep="\n",end="\n\n")

"""
    ACCESSING DATA
"""
print("ACCESSING DATA USING loc[]",coffee.loc[[0,1,5],"Day"],sep="\n",end="\n\n")                 # loc[ROW,COLUMN]
print("ACCESSING DATA USING iloc[]",coffee.iloc[0:6,1],sep="\n",end="\n\n")                       # iloc[ROW,COLUMN] BUT ONLY INDEXES 0,1,..
print("ACCESSING DATA USING at[]",coffee.at[0,"Day"],sep="\n",end="\n\n")                         # at[ROW,COLUMN] GIVE EXACT ANSWER
print("DESCENDING ORDER",coffee.sort_values("Units Sold",ascending=False),sep="\n",end="\n\n")    # SORT VALUES IN DESCENDING ORDER

"""
    FILTERING DATA
"""
bios = pd.read_csv('./data/bios.csv')                # READ CSV
print("BIOS DATA",bios.head(),sep="\n",end="\n\n")
print("PLAYERS WITH HEIGHT > 215cm",bios.loc[(bios['height_cm']>215)&(bios['born_country']=='USA'),['name','height_cm']],sep="\n",end="\n\n") 
print("FILTERING USING QUERY",bios.query('born_country=="USA" and born_city=="Seattle"'),sep="\n",end="\n\n") 

"""
    STRING OPERATIONS
"""
print("PLAYERS WITH HEIGHT > 215cm",bios[bios['name'].str.contains("Keith",case=True)],sep="\n",end="\n\n") 

"""
    MANIPULATING COLUMNS
"""
coffee['Price'] = np.where(coffee['Coffee Type']=='Espresso',3.99,5.99)
print("ADDED A price COLUMN",coffee.head(),sep="\n",end="\n\n") 