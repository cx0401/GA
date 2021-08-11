import pandas as pd
import numpy as np
def ifnull(x):
    sum=0
    for i in x.columns:
        sum=sum+x[i].isnull().sum()
    return sum
df=pd.read_excel('test.xlsx')
ifnull(df)
print(df)
print(df.columns[2])