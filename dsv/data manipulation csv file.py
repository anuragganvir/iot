import pandas as pd
import numpy as np

# read datasdet using pandas
df = pd.read_csv('/content/employees.csv')
df.head(5)

df.shape

df.describe()

df.info()

df.nunique()

df.isnull().sum()

df['Gender'].fillna("No_gender", inplace=True)
df.isnull().sum()

mode = df['Senior Management'].mode().values[0]
df['Senior Management']= df['Senior Management'].replace(np.nan, mode)
df.isnull().sum()

df = df.dropna(axis = 0, how ='any')
print(df.isnull().sum())

df.shape

