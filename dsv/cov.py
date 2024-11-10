import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Setting a seed so the example is reproducible
np.random.seed(4272018)

df=pd.DataFrame(np.random.randint(low=0, high=20, size=(5,2), dtype=int),columns=['commersial_Watched', 'Product Purchased'])
df

df.agg(["mean", "std"])

df.var()

df.cov()

df.corr()

plt.scatter(df['commersial_Watched'],df['Product Purchased'], color='blue');
plt.xlabel('commersial_Watched');
plt.ylabel('Product Purchased')
