import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import randint
import warnings
warnings.filterwarnings('ignore')

test=randint(30,70,1000)
print(test)

import statistics

mean=statistics.mean(test)
print("mean:",mean)

variance=statistics.variance(test)
print("variance:",variance)

median=statistics.median(test)
print("median:",median)

mode=statistics.mode(test)
print("mode:", mode)

Standard_Deviation=np.std(test)
print("standard_deviation",Standard_Deviation)

size=10000
values=np.random.normal(mean,Standard_Deviation,size)

plt.hist(values,100)
# plotting mean
plt.axvline(values.mean(),color='b',linestyle="--",linewidth=2)
plt.show()

# Central Limit Therom
roll=randint(1,6,100)
print(roll)

from numpy import mean
means=[mean(randint(1,6,100)) for _ in range(1000)]
plt.hist(means)
plt.show()

mean(means)

import seaborn as sns
sns.displot(data=means, kind="kde")

data=randint(1,6,1000000)
data
statistics.mean(data)

# Correlation
import sklearn

np.random.seed(100)

#create array of 50 random integers between 0 and 10
x = np.random.randint(0, 10, 50)

np.corrcoef(x,x)


#create a positively correlated array with some random noise
y = x + np.random.randint(10, 20, 50)

#calculate the correlation between the two arrays
np.corrcoef(x,y)

# plotting the data
plt.scatter(x, y)

ax = sns.heatmap(np.corrcoef(x,y), annot=True)

