import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("/content/Iris (1).csv")
data.head(5)

data['Species'].value_counts()

data['SepalLengthCm'].plot(kind='hist')
plt.show()

data.drop("Id", axis=1).hist(by="Species", figsize=(12, 10))

data.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=data, size=5)

sns.FacetGrid(data, hue="Species", palette="husl", height=5) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()

sns.FacetGrid(data, hue="Species", palette="husl", height=5) \
   .map(plt.scatter, "PetalLengthCm", "PetalWidthCm") \
   .add_legend()

 #We can look at an individual feature in Seaborn through mnay different kinds of plots.
# Here's a boxplot
sns.boxplot(x="Species", y="PetalLengthCm", palette="husl", data=data)

sns.FacetGrid(data, hue="Species", palette="husl", height=5)\
     .map(sns.kdeplot, "PetalLengthCm") \
   .add_legend()

# Now that we've covered seaborn, let's go back to pandas to see what kinds of plots
# we can make with the pandas library.
# We can quickly make a boxplot with Pandas on each feature split out by species
data.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 10))

sns.pairplot(data.drop("Id", axis=1), hue="Species", size=3)
plt.show()

from pandas.plotting import  parallel_coordinates
parallel_coordinates(data.drop("Id", axis=1), "Species",colormap='rainbow')
plt.show()

from pandas.plotting import andrews_curves
andrews_curves(data.drop("Id", axis=1), "Species",colormap='rainbow')
plt.show()

