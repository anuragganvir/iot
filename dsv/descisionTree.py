import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("iris.csv")
df.head()

# Check for null values
df.info()
df.isnull().sum()

# Descriptive statistics
df.describe().T

# Finding and removing outliers for 'sepal.width'
q1 = df['sepal.width'].quantile(0.25)
q3 = df['sepal.width'].quantile(0.75)
iqr = q3 - q1
df = df[(df['sepal.width'] >= q1 - 1.5 * iqr) & (df['sepal.width'] <= q3 + 1.5 * iqr)]
print("New shape after removing outliers:", df.shape)

# Plotting to confirm removal of outliers
sns.boxplot(y=df['sepal.width'])
plt.show()

# Splitting data into features and target
Y = df['variety']
X = df.drop("variety", axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Check shapes of X and Y
print("Feature matrix shape:", X.shape)
print("Target vector shape:", Y.shape)

# Define and fit the Decision Tree Classifier on training data
dt = DecisionTreeClassifier(random_state=50)
dt.fit(x_train, y_train)

# Predict and evaluate on test set
y_pred = dt.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plotting the decision tree
plt.figure(figsize=(10, 6))
plot_tree(dt, filled=True)
plt.title("Decision Tree trained on Iris features")
plt.show()
