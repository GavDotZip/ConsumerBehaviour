import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:/Users/gavin/PycharmProjects/ConsumerBehaviour/src/data/Mall_Customers.csv')

plt.figure(figsize=(11, 8))
sns.lineplot(data=df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], palette='magma')
plt.title('Line Plot of Age, Annual Income, and Spending Score')
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.show()