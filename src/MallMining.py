import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:/Users/gavin/PycharmProjects/ConsumerBehaviour/src/data/Mall_Customers.csv')


# Count the occurrences of each gender
gender_counts = df['Gender'].value_counts()

# Plot the bar chart
plt.bar(gender_counts.index, gender_counts.values, color=['blue', 'red'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Count of Each Gender')
plt.grid(axis='y')
plt.show()

# plt.figure(figsize=(11, 8))
# sns.lineplot(data=df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], palette='magma')
# plt.title('Line Plot of Age, Annual Income, and Spending Score')
# plt.xlabel('Data Points')
# plt.ylabel('Values')
# plt.show()