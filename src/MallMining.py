import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans



def plot_gender_count(data):
    # Count the occurrences of each gender
    gender_counts = data['Gender'].value_counts()

    # Plot the bar chart for Count of Gender
    plt.bar(gender_counts.index, gender_counts.values, color=['blue', 'red'])
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Count of Each Gender')
    plt.grid(axis='y')
    plt.show()


def plot_line_chart(data):
    # Plot the line chart for Age, Annual Income and Spending Score
    plt.figure(figsize=(11, 8))
    sns.lineplot(data=data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], palette='magma')
    plt.title('Line Plot of Age, Annual Income, and Spending Score')
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.show()


def perform_classification(data):
    # Prepare features and target variable
    X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    y = data['Gender']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def perform_clustering(data):
    X = df[['Age', 'Spending Score (1-100)']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

# Load the data
df = pd.read_csv('C:/Users/gavin/PycharmProjects/ConsumerBehaviour/src/data/Mall_Customers.csv')

# Plot the count of each gender
plot_gender_count(df)

# Plot the line chart for Age, Annual Income and Spending Score
plot_line_chart(df)

# Perform classification and evaluate accuracy
perform_classification(df)
