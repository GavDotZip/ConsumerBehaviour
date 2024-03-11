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
    df['Cluster'] = kmeans.labels_

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(df[df['Cluster'] == 0]['Age'], df[df['Cluster'] == 0]['Spending Score (1-100)'], s=50, c='red',
                label='Cluster 1')  # Likely younger individuals with higher spending scores
    plt.scatter(df[df['Cluster'] == 1]['Age'], df[df['Cluster'] == 1]['Spending Score (1-100)'], s=50, c='blue',
                label='Cluster 2')  # Wide age range with a lower spending score
    plt.scatter(df[df['Cluster'] == 2]['Age'], df[df['Cluster'] == 2]['Spending Score (1-100)'], s=50, c='green',
                label='Cluster 3')  # Older consumers with a mid-high spending score
    # Average position for each cluster
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')
    plt.title('Clustering of Customers based on Age and Spending Score')
    plt.xlabel('Age')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Load the data
df = pd.read_csv('C:/Users/gavin/PycharmProjects/ConsumerBehaviour/src/data/Mall_Customers.csv')

# Plot the count of each gender
plot_gender_count(df)

# Plot the line chart for Age, Annual Income and Spending Score
plot_line_chart(df)

# Perform classification and evaluate accuracy
perform_classification(df)

# Perform K-Means Clustering for Age and Spending Score
perform_clustering(df)
