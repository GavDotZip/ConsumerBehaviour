import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def top_spenders(data):
    # Grouping the data by Customer ID and summing the purchase amounts
    top_spenders = data.groupby('Customer ID')['Purchase Amount (USD)'].sum()

    # Sorting the top spenders in descending order and selecting the top 5
    top_spenders = top_spenders.sort_values(ascending=False).head(5)

    # Creating the bar chart
    plt.figure(figsize=(10, 6))

    # Plotting the bar chart
    top_spenders.plot(kind='bar', color='skyblue')

    # Adding labels and title
    plt.xlabel('Customer ID')
    plt.ylabel('Total Purchase Amount (USD)')
    plt.title('Top 5 Spenders by Customer ID')

    # Rotating x-axis labels for better readability
    plt.xticks(rotation=0)

    # Show plot
    plt.tight_layout()
    plt.show()


def cluster_by_frequency(data):
    # Extracting relevant columns (age and season)
    X = data[['Age', 'Previous Purchases']]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)

    # Purple = Older Age, Less Purchases
    # Green = Younger Age, More Purchases
    # Yellow = Older Age, More Purchases

    kmeans.fit(X_scaled)
    clusters = kmeans.predict(X_scaled)

    # Visualize the clusters
    plt.figure(figsize=(10, 6))

    # Plotting the data points with color-coded clusters
    scatter = plt.scatter(X['Age'], X['Previous Purchases'], c=clusters, cmap='viridis', s=50, alpha=0.5)
    # Adding color legend
    plt.legend(*scatter.legend_elements(), title='Clusters')
    plt.xlabel('Age')
    plt.ylabel('Previous Purchases')
    plt.title('Clustering Customers based on Age and Previous Purchases')

    # Show plot
    plt.show()


def customer_regression(data):
    # Selecting features and target variable
    X = data[['Age', 'Purchase Amount (USD)', 'Review Rating']]
    y = data['Subscription Status']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initializing and training the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Generating a classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# Load the data
df = pd.read_csv('C:/Users/gavin/PycharmProjects/ConsumerBehaviour/src/data/shopping_behavior_updated.csv')

# Plot the Bar Chart for Top 5 Customers
top_spenders(df)

# Perform K-Means Clustering on Age and Previous Purchases
cluster_by_frequency(df)

customer_regression(df)
