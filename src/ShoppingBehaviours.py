import numpy as np
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


def customer_stacked_bar_chart(data):
    # Grouping the data by age and gender and counting occurrences
    age_gender_counts = data.groupby(['Age', 'Gender']).size().unstack(fill_value=0)

    # Plotting the stacked bar chart
    age_gender_counts.plot(kind='bar', stacked=True, figsize=(10, 6))

    # Adding labels and title
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Stacked Bar Chart of Age and Gender')

    # Show plot
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.show()


def deeper_customer_bar_chart(data):
    # Grouping the data by age and gender and counting occurrences
    age_gender_counts = data.groupby(['Age', 'Gender']).size().unstack(fill_value=0)

    # Calculating average review rating and purchase amount by age
    avg_review_rating = data.groupby('Age')['Review Rating'].mean()
    avg_purchase_amount = data.groupby('Age')['Purchase Amount (USD)'].mean()

    # Plotting the stacked bar chart
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Stacked bar chart
    age_gender_counts.plot(kind='bar', stacked=True, ax=ax1, color=['skyblue', 'salmon'])

    # Adding labels and title for bar chart
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Count')
    ax1.set_title('Stacked Bar Chart of Age and Gender (Avg Rating & Purchase Amount (USD)')

    # Adding legend for gender
    ax1.legend(title='Gender')

    # Creating a secondary y-axis for the line plots
    ax2 = ax1.twinx()

    # Line plot for average review rating
    avg_review_rating.plot(ax=ax2, color='green', linestyle='--', marker='o', label='Avg Review Rating')

    # Line plot for average purchase amount
    avg_purchase_amount.plot(ax=ax2, color='purple', linestyle='-', marker='s', label='Avg Purchase Amount (USD)')

    # Adding labels and title for line plots
    ax2.set_ylabel('Average')

    # Adding legend for lines
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    # Show plot
    plt.tight_layout()
    plt.show()


def violin_frequency_by_age(data):
    # Filter out rows with NaN values in 'Age' and 'Frequency of Purchases'
    data.dropna(subset=['Age', 'Frequency of Purchases'], inplace=True)

    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Frequency of Purchases', y='Age', data=data, inner='quartile')

    # Adding labels and title
    plt.xlabel('Frequency of Purchases')
    plt.ylabel('Age')
    plt.title('Violin Plot of Age and Frequency of Purchases')

    # Show plot
    plt.show()


# Load the data
df = pd.read_csv('C:/Users/gavin/PycharmProjects/ConsumerBehaviour/src/data/shopping_behavior_updated.csv')

# Plot the Bar Chart for Top 5 Customers
# top_spenders(df)

# Perform K-Means Clustering on Age and Previous Purchases
# cluster_by_frequency(df)

# Plot a Stacked Customer Bar Chart by Age and Gender
# customer_stacked_bar_chart(df)

# Plot a Stacked Bar Customer Bar Chart by Age and Gender
# Include Average Purchase Amount (USD) and Average Review Rating
deeper_customer_bar_chart(df)

violin_frequency_by_age(df)
