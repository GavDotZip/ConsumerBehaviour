# Dataset downloaded from https://www.kaggle.com/datasets/swathiunnikrishnan/amazon-consumer-behaviour-dataset/data
# Load the required library for data manipulation
library(dplyr)

# Read the CSV file into a dataframe
amazon <- read.csv("C:/Users/gavin/Downloads/Amazon_Customer_Behavior_Survey.csv")

# View the structure of the dataframe
str(amazon) 

# Summary statistics for numerical variables
summary(amazon$age)  # Summary statistics for age

# Frequency table for categorical variables
table(amazon$Gender)  # Frequency table for gender

# Proportion of customers who leave reviews
prop.table(table(amazon$Review_Left))

# Average shopping satisfaction by gender
amazon %>%
  group_by(Gender) %>%
  summarise(avg_satisfaction = mean(Shopping_Satisfaction))

# Correlation between variables
cor(amazon$age, amazon$Shopping_Satisfaction)

# Bar plot of Top 5 Purchase Categories
library(ggplot2)
# Calculate frequency of each purchase category
category_counts <- amazon %>%
  count(Purchase_Categories) %>%
  arrange(desc(n))  # Arrange in descending order of frequency

# Select top 5 categories
top_5_categories <- head(category_counts$Purchase_Categories, 5)

# Filter data for only the top 5 categories
amazon_top_5 <- amazon %>%
  filter(Purchase_Categories %in% top_5_categories)

# Plot the bar chart for top 5 categories
ggplot(amazon_top_5, aes(x = Purchase_Categories)) +
  geom_bar() +
  labs(title = "Top 5 Purchase Categories", x = "Category", y = "Count")

# Box plot of Age by Gender
ggplot(amazon, aes(x = Gender, y = age)) +
  geom_boxplot() +
  labs(title = "Age Distribution by Gender", x = "Gender", y = "Age")

# Scatter plot of Age vs. Shopping Satisfaction
ggplot(amazon, aes(x = age, y = Shopping_Satisfaction)) +
  geom_point() +
  labs(title = "Age vs. Shopping Satisfaction", x = "Age", y = "Shopping Satisfaction")

# Classification of Gender and Purchase Frequency
library(caret)

# Preprocess the data
# Convert categorical variables to factors
amazon$Gender <- as.factor(amazon$Gender)
amazon$Purchase_Frequency <- as.factor(amazon$Purchase_Frequency)
# Convert "Review_Left" to a factor (the target variable)
amazon$Review_Left <- as.factor(amazon$Review_Left)

# Split the data into training and testing sets
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(amazon$Review_Left, p = 0.7, list = FALSE)
train_data <- amazon[train_index, ]
test_data <- amazon[-train_index, ]
