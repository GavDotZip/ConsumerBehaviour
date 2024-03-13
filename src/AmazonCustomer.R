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
