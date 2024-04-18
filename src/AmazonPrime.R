# Dataset downloaded from https://www.kaggle.com/datasets/arnavsmayan/amazon-prime-userbase-dataset
# Load the required library for data manipulation
library(dplyr)

# Read the CSV file into a dataframe
prime <- read.csv("C:/Users/gavin/PycharmProjects/ConsumerBehaviour/src/data/amazon_prime_users.csv")

# View the structure of the dataframe
str(prime)

# Frequency table for categorical variables
table(prime$Gender)  # Frequency table for gender

# Average shopping satisfaction by gender
prime %>%
  group_by(Gender) %>%
  summarise(avg_satisfaction = mean(Feedback.Ratings))

# Average customer service satisfaction by gender
prime %>%
  group_by(Gender) %>%
  summarise(avg_cs_satisfaction = mean(Customer.Support.Interactions))

# Bar plot of Top 5 Purchase Categories
library(ggplot2)
# Calculate frequency of each purchase category
category_counts <- prime %>%
  count(Purchase.History) %>%
  arrange(desc(n))  # Arrange in descending order of frequency

# Select top 3 categories
top_3_categories <- head(category_counts$Purchase.History, 3)

# Filter data for only the top 3 categories
prime_top_3 <- prime %>%
  filter(Purchase.History %in% top_3_categories)

# Plot the bar chart for top 5 categories
ggplot(category_counts, aes(x = Purchase.History)) +
  geom_bar() +
  labs(title = "Top Purchases", x = "Category", y = "Count") 
