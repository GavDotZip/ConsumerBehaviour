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
