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
