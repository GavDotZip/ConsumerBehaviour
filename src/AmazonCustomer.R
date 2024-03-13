# Dataset downloaded from https://www.kaggle.com/datasets/swathiunnikrishnan/amazon-consumer-behaviour-dataset/data
# Load the required library for data manipulation
library(dplyr)

# Read the CSV file into a dataframe
amazon <- read.csv("C:/Users/gavin/PycharmProjects/ConsumerBehaviour/src/data/Amazon_Customer_Behavior_Survey.csv")

# View the structure of the dataframe
str(amazon) 
