# Load necessary libraries
library(data.table)
library(caret)
library(rpart)
library(dplyr)
library(tidyverse)
library(vtreat)
library(dataPreparation)

# Load the train and test datasets
train_data <- fread("D:/4th year CS/Distrputed Computing/Project/train.csv")
test_data <- fread("D:/4th year CS/Distrputed Computing/Project/test.csv")



###### EXPLORE Data ######

explore_data <- function(data) {
  print(is.data.frame(data))
  print(ncol(data))
  print(nrow(data))
}


##### Removing Nulls #####
removeColumnsNulls <- function(data) {
  to.remove <- c("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "LotFrontage")
  `%ni%` <- Negate(`%in%`)
  train_data <- subset(data, select = names(data) %ni% to.remove)
  return(train_data)
}
removeRowsNulls <- function(data) {
  data <- data[complete.cases(data), ]
  return(data)
}


#### Encoding ####
encode_categorical_data <- function(data) {
  encoded_data <- model.matrix(~ . - 1, data = data)
  return(encoded_data)
}



#### Get Char Cols ####
get_character_columns <- function(data) {
  character_cols <- c()
  
  for (col in names(data)) {
    if (is.character(data[[col]])) {
      unique_count <- length(unique(data[[col]]))
      cat(col, "with", unique_count, "unique values\n")
      character_cols <- c(character_cols, col)
    }
  }
  
  return(character_cols)
}








#explore data
explore_data(train_data)
explore_data(test_data)

#remove nulls
train_data <- removeColumnsNulls(train_data)
train_data <- removeRowsNulls(train_data)
test_data <- removeColumnsNulls(test_data)
test_data <- removeRowsNulls(test_data)

#encoding
train_data <- encode_categorical_data(train_data)
test_data <- encode_categorical_data(test_data)
data <- as.data.frame(encoded_train_data)
encoded_test_data <- as.data.frame(encoded_test_data)
explore_data(encoded_train_data)
explore_data(encoded_test_data)


# Calculate correlation matrix
correlation_matrix <- cor(data[, -which(names(data) == "SalePrice")], data$SalePrice)
print(correlation_matrix)



#splitting data
set.seed(42)  # Set a seed for reproducibility
train_indices <- sample(1:nrow(data), 0.8 * nrow(data))  # 70% for training
train <- data[train_indices, ]
test <- data[-train_indices, ]

#create the model
model <- lm(SalePrice ~ ., data = train)
predictions <- predict(model, newdata = test)
actual <- test$SalePrice

# Evaluate the model
mse <- mean((predictions - actual)^2)  # Mean squared error
rmse <- sqrt(mse)  # Root mean squared error
r_squared <- cor(predictions, actual)^2  # R-squared value

# Print evaluation metrics
cat("Evaluation Metrics:\n")
cat("MSE:", mse, "\n")
cat("RMSE:", rmse, "\n")
cat("R-squared:", r_squared, "\n")



#### Apply the Model ####
finpredictions <- predict(model, newdata = test_data)
results <- data.frame(Id = test_data$Id, SalePrice = finpredictions)
write.csv(results, "predictions.csv", row.names = FALSE)





