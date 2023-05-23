# Load necessary libraries
library(caret)
library(randomForest)
library(gbm)


# Load the train and test datasets
train_data <- read.csv("D:/4th year CS/Distrputed Computing/House-Pricing-Project/train.csv")
test_data <- read.csv("D:/4th year CS/Distrputed Computing/House-Pricing-Project/test.csv")



###### EXPLORE Data ######
ExploreData <- function(data) {
  print(ncol(data))
  print(nrow(data))
  print(sum(sum(is.na(data))))
  print(summary(data))
}

##### Dropping Columns That Have More than a Percentage of NULLs #####
RemoveNullColumns <- function(threshold, train_data, test_data) {
  # Calculate NULLs Percentage on each Column
  nulls <- colSums(is.na(train_data)) / nrow(train_data) * 100
  print(nulls)
  
  columns_to_drop <- names(nulls[nulls > threshold])
  print(columns_to_drop)
  
  # Drop Columns
  train_data <- train_data[, !(names(train_data) %in% columns_to_drop)]
  train_data <- as.data.frame(train_data)
  test_data <- test_data[, !(names(test_data) %in% columns_to_drop)]
  test_data <- as.data.frame(test_data)
  
  return(list(train_data, test_data))
}

#### Encoding ####
EncodeCategoricalData <- function(train_data, test_data) {
  # Save the target variable from the train dataset
  target_variable <- train_data$SalePrice
  
  # Remove the target variable from the train dataset
  train_data <- train_data[, -which(names(train_data) == "SalePrice")]
  
  # Concat train and test datasets
  combined_data <- rbind(train_data, test_data)
  
  
  # Apply one-hot encoding
  encoded_data <- predict(dummyVars(~., data = combined_data, fullRank = TRUE), newdata = combined_data)
  
  # Split the encoded data back into train and test
  encoded_train_data <- encoded_data[1:nrow(train_data), ]
  encoded_test_data <- encoded_data[(nrow(train_data) + 1):nrow(encoded_data), ]
  
  # Cast to dataframes
  encoded_train_data <- as.data.frame(encoded_train_data)
  encoded_test_data <- as.data.frame(encoded_test_data)
  
  # Continue with your model training using the train_processed dataset
  
  return(list(encoded_train_data, encoded_test_data, target_variable))
}

FillNulls <- function(data){
  
  for (col in names(data)) {
    # Calculate the median for the column
    col_median <- median(data[[col]], na.rm = TRUE)
    
    # Replace null values with the column median
    data[[col]][is.na(data[[col]])] <- col_median
  }
  
  return(data)
}

# Apply All Pre processing and return TrainIDs, TestIDs, XTrain, YTrain, XTest
Preprocess <- function(train_data, test_data){
  
  # Save Data IDs
  train_ids <- train_data$Id
  test_ids <- test_data$Id
  
  # Remove IDs from dataset
  train_data <- train_data[, -which(names(train_data) == "Id")]
  test_data <- test_data[, -which(names(test_data) == "Id")]
  
  
  ######### Remove Columns that contains more the {30}% Null Values #########
  processed_data <- RemoveNullColumns(15, train_data, test_data)
  train_data <- as.data.frame(processed_data[1])
  test_data <- as.data.frame(processed_data[2])
  ######### _______________________________________________________ #########
  

  ######### Apply One Hot Encoding To all Categorical Data ######### 
  processed_data <- EncodeCategoricalData(train_data, test_data)
  x_train <- as.data.frame(processed_data[1])
  x_test <- as.data.frame(processed_data[2])
  y_train <- as.data.frame(processed_data[3])
  ######### ______________________________________________ #########
  
  
  ######### Fill Nulls With Median Value of each columns ######### 
  x_train <- as.data.frame(FillNulls(x_train))
  x_test <- as.data.frame(FillNulls(x_test))
  ######### ________________________________ ___________ #########
  
  
  return(list(train_ids, x_train, y_train, test_ids,x_test))
}

# Split The Data into Train & Validate
TrainValidateSplit <- function(x, y, ids, splitSize=0.8, seed=100) {
  # Add Ids
  x$Id <- ids
  y$Id <- ids
  
  set.seed(seed)
  train_index <- createDataPartition(y$SalePrice, p= splitSize, list =FALSE)
  x_train <- x[train_index, ]
  y_train <- y[train_index, ]
  x_val <- x[-train_index, ]
  y_val <- y[-train_index, ]
  
  return(list(x_train, y_train, x_val, y_val))
}

# Remove Id from train data
process_train_data <- function(x_train, y_train) {
  # Remove 'Id' column from x_train
  x_train <- x_train[, !(names(x_train) %in% c('Id'))]
  
  # Remove 'Id' column from y_train
  y_train <- y_train[, !(names(y_train) %in% c('Id'))]
  
  # Create a data frame with 'SalePrice' column from y_train
  SalePrice <- data.frame(SalePrice = y_train)
  
  # Combine x_train and SalePrice into a new data frame without 'Id' column
  train_data <- cbind(x_train, SalePrice)
  
  # Return the modified train_data
  return(train_data)
}

# Remove least correlation 
remove_low_correlation <- function(x_train, y_train, threshold = 0.4) {
  correlation_matrix <- cor(x_train, y_train)
  abs_correlation <- abs(correlation_matrix)
  
  num_vars_to_keep <- ceiling((1 - threshold) * ncol(x_train))
  sorted_vars <- order(abs_correlation, decreasing = TRUE)
  
  x_train_filtered <- x_train[, sorted_vars[1:num_vars_to_keep]]
  
  return(x_train_filtered)
}




# Exploring Dataset
ExploreData(train_data)
ExploreData(test_data)

# Apply Preprocessing
processed_data <- Preprocess(train_data, test_data)
train_ids <- as.data.frame(processed_data[1])
x_train <- as.data.frame(processed_data[2])
y_train <- as.data.frame(processed_data[3])
colnames(y_train)[1]<- "SalePrice"
test_ids <- as.data.frame(processed_data[4])
x_test <- as.data.frame(processed_data[5])

# Calculate Correlations
str(x_train)
x_train <- remove_low_correlation(x_train, y_train, threshold = 0.5)
str(x_train)

# Split The Data
splitted_data <- TrainValidateSplit(x_train, y_train, train_ids, 0.8, 120)
x_train <- as.data.frame(splitted_data[1])
y_train <- as.data.frame(splitted_data[2])
x_val <- as.data.frame(splitted_data[3])
y_val <- as.data.frame(splitted_data[4])



#Create the model
#removing id from train data
train_data <- process_train_data(x_train, y_train)


# Fit the random forest model
#model <- randomForest(SalePrice ~ ., data = train_data, ntree = 200)
gbm_model <- gbm(SalePrice ~ ., data = train_data, n.trees = 1000, shrinkage = 0.01, interaction.depth = 4)


# Predictions for x_val
y_val <- y_val[, !(names(y_val) %in% c('Id'))]
SalePrice <- data.frame(SalePrice=y_train)
#predictions <- predict(model, newdata = x_val)
predictions <- predict(gbm_model, newdata = x_val)
str(predictions)



# Evaluate the model
mse <- mean((predictions - y_val)^2)  # Mean squared error
rmse <- sqrt(mean((predictions - y_val)^2))
r_squared <- cor(predictions, y_val)^2  # R-squared value

# Print evaluation metrics
cat("Evaluation Metrics:\n")
cat("MSE:", mse, "\n")
cat("RMSE:", rmse, "\n")
cat("R-squared:", r_squared, "\n")

# Create a data frame with 'Id' and 'SalePrice' columns
predictions <- predict(gbm_model, newdata = x_test)
start_id <- 1461
end_id <- 2919

# Create a data frame with sequential 'Id' and 'SalePrice' columns
predictions_df <- data.frame(Id = seq(start_id, end_id), SalePrice = predictions)


# Save the data frame to a CSV file
write.csv(predictions_df, file = "D:/4th year CS/Distrputed Computing/House-Pricing-Project/predictions.csv", row.names = FALSE)






