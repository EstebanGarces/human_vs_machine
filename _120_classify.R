library(caret)
library(randomForest)
library(pROC)

preprocess_data <- function(dat){
  # Convert 'method' to a factor
  dat$source <- as.factor(dat$source)
  
  dat -> dat[!is.na(dat$coh) & !is.na(dat$div),]
  
  # Return nothing if there are not enough rows or unique sources
  if (nrow(dat) < 20 || length(unique(dat$source)) < 2) {
    return(NULL)
  }
  
  trainIndex <- createDataPartition(dat$source, p = 0.8, list = FALSE)
  train_data <- dat[trainIndex,]
  test_data <- dat[-trainIndex,]
  
  # Return nothing if there are not enough unique sources in train or test data
  if (length(unique(train_data$source)) < 2 || length(unique(test_data$source)) < 2) {
    return(NULL)
  }
  
  return(list(train = train_data, test = test_data))
}

file_names <- list.files(path = "./_100_data")

