library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(PRROC)

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

# Functions to classify and evaluate a dataset
fit_random_forest <- function(split_data){
  rf_model <- randomForest(
    source ~ div + coh,
    data = split_data$train,
    ntree = 500,
    importance = FALSE
  )
  
  rf_pred_test <- predict(rf_model, split_data$test)
  rf_prob_test <- predict(rf_model, split_data$test, type = "prob")
  
  return(list(model = rf_model, preds = rf_pred_test, probs = rf_prob_test))
}

fit_logistic_reg <- function(split_data){
  logit_model <- glm(
    source ~ div + coh,
    data = split_data$train,
    family = binomial(link = "logit")
  )
  
  logit_pred_test <- predict(logit_model, split_data$test, type = "response")
  
  logit_preds <- ifelse(
    logit_pred_test > 0.5,
    levels(split_data$test$source)[2],
    levels(split_data$test$source)[1]
  ) |> 
    factor(levels = levels(split_data$test$source))
  
  return(list(model = logit_model, preds = logit_preds, probs = logit_pred_test))
}

fit_naive_bayes <- function(split_data){
  nb_model <- naiveBayes(
    source ~ div + coh,
    data = split_data$train
  )
  
  nb_pred_test <- predict(nb_model, split_data$test, type = "class")
  nb_prob_test <- predict(nb_model, split_data$test, type = "raw")
  
  return(list(model = nb_model, preds = nb_pred_test, probs = nb_prob_test))
}

# Function to calculate performance metrics including AUC-PR
classifier_metrics <- function(actual, preds, probs) {
  
  # Basic confusion matrix metrics
  cm <- confusionMatrix(preds, actual)
  
  # ROC and PR curves
  roc_obj <- roc(actual, probs, levels = c("gen", "human"), direction = "<")
  
  # For PR curve, we need to handle the binary labels properly
  actual_binary <- ifelse(actual == "human", 1, 0)
  pr_obj <- pr.curve(
    scores.class0 = probs,
    weights.class0 = actual_binary, 
    curve = FALSE
  )
  
  metrics <- data.frame(
    Accuracy = cm$overall["Accuracy"],
    Precision = cm$byClass["Pos Pred Value"],
    Recall = cm$byClass["Sensitivity"],
    F1_Score = cm$byClass["F1"],
    AUC_ROC = auc(roc_obj),
    AUC_PR = pr_obj$auc.integral
  )
  rownames(metrics) <- NULL
  
  return(metrics)
}

# Function to calculate statistical metrics for features
calculate_feature_stats <- function(data) {
  stats <- data.frame(
    mean_coh = mean(data$coh, na.rm = TRUE),
    median_coh = median(data$coh, na.rm = TRUE),
    var_coh = var(data$coh, na.rm = TRUE),
    mad_coh = mad(data$coh, na.rm = TRUE),
    mean_div = mean(data$div, na.rm = TRUE),
    med_div = median(data$div, na.rm = TRUE),
    var_div = var(data$div, na.rm = TRUE),
    mad_div = mad(data$div, na.rm = TRUE)
  )
  return(stats)
}


file_names <- list.files(path = "./_100_data")

results <- list()
for (fname in file_names){
  dat <- read.csv(paste0("./_100_data/", fname))
  model_name <- unique(dat$model)
  
  for (method in unique(dat$method)) {
    
    dataset_names <- unique((dat[dat$method == method, ])$dataset)
    
    for (ds_name in dataset_names) {
      cat(
        paste0("Processing dataset: ", ds_name, ", model: ", model_name, ", method: ", method, "\n")
      )
      
      split_data <- preprocess_data(
        dat[(dat$dataset == ds_name) & (dat$method == method), ]
      )
      
      if (is.null(split_data)) {
        next
      }
      # Fit models
      rf_results <- fit_random_forest(split_data)
      logit_results <- fit_logistic_reg(split_data)
      nb_results <- fit_naive_bayes(split_data)
      
      # Calculate metrics
      rf_metrics <- classifier_metrics(
        split_data$test$source,
        rf_results$preds,
        rf_results$probs[, "human"]
      )
      
      logit_metrics <- classifier_metrics(
        split_data$test$source,
        logit_results$preds,
        logit_results$probs
      )
      
      nb_metrics <- classifier_metrics(
        split_data$test$source,
        nb_results$preds,
        nb_results$probs[, "human"]
      )
      
      res <- rbind(rf_metrics, logit_metrics, nb_metrics) |>
        cbind(
          data.frame(
            classifier = c("Random Forest", "Logistic Regression", "Naive Bayes")
          )
        )
      res$dataset <- ds_name
      res$model <- model_name
      res$method <- method
      
      results[[length(results) + 1]] <- res
      
    }
  }
}

# Combine all results into a single data frame
results_df <- do.call(rbind, results)
# Save the results to a CSV file
write.csv(
  results_df,
  file = paste0("./_900_output/classify_res.csv"),
  row.names = FALSE
)
