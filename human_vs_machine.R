# Load required libraries
library(shiny)
library(readr)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(MASS)
library(plotly)
library(tidyr)
library(randomForest)
library(caret)
library(pROC)
library(gridExtra)
library(viridis)
library(scales)
library(RColorBrewer)
library(openxlsx)  # For Excel export
library(PRROC)     # For PR-AUC calculation

# Set seed for reproducibility
set.seed(123)

# Load the data
data <- read_csv("C:/Users/ru84fuj/Desktop/INLG2025/results_including_qtext.csv")
data <- as.data.frame(data)

# Function to calculate performance metrics including AUC-PR
calculate_extended_metrics <- function(actual, predicted, probabilities) {
  # Convert to factors if needed
  actual <- factor(actual, levels = c("Human", "Machine"))
  predicted <- factor(predicted, levels = c("Human", "Machine"))
  
  # Basic confusion matrix metrics
  cm <- confusionMatrix(predicted, actual)
  
  # ROC and PR curves
  roc_obj <- roc(actual, probabilities, levels = c("Human", "Machine"), direction = "<")
  
  # For PR curve, we need to handle the binary labels properly
  actual_binary <- ifelse(actual == "Machine", 1, 0)
  pr_obj <- pr.curve(scores.class0 = probabilities, 
                     weights.class0 = actual_binary, 
                     curve = FALSE)
  
  metrics <- data.frame(
    Accuracy = cm$overall["Accuracy"],
    Precision = cm$byClass["Pos Pred Value"],
    Recall = cm$byClass["Sensitivity"],
    F1_Score = cm$byClass["F1"],
    AUC_ROC = auc(roc_obj),
    AUC_PR = pr_obj$auc.integral
  )
  
  return(metrics)
}

# Function to calculate statistical metrics for features
calculate_feature_stats <- function(data) {
  stats <- data.frame(
    Mean_Coherence = mean(data$Coherence, na.rm = TRUE),
    Median_Coherence = median(data$Coherence, na.rm = TRUE),
    Variance_Coherence = var(data$Coherence, na.rm = TRUE),
    MAD_Coherence = mad(data$Coherence, na.rm = TRUE),
    Mean_Diversity = mean(data$Diversity, na.rm = TRUE),
    Median_Diversity = median(data$Diversity, na.rm = TRUE),
    Variance_Diversity = var(data$Diversity, na.rm = TRUE),
    MAD_Diversity = mad(data$Diversity, na.rm = TRUE)
  )
  return(stats)
}

# Initialize results dataframe
results_df <- data.frame()

# Get unique combinations
unique_datasets <- unique(data$Dataset)
unique_models <- unique(data$Model)
unique_strategies <- unique(data$Strategy)

# Progress counter
total_combinations <- 0
processed_combinations <- 0

# First count total combinations
for (dataset in unique_datasets) {
  for (model in unique_models) {
    for (strategy in unique_strategies) {
      # Get unique hyperparameters for this combination
      hyperparams <- unique(data$Hyperparameter[
        data$Dataset == dataset & 
          data$Model == model & 
          data$Strategy == strategy
      ])
      total_combinations <- total_combinations + length(hyperparams)
    }
  }
}

cat("Total combinations to process:", total_combinations, "\n\n")

# Process each combination
for (dataset in unique_datasets) {
  cat("\nProcessing dataset:", dataset, "\n")
  
  for (model in unique_models) {
    for (strategy in unique_strategies) {
      # Get unique hyperparameters for this combination
      hyperparams <- unique(data$Hyperparameter[
        data$Dataset == dataset & 
          data$Model == model & 
          data$Strategy == strategy
      ])
      
      for (hyperparam in hyperparams) {
        processed_combinations <- processed_combinations + 1
        
        # Create Dec_Method string
        dec_method <- paste0(model, " ", strategy, " (", hyperparam, ")")
        
        # Progress indicator
        if (processed_combinations %% 10 == 0) {
          cat("Progress:", processed_combinations, "/", total_combinations, 
              "- Processing:", dec_method, "\n")
        }
        
        # Filter data for current combination
        subset <- data %>% 
          filter(Model == model,
                 Dataset == dataset,
                 Strategy == strategy,
                 Hyperparameter == hyperparam) %>%
          select("id", "Model", "Strategy", "Dataset", "Hyperparameter",
                 "Method", "Length", "Reference.coherence", "Reference.diversity", 
                 "Generation.coherence", "Generation.diversity")
        
        # Skip if no data for this combination
        if (nrow(subset) == 0) {
          next
        }
        
        # Prepare human and machine data
        subset_human <- subset %>% 
          select(id, Model, Strategy, Dataset, Hyperparameter,
                 Method, Length, Reference.coherence, Reference.diversity) %>%
          rename(Coherence = Reference.coherence,
                 Diversity = Reference.diversity) %>%
          mutate(Source = "Human")
        
        subset_machine <- subset %>% 
          select(id, Model, Strategy, Dataset, Hyperparameter,
                 Method, Length, Generation.coherence, Generation.diversity) %>%
          rename(Coherence = Generation.coherence,
                 Diversity = Generation.diversity) %>%
          mutate(Source = "Machine")
        
        # Combine datasets
        df <- rbind(subset_human, subset_machine)
        df$Source <- as.factor(df$Source)
        
        # Remove rows with missing values
        df <- df %>% filter(!is.na(Coherence) & !is.na(Diversity))
        
        # Skip if insufficient data
        if (nrow(df) < 20 || length(unique(df$Source)) < 2) {
          next
        }
        
        # Try to process this combination
        tryCatch({
          # Split data into train and test sets (80-20 split)
          trainIndex <- createDataPartition(df$Source, p = 0.8, list = FALSE)
          train_data <- df[trainIndex,]
          test_data <- df[-trainIndex,]
          
          # Check if both classes are present in train and test
          if (length(unique(train_data$Source)) < 2 || length(unique(test_data$Source)) < 2) {
            next
          }
          
          # RANDOM FOREST
          rf_model <- randomForest(Source ~ Diversity + Coherence, 
                                   data = train_data, 
                                   ntree = 500,
                                   importance = TRUE)
          
          rf_pred_test <- predict(rf_model, test_data)
          rf_prob_test <- predict(rf_model, test_data, type = "prob")[, "Machine"]
          
          rf_metrics <- calculate_extended_metrics(test_data$Source, rf_pred_test, rf_prob_test)
          
          # Calculate feature statistics for the combined data
          feature_stats <- calculate_feature_stats(df)
          
          # Add to results
          rf_result <- data.frame(
            Dataset = dataset,
            Model = model,
            Strategy = strategy,
            Hyperparameter = hyperparam,
            Dec_Method = dec_method,
            Classifier = "Random Forest",
            Accuracy = rf_metrics$Accuracy,
            Precision = rf_metrics$Precision,
            Recall = rf_metrics$Recall,
            F1_Score = rf_metrics$F1_Score,
            AUC_ROC = rf_metrics$AUC_ROC,
            AUC_PR = rf_metrics$AUC_PR,
            Mean_Coherence = feature_stats$Mean_Coherence,
            Median_Coherence = feature_stats$Median_Coherence,
            Variance_Coherence = feature_stats$Variance_Coherence,
            MAD_Coherence = feature_stats$MAD_Coherence,
            Mean_Diversity = feature_stats$Mean_Diversity,
            Median_Diversity = feature_stats$Median_Diversity,
            Variance_Diversity = feature_stats$Variance_Diversity,
            MAD_Diversity = feature_stats$MAD_Diversity
          )
          
          results_df <- rbind(results_df, rf_result)
          
          # LOGISTIC REGRESSION
          lr_model <- glm(Source ~ Diversity + Coherence, 
                          family = binomial(link = "logit"), 
                          data = train_data)
          
          lr_prob_test <- predict(lr_model, test_data, type = "response")
          lr_pred_test <- ifelse(lr_prob_test > 0.5, "Machine", "Human")
          
          lr_metrics <- calculate_extended_metrics(test_data$Source, lr_pred_test, lr_prob_test)
          
          # Add to results
          lr_result <- data.frame(
            Dataset = dataset,
            Model = model,
            Strategy = strategy,
            Hyperparameter = hyperparam,
            Dec_Method = dec_method,
            Classifier = "Logistic Regression",
            Accuracy = lr_metrics$Accuracy,
            Precision = lr_metrics$Precision,
            Recall = lr_metrics$Recall,
            F1_Score = lr_metrics$F1_Score,
            AUC_ROC = lr_metrics$AUC_ROC,
            AUC_PR = lr_metrics$AUC_PR,
            Mean_Coherence = feature_stats$Mean_Coherence,
            Median_Coherence = feature_stats$Median_Coherence,
            Variance_Coherence = feature_stats$Variance_Coherence,
            MAD_Coherence = feature_stats$MAD_Coherence,
            Mean_Diversity = feature_stats$Mean_Diversity,
            Median_Diversity = feature_stats$Median_Diversity,
            Variance_Diversity = feature_stats$Variance_Diversity,
            MAD_Diversity = feature_stats$MAD_Diversity
          )
          
          results_df <- rbind(results_df, lr_result)
          
        }, error = function(e) {
          cat("Error processing:", dec_method, "- Error:", e$message, "\n")
        })
      }
    }
  }
}

# Convert numeric columns to proper format
numeric_cols <- c("Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC", "AUC_PR")
results_df[numeric_cols] <- lapply(results_df[numeric_cols], as.numeric)

# Round numeric columns to 4 decimal places
results_df[numeric_cols] <- round(results_df[numeric_cols], 4)

# Also round the feature statistics
feature_cols <- c("Mean_Coherence", "Median_Coherence", "Variance_Coherence", "MAD_Coherence",
                  "Mean_Diversity", "Median_Diversity", "Variance_Diversity", "MAD_Diversity")
results_df[feature_cols] <- round(results_df[feature_cols], 4)

# Create human generation statistics
cat("\nCalculating human generation statistics...\n")

human_stats <- data.frame()

for (dataset in unique_datasets) {
  # Get all human data for this dataset
  human_data <- data %>%
    filter(Dataset == dataset) %>%
    select(Reference.coherence, Reference.diversity) %>%
    rename(Coherence = Reference.coherence,
           Diversity = Reference.diversity) %>%
    filter(!is.na(Coherence) & !is.na(Diversity))
  
  if (nrow(human_data) > 0) {
    stats <- data.frame(
      Dataset = dataset,
      Method = "Human",
      Mean_Coherence = mean(human_data$Coherence, na.rm = TRUE),
      Median_Coherence = median(human_data$Coherence, na.rm = TRUE),
      Variance_Coherence = var(human_data$Coherence, na.rm = TRUE),
      MAD_Coherence = mad(human_data$Coherence, na.rm = TRUE),
      Mean_Diversity = mean(human_data$Diversity, na.rm = TRUE),
      Median_Diversity = median(human_data$Diversity, na.rm = TRUE),
      Variance_Diversity = var(human_data$Diversity, na.rm = TRUE),
      MAD_Diversity = mad(human_data$Diversity, na.rm = TRUE),
      N_Samples = nrow(human_data)
    )
    
    human_stats <- rbind(human_stats, stats)
  }
}

# Round human statistics
human_stats[feature_cols] <- round(human_stats[feature_cols], 4)

# Save to Excel
wb <- createWorkbook()

# Add main results sheet
addWorksheet(wb, "Classification_Results")
writeData(wb, "Classification_Results", results_df)

# Add human generation statistics sheet
addWorksheet(wb, "Human_Generation_Stats")
writeData(wb, "Human_Generation_Stats", human_stats)

# Add summary by Dataset
summary_dataset <- results_df %>%
  group_by(Dataset, Classifier) %>%
  summarise(
    Avg_Accuracy = mean(Accuracy, na.rm = TRUE),
    Avg_Precision = mean(Precision, na.rm = TRUE),
    Avg_Recall = mean(Recall, na.rm = TRUE),
    Avg_F1_Score = mean(F1_Score, na.rm = TRUE),
    Avg_AUC_ROC = mean(AUC_ROC, na.rm = TRUE),
    Avg_AUC_PR = mean(AUC_PR, na.rm = TRUE),
    N_Experiments = n()
  ) %>%
  arrange(Dataset, Classifier)

addWorksheet(wb, "Summary_by_Dataset")
writeData(wb, "Summary_by_Dataset", summary_dataset)

# Add summary by Model
summary_model <- results_df %>%
  group_by(Model, Classifier) %>%
  summarise(
    Avg_Accuracy = mean(Accuracy, na.rm = TRUE),
    Avg_Precision = mean(Precision, na.rm = TRUE),
    Avg_Recall = mean(Recall, na.rm = TRUE),
    Avg_F1_Score = mean(F1_Score, na.rm = TRUE),
    Avg_AUC_ROC = mean(AUC_ROC, na.rm = TRUE),
    Avg_AUC_PR = mean(AUC_PR, na.rm = TRUE),
    N_Experiments = n()
  ) %>%
  arrange(Model, Classifier)

addWorksheet(wb, "Summary_by_Model")
writeData(wb, "Summary_by_Model", summary_model)

# Add summary by Strategy
summary_strategy <- results_df %>%
  group_by(Strategy, Classifier) %>%
  summarise(
    Avg_Accuracy = mean(Accuracy, na.rm = TRUE),
    Avg_Precision = mean(Precision, na.rm = TRUE),
    Avg_Recall = mean(Recall, na.rm = TRUE),
    Avg_F1_Score = mean(F1_Score, na.rm = TRUE),
    Avg_AUC_ROC = mean(AUC_ROC, na.rm = TRUE),
    Avg_AUC_PR = mean(AUC_PR, na.rm = TRUE),
    N_Experiments = n()
  ) %>%
  arrange(Strategy, Classifier)

addWorksheet(wb, "Summary_by_Strategy")
writeData(wb, "Summary_by_Strategy", summary_strategy)

# Add best performers sheet
best_performers <- results_df %>%
  group_by(Classifier) %>%
  slice_max(order_by = AUC_ROC, n = 10) %>%
  arrange(Classifier, desc(AUC_ROC))

addWorksheet(wb, "Top_10_by_AUC_ROC")
writeData(wb, "Top_10_by_AUC_ROC", best_performers)

# Save the workbook
saveWorkbook(wb, "C:/Users/ru84fuj/Desktop/INLG2025/classification_results_all_methods.xlsx", overwrite = TRUE)

# Print summary statistics
cat("\n\n=== ANALYSIS COMPLETE ===\n")
cat("Total experiments processed:", nrow(results_df), "\n")
cat("Results saved to: classification_results_all_methods.xlsx\n\n")

# Print human generation statistics
cat("Human Generation Statistics:\n")
print(human_stats)
cat("\n")

# Print top performers
cat("Top 5 configurations by AUC-ROC (Random Forest):\n")
top_rf <- results_df %>% 
  filter(Classifier == "Random Forest") %>%
  arrange(desc(AUC_ROC)) %>%
  head(5)
print(top_rf[, c("Dec_Method", "AUC_ROC", "F1_Score")])

cat("\n\nTop 5 configurations by AUC-ROC (Logistic Regression):\n")
top_lr <- results_df %>% 
  filter(Classifier == "Logistic Regression") %>%
  arrange(desc(AUC_ROC)) %>%
  head(5)
print(top_lr[, c("Dec_Method", "AUC_ROC", "F1_Score")])

# Create visualization of results
library(ggplot2)

# Plot average performance by model
avg_by_model <- results_df %>%
  group_by(Model, Classifier) %>%
  summarise(
    Avg_AUC_ROC = mean(AUC_ROC, na.rm = TRUE),
    SE = sd(AUC_ROC, na.rm = TRUE) / sqrt(n())
  )

p_model <- ggplot(avg_by_model, aes(x = reorder(Model, Avg_AUC_ROC), 
                                    y = Avg_AUC_ROC, 
                                    fill = Classifier)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = Avg_AUC_ROC - SE, ymax = Avg_AUC_ROC + SE),
                position = position_dodge(width = 0.8), width = 0.25) +
  coord_flip() +
  labs(title = "Average AUC-ROC by Model",
       x = "Model",
       y = "Average AUC-ROC") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")

ggsave("C:/Users/ru84fuj/Desktop/INLG2025/avg_performance_by_model.png", p_model, width = 10, height = 6, dpi = 300)

cat("\n\nVisualization saved as: avg_performance_by_model.png\n")

# Create visualization of human feature distributions by dataset
human_data_all <- data.frame()

for (dataset in unique_datasets) {
  temp_data <- data %>%
    filter(Dataset == dataset) %>%
    select(Dataset, Reference.coherence, Reference.diversity) %>%
    rename(Coherence = Reference.coherence,
           Diversity = Reference.diversity) %>%
    filter(!is.na(Coherence) & !is.na(Diversity))
  
  human_data_all <- rbind(human_data_all, temp_data)
}

# Create violin plot for human feature distributions
p_human <- human_data_all %>%
  pivot_longer(cols = c(Coherence, Diversity), 
               names_to = "Feature", 
               values_to = "Value") %>%
  ggplot(aes(x = Dataset, y = Value, fill = Dataset)) +
  geom_violin(alpha = 0.7, scale = "width", trim = FALSE) +
  geom_boxplot(width = 0.2, alpha = 0.9, outlier.shape = 21, 
               outlier.fill = "white", outlier.size = 2) +
  facet_wrap(~Feature, scales = "free_y", ncol = 2) +
  labs(title = "Human Generation Feature Distributions by Dataset",
       subtitle = "Violin plots with embedded boxplots",
       x = "Dataset",
       y = "Feature Value") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")

ggsave("C:/Users/ru84fuj/Desktop/INLG2025/human_feature_distributions.png", p_human, width = 12, height = 6, dpi = 300)

cat("Human feature distribution visualization saved as: human_feature_distributions.png\n")