# Load required libraries
library(ggplot2)
library(readxl)
library(dplyr)
library(tidyr)
library(scales)
library(viridis)
library(ggh4x)     # For facet_grid2 with independent scales
library(patchwork) # For combining plots
library(grid)      # For unit()

# If ggh4x is not installed, run:
# install.packages("ggh4x")
# If patchwork is not installed, run:
# install.packages("patchwork")

# Set working directory and read the data
setwd("C:/Users/ru84fuj/Desktop/INLG2025/human_vs_machine")
data <- readxl::read_excel("performance_by_classifier.xlsx")

# Filter out CS strategy and convert Hyperparameter to numeric
data_filtered <- data %>%
  dplyr::filter(Strategy != "CS") %>%
  dplyr::mutate(Hyperparameter = as.numeric(Hyperparameter))

# Define the order of models and strategies for consistent plotting
model_order <- c("GPT2-XL", "Qwen 2", "Mistral 3", "Deepseek", "Llama 3", "Falcon 2")
strategy_order <- c("beam", "topk", "topp")
strategy_labels <- c(
  "beam" = "Beam",
  #"temp" = "Temperature",
  "topk" = "Top-k",
  "topp" = "Top-p"
)

# Define elegant color palette for classifiers
classifier_colors <- c(
  "Logistic Regression" = "#2E86AB",
  "Random Forest" = "#A23B72",
  "Naive Bayes" = "#F18F01"
)

# Function to create a plot for each dataset with control over labels
create_auc_plot <- function(dataset_name, plot_title,
                            show_x_label = TRUE,
                            show_legend = TRUE) {
  
  # Filter data for the specific dataset
  plot_data <- data_filtered %>%
    dplyr::filter(Dataset == dataset_name) %>%
    dplyr::filter(Model %in% model_order) %>%
    dplyr::filter(Strategy %in% strategy_order) %>%
    dplyr::mutate(
      Model = factor(Model, levels = model_order),
      Strategy = factor(Strategy, levels = strategy_order, labels = strategy_labels)
    ) %>%
    dplyr::arrange(Model, Strategy, Classifier, Hyperparameter)
  
  # Create the plot
  p <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
      x = Hyperparameter,
      y = AUC_ROC,
      color = Classifier,
      group = Classifier
    )
  ) +
    ggplot2::geom_line(linewidth = 1.2, alpha = 0.8) +
    ggplot2::geom_point(size = 2, alpha = 0.7) +
    ggh4x::facet_grid2(Strategy ~ Model, scales = "free_x", independent = "x") +
    ggplot2::scale_color_manual(values = classifier_colors) +
    ggplot2::scale_y_continuous(
      limits = c(0.5, 1),
      breaks = seq(0.5, 1, 0.1)
    ) +
    ggplot2::labs(
      title = paste("AUC-ROC Evolution by Decoding Strategy -", plot_title),
      x = if (show_x_label) "Hyperparameter Value" else "",
      y = "AUC-ROC",
      color = "Classifier"
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      # Title formatting
      plot.title = ggplot2::element_text(
        size = 16,
        face = "bold",
        hjust = 0.5,
        margin = ggplot2::margin(b = 20, unit = "pt")
      ),
      
      # Facet formatting
      strip.text.x = ggplot2::element_text(
        size = 11,
        face = "bold",
        margin = ggplot2::margin(t = 5, b = 5, unit = "pt")
      ),
      strip.text.y = ggplot2::element_text(
        size = 11,
        face = "bold",
        margin = ggplot2::margin(l = 5, r = 5, unit = "pt")
      ),
      strip.background = ggplot2::element_rect(fill = "grey95", color = "grey80"),
      
      # Axis formatting
      axis.title = ggplot2::element_text(size = 12, face = "bold"),
      axis.title.x = if (!show_x_label) {
        ggplot2::element_blank()
      } else {
        ggplot2::element_text(size = 12, face = "bold")
      },
      axis.text = ggplot2::element_text(size = 10),
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
      
      # Legend formatting
      legend.position = if (show_legend) "bottom" else "none",
      legend.title = ggplot2::element_text(size = 12, face = "bold"),
      legend.text = ggplot2::element_text(size = 11),
      legend.key.width = grid::unit(1.5, "cm"),
      
      # Panel formatting
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major = ggplot2::element_line(color = "grey90"),
      panel.border = ggplot2::element_rect(color = "grey80", fill = NA),
      panel.spacing = grid::unit(0.5, "lines"),
      
      # Reduce bottom margin if x-label is not shown
      plot.margin = if (!show_x_label) {
        ggplot2::margin(t = 5, r = 5, b = 5, l = 5, unit = "pt")
      } else {
        ggplot2::margin()
      }
    )
  
  return(p)
}

# Create plots for each dataset
# Book and Wikinews without x-label and legend
plot_book <- create_auc_plot(
  dataset_name = "book",
  plot_title = "Book Dataset",
  show_x_label = FALSE,
  show_legend = FALSE
)

plot_wikinews <- create_auc_plot(
  dataset_name = "wikinews",
  plot_title = "Wikinews Dataset",
  show_x_label = FALSE,
  show_legend = FALSE
)

# Wikitext with both labels (at the bottom)
plot_wikitext <- create_auc_plot(
  dataset_name = "wikitext",
  plot_title = "Wikitext Dataset",
  show_x_label = TRUE,
  show_legend = TRUE
)

# Display the plots
print(plot_book)
print(plot_wikinews)
print(plot_wikitext)

# Save the plots as high-quality PDF files
# PDF is a vector format, so no dpi parameter is needed
ggplot2::ggsave("auc_roc_book.pdf", plot_book, width = 16, height = 10)
ggplot2::ggsave("auc_roc_wikinews.pdf", plot_wikinews, width = 16, height = 10)
ggplot2::ggsave("auc_roc_wikitext.pdf", plot_wikitext, width = 16, height = 10)

# Create a vertically stacked combined plot
combined_vertical <- (plot_book / plot_wikinews / plot_wikitext) +
  patchwork::plot_layout(guides = "collect") &
  ggplot2::theme(legend.position = "bottom")

# Save the combined plot as PDF
ggplot2::ggsave("auc_roc_stacked_vertical.pdf", combined_vertical, width = 16, height = 24)

# Print summary statistics for verification
cat("\nData Summary:\n")
cat("Total records after filtering CS:", nrow(data_filtered), "\n")
cat("Datasets:", paste(unique(data_filtered$Dataset), collapse = ", "), "\n")
cat("Models:", paste(unique(data_filtered$Model), collapse = ", "), "\n")
cat("Strategies:", paste(unique(data_filtered$Strategy), collapse = ", "), "\n")
cat("Classifiers:", paste(unique(data_filtered$Classifier), collapse = ", "), "\n")