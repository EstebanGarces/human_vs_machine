# Load required libraries
library(ggplot2)
library(readxl)
library(dplyr)
library(tidyr)
library(scales)
library(viridis)
library(stringr)
library(ggh4x)  # For facet_grid2

# Set working directory and read the data
setwd("C:/Users/ru84fuj/Desktop/INLG2025/human_vs_machine")
data <- read_excel("performance_by_classifier.xlsx")

# Filter for CS strategy only and parse hyperparameters
data_cs <- data %>%
  filter(Strategy == "CS") %>%
  # Extract alpha and k from the Hyperparameter column
  # Assuming format is "('alpha_value', 'k_value')"
  mutate(
    # Remove parentheses and quotes, then split
    param_clean = gsub("[()']", "", Hyperparameter),
    alpha = as.numeric(str_split_fixed(param_clean, ", ", 2)[,1]),
    k = as.numeric(str_split_fixed(param_clean, ", ", 2)[,2])
  )

# Define the order of models and classifiers
model_order <- c("GPT2-XL", "Qwen 2", "Mistral 3", "Deepseek", "Llama 3", "Falcon 2")
classifier_order <- c("Logistic Regression", "Random Forest", "Naive Bayes")

# Define elegant color palette for heatmap
# Soft, reversed intensity palette - darker/more intense for low values, lighter for high values
# This ensures black text is always readable
heatmap_colors <- c(
  "#8B4789",  # Deep purple for lowest values (0.5)
  "#9F6B9D",  # Medium purple
  "#B68FB1",  # Light purple
  "#CCB3C5",  # Very light purple
  "#E2D6D9",  # Pale purple-gray
  "#F5F0F0",  # Almost white
  "#FFF5F0",  # Cream white
  "#FFFAF5",  # Very light cream
  "#FFFFFF"   # Pure white for highest values (1.0)
)

# Alternative elegant color palettes (uncomment to use):
# Option 2: Blue-green palette
# heatmap_colors <- c("#2C5F7C", "#4A7A8C", "#68959C", "#86B0AC", "#A4CBBC", 
#                     "#C2E6CC", "#E0F5DC", "#F0FAEC", "#FFFFFF")

# Option 3: Warm earth tones
# heatmap_colors <- c("#8B6355", "#A17A6E", "#B79187", "#CDA8A0", "#E3BFB9", 
#                     "#F0D6D2", "#F7E6E3", "#FBF0EE", "#FFFFFF")

# Function to create a heatmap for each dataset
create_cs_heatmap <- function(dataset_name, plot_title) {
  # Filter data for the specific dataset
  plot_data <- data_cs %>%
    filter(Dataset == dataset_name) %>%
    filter(Model %in% model_order) %>%
    filter(Classifier %in% classifier_order) %>%
    # Ensure proper ordering
    mutate(
      Model = factor(Model, levels = model_order),
      Classifier = factor(Classifier, levels = classifier_order),
      k = factor(k, levels = c(1, 3, 5, 10, 15, 20, 50)),
      alpha = factor(alpha, levels = c(0.2, 0.4, 0.6, 0.8, 1.0))
    )
  
  # Create the heatmap
  p <- ggplot(plot_data, aes(x = alpha, y = k, fill = AUC_ROC)) +
    geom_tile(color = "white", size = 0.5) +
    # Add text labels for AUC_ROC values, except when it rounds to 1.00
    geom_text(aes(label = ifelse(round(AUC_ROC, 2) == 1.00, "", sprintf("%.2f", AUC_ROC))), 
              size = 3, color = "black") +
    facet_grid2(Classifier ~ Model, scales = "fixed") +
    scale_fill_gradientn(
      colors = heatmap_colors,
      limits = c(0.5, 1.0),
      breaks = seq(0.5, 1.0, 0.1),
      name = "AUC-ROC"
    ) +
    labs(
      title = paste("Contrastive Search AUC-ROC Heatmap -", plot_title),
      x = "Alpha (α)",
      y = "k"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      # Title formatting
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5, 
                                margin = ggplot2::margin(b = 20, unit = "pt")),
      
      # Facet formatting
      strip.text.x = element_text(size = 11, face = "bold", 
                                  margin = ggplot2::margin(t = 5, b = 5, unit = "pt")),
      strip.text.y = element_text(size = 11, face = "bold", 
                                  margin = ggplot2::margin(l = 5, r = 5, unit = "pt")),
      strip.background = element_rect(fill = "grey95", color = "grey80"),
      
      # Axis formatting
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      
      # Legend formatting
      legend.position = "right",
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 10),
      legend.key.height = unit(2, "cm"),
      legend.key.width = unit(0.5, "cm"),
      
      # Panel formatting
      panel.grid = element_blank(),
      panel.border = element_rect(color = "grey80", fill = NA),
      panel.spacing = unit(0.5, "lines")
    )
  
  return(p)
}

# Alternative version using viridis color scale (also colorblind-friendly)
create_cs_heatmap_viridis <- function(dataset_name, plot_title) {
  # Filter data for the specific dataset
  plot_data <- data_cs %>%
    filter(Dataset == dataset_name) %>%
    filter(Model %in% model_order) %>%
    filter(Classifier %in% classifier_order) %>%
    mutate(
      Model = factor(Model, levels = model_order),
      Classifier = factor(Classifier, levels = classifier_order),
      k = factor(k, levels = c(1, 3, 5, 10, 15, 20, 50)),
      alpha = factor(alpha, levels = c(0.2, 0.4, 0.6, 0.8, 1.0))
    )
  
  # Create the heatmap with viridis color scale
  p <- ggplot(plot_data, aes(x = alpha, y = k, fill = AUC_ROC)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = ifelse(AUC_ROC == 1.0, "", sprintf("%.2f", AUC_ROC))), 
              size = 3, color = "black") +
    facet_grid2(Classifier ~ Model, scales = "fixed") +
    scale_fill_viridis_c(
      limits = c(0.5, 1.0),
      breaks = seq(0.5, 1.0, 0.1),
      name = "AUC-ROC",
      option = "mako",  # Softer option, can also try "rocket" for purple tones
      direction = -1    # Reverse direction for intensity
    ) +
    labs(
      title = paste("Contrastive Search AUC-ROC Heatmap -", plot_title),
      x = "Alpha (α)",
      y = "k"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5, 
                                margin = ggplot2::margin(b = 20, unit = "pt")),
      strip.text.x = element_text(size = 11, face = "bold", 
                                  margin = ggplot2::margin(t = 5, b = 5, unit = "pt")),
      strip.text.y = element_text(size = 11, face = "bold", 
                                  margin = ggplot2::margin(l = 5, r = 5, unit = "pt")),
      strip.background = element_rect(fill = "grey95", color = "grey80"),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      legend.position = "right",
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 10),
      legend.key.height = unit(2, "cm"),
      legend.key.width = unit(0.5, "cm"),
      panel.grid = element_blank(),
      panel.border = element_rect(color = "grey80", fill = NA),
      panel.spacing = unit(0.5, "lines")
    )
  
  return(p)
}

# Create heatmaps for each dataset using the first color scheme
cs_plot_wikitext <- create_cs_heatmap("wikitext", "Wikitext Dataset")
cs_plot_wikinews <- create_cs_heatmap("wikinews", "Wikinews Dataset")
cs_plot_book <- create_cs_heatmap("book", "Book Dataset")

# Display the plots
print(cs_plot_wikitext)
print(cs_plot_wikinews)
print(cs_plot_book)

# Save the plots as high-quality PDF files
# PDF is vector-based so no DPI needed - they will scale perfectly at any resolution
ggsave("cs_heatmap_wikitext.pdf", cs_plot_wikitext, width = 14, height = 8, device = "pdf")
ggsave("cs_heatmap_wikinews.pdf", cs_plot_wikinews, width = 14, height = 8, device = "pdf")
ggsave("cs_heatmap_book.pdf", cs_plot_book, width = 14, height = 8, device = "pdf")

# Optional: Create versions with viridis color scale
# The viridis version uses a scientifically-designed color palette that's colorblind-friendly
# with reversed direction so lower values have more intensity
# cs_plot_wikitext_v <- create_cs_heatmap_viridis("wikitext", "Wikitext Dataset")
# cs_plot_wikinews_v <- create_cs_heatmap_viridis("wikinews", "Wikinews Dataset")
# cs_plot_book_v <- create_cs_heatmap_viridis("book", "Book Dataset")
# 
# ggsave("cs_heatmap_wikitext_viridis.pdf", cs_plot_wikitext_v, width = 14, height = 8, device = "pdf")
# ggsave("cs_heatmap_wikinews_viridis.pdf", cs_plot_wikinews_v, width = 14, height = 8, device = "pdf")
# ggsave("cs_heatmap_book_viridis.pdf", cs_plot_book_v, width = 14, height = 8, device = "pdf")

# Alternative: Save both PDF and high-res PNG versions
# If you want both formats, uncomment the lines below:
# ggsave("cs_heatmap_wikitext.png", cs_plot_wikitext, width = 14, height = 8, dpi = 600)
# ggsave("cs_heatmap_wikinews.png", cs_plot_wikinews, width = 14, height = 8, dpi = 600)
# ggsave("cs_heatmap_book.png", cs_plot_book, width = 14, height = 8, dpi = 600)

# Print summary statistics for CS data
cat("\nContrastive Search Data Summary:\n")
cat("Total CS records:", nrow(data_cs), "\n")
cat("Alpha values:", sort(unique(data_cs$alpha)), "\n")
cat("k values:", sort(unique(data_cs$k)), "\n")
cat("AUC-ROC range:", min(data_cs$AUC_ROC), "-", max(data_cs$AUC_ROC), "\n")