library(ggplot2)
library(RColorBrewer)
library(dplyr)
library(patchwork)

METHODS <- c("beam (50)", "temp (1)", "topp (0.95)")

DS_NAME_MAP <- c(
  "beam (50)" = "wikitext",
  "temp (1)" = "wikitext",
  "topp (0.95)" = "book"
)

METHODS_NAME_MAP <- c(
  "beam (50)" = "Beam Search (50)",
  "temp (1)" = "Temperature (1)",
  "topp (0.95)" = "Top-p (0.95)"
)

dat <- read.csv("./_100_data/falcon_2.csv") |> 
  dplyr::filter(method %in% METHODS)
dat$source <- ifelse(dat$source == "human", "Human", "Falcon 2") |>
  factor(levels = c("Falcon 2", "Human"))

plots <- list()
for (m in METHODS) {
  ds_name <- unname(DS_NAME_MAP[m])
  cut_ds <- dat[(dat$method == m)&(dat$dataset == ds_name),]
  
  lr <- glm(
    source ~ div + coh,
    data = cut_ds,
    family = binomial(link = "logit")
  )
  
  div_seq <- seq(min(cut_ds$div), max(cut_ds$div), length.out = 500)
  coh_seq <- seq(min(cut_ds$coh), max(cut_ds$coh), length.out = 500)
  
  grid <- expand.grid(coh = coh_seq, div = div_seq)
  grid$probs <- predict(lr, newdata = grid, type = "response")
  
  class_res <- read.csv("./_900_output/classify_res.csv")
  auc_roc <- dplyr::filter(
    class_res,
    dataset == ds_name,
    model == "Falcon 2",
    classifier == "Logistic Regression",
    method == m
  )$Accuracy
  
  p <- ggplot(data = cut_ds, mapping = aes(x = div, y = coh)) +
    geom_raster(
      data = grid,
      mapping = aes(x = div, y = coh, fill = probs),
      interpolate = TRUE,
      alpha = 0.35
    ) +
    geom_point(mapping = aes(color = source), alpha = 0.65) +
    scale_fill_distiller(palette = "RdBu") +
    scale_color_manual(values = c("#0571B0", "#CA0020")) +
    labs(
      x = "Diversity score",
      y = "Coherence score",
      fill = "Prob(Human)",
      color = NULL
    ) +
    ggtitle(
      paste0(
        unname(METHODS_NAME_MAP[m]), 
        "\nDataset: ", tools::toTitleCase(ds_name), 
        "\nAccuracy: ", round(auc_roc, 4)
      )
    ) +
    theme_minimal()
  if (length(plots) != 1){
    p <- p + theme(
      legend.position = "none",
      legend.text = element_text(size = 8),
      plot.title = element_text(size = 8)
    )
  } else {
    p <- p + theme(
      legend.position = "bottom",
      legend.text = element_text(size = 8),
      plot.title = element_text(size = 8)
    )
  }
  plots[[length(plots) + 1]] <- p
}

final_plot <- plots[[1]] + plots[[2]] + plots[[3]]

ggsave(
  filename = paste0("./_900_output/figures/class_sep.pdf"),
  plot = final_plot,
  height = 7 * 1.5,
  width = 9.9 * 2.0,
  units = "cm",
  dpi = 300
)


