library(ggplot2)
library(ggh4x)

DATASET_NAME_MAP <- c(
  "book" = "Book",
  "wikinews" = "Wikinews",
  "wikitext" = "Wikitext"
)

METHOD_NAME_MAP <- c(
  "beam" = "Beam",
  "temp" = "Temp",
  "topk" = "Top-k",
  "topp" = "Top-p",
  "CS" = "CS"
)
FONT_SIZE <- 10

process_method_name <- function(s) {
  
  s <- gsub("[\\(\\)'\",]", "", s) |> #Remove all brackets and quotes
    strsplit(" ") # Split by space
  
  s <- s[[1]]
  
  if (length(s) < 3){
    s <- c(s, NA)
  }
  
  return(s)
}

res <- read.csv("./_900_output/classify_res.csv")
clean_methods <- lapply(res$method, process_method_name)
clean_methods <- do.call(rbind, clean_methods)
colnames(clean_methods) <- c("method_name", "h1", "h2")

res <- cbind(res, clean_methods)
res$h1 <- as.numeric(res$h1)
res$h2 <- as.numeric(res$h2)

res$dataset <- unname(DATASET_NAME_MAP[res$dataset])
res$method_name <- unname(METHOD_NAME_MAP[res$method_name])
res$model <- factor(res$model, levels =  c("GPT2-XL", "Qwen 2", "Mistral 3", "Deepseek", "LLama 3", "Falcon 2"))

nocs <- res[(res$method_name != "CS"),]
for (i in seq_along(unique(nocs$dataset))) {
  ds_name <- unique(nocs$dataset)[i]
  nocs_ds <- nocs[nocs$dataset == ds_name,]
  
  p <- ggplot(
    data = nocs_ds,
    mapping = aes(x = h1, y = AUC_ROC, colour = classifier)
  ) +
    geom_line(linewidth = 0.85) +
    geom_point(size = 1.15) +
    ylab("ROC AUC") +
    facet_grid2(method_name ~ model, scales = "free_x", independent = "x") +
    ggtitle(ds_name) +
    theme_bw()
  
  if (i == 1){
    p <- p +
      theme(legend.position = "none", text = element_text(size = FONT_SIZE)) +
      xlab("")
  }  else if (i != length(unique(nocs$dataset))) {
    p <- p + 
      theme(
        legend.position = "none",
        strip.background.x = element_blank(),
        strip.text.x.top = element_blank(),
        text = element_text(size = FONT_SIZE)
      ) +
      xlab("")
  }  else {
    p <- p + theme(
      legend.position = "bottom",
      strip.background.x = element_blank(),
      strip.text.x.top = element_blank(),
      text = element_text(size = FONT_SIZE),
      legend.title = element_blank()
    ) +
      xlab("Hyperparameter")
  }
  
  ggsave(
    filename = paste0("./_900_output/figures/", ds_name, ".pdf"),
    plot = p,
    height = 7 * 1.5,
    width = 9.9 * 2.0,
    units = "cm",
    dpi = 300
  )
  
}

#cs_res <- res[(res$dataset == "book")&(res$method_name == "CS"),]
#cs_res$h1 <- as.factor(cs_res$h1)
#cs_res$h2 <- as.factor(cs_res$h2)

#ggplot(
#  data = cs_res,
#  mapping = aes(x = h1, y = h2, fill = AUC_ROC)
#) +
#  geom_tile() +
#  geom_text(aes(label = round(AUC_ROC, 2)), color = "black", size = 3) +
#  scale_fill_distiller(palette = "RdPu") +
#  facet_grid(model ~ classifier, scales = "free") +
#  ggtitle("Book dataset")
  

