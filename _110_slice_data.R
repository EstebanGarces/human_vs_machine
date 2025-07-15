# Read the data and subset it to the relevant columns
dat <- read.csv("./_000_raw_data/results_including_qtext.csv")
sel_cols <- c(
  "id", "Model", "Strategy",
  "Dataset", "Hyperparameter", "Method",
  "Length", "Reference.coherence", "Reference.diversity", 
  "Generation.coherence", "Generation.diversity"
)
dat <- dat[sel_cols]

# Format column names
new_names <- colnames(dat) |>
  tolower()

new_names <- gsub("reference.", "human_", new_names)
new_names <- gsub("generation.", "gen_", new_names)
new_names <- gsub("hyperparameter", "hp", new_names)
new_names <- gsub("coherence", "coh", new_names)
new_names <- gsub("diversity", "div", new_names)
colnames(dat) <- new_names

# Write the per-model CSV files
unique_models <- unique(dat$model)
for (model in unique_models) {
  
  model_data <- dat[dat$model == model, ]
  cases <- nrow(model_data)
  
  # Split and stack by source
  hum_data <- model_data[,-c(10, 11)]
  gen_data <- model_data[,-c(8, 9)]
  
  colnames(hum_data)[c(8, 9)] <- c("coh", "div")
  colnames(gen_data)[c(8, 9)] <- c("coh", "div")
  
  model_data <- rbind(hum_data, gen_data)
  model_data$source <- c(
    rep("human", cases),
    rep("gen", cases)
  )
  
  file_name <- gsub(" ", "_", model) |>
    tolower()

  write.csv(
    model_data,
    file = paste0("./_100_data/", file_name, ".csv"),
    row.names = FALSE
  )
}