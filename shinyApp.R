# Load required libraries
# Check and install packages if needed
packages <- c("shiny", "shinydashboard", "ggplot2", "readxl", 
              "dplyr", "tidyr", "plotly", "DT", "stringr")

# Suppress package startup messages
suppressPackageStartupMessages({
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      install.packages(pkg)
      library(pkg, character.only = TRUE, quietly = TRUE)
    }
  }
})

# Define UI
ui <- dashboardPage(
  dashboardHeader(title = "Machine Text Detection Analysis"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Visualization", tabName = "viz", icon = icon("chart-line")),
      menuItem("Detection Analysis", tabName = "insights", icon = icon("lightbulb")),
      menuItem("Data Table", tabName = "data", icon = icon("table"))
    ),
    
    hr(),
    
    # File input
    fileInput("file", "Upload Excel File",
              accept = c(".xlsx", ".xls")),
    
    # Add help text
    helpText("File should contain columns: Dataset, Model, Strategy, Classifier, Hyperparameter, AUC_ROC"),
    helpText("For CS strategy, Hyperparameter format: ('alpha', 'k'), e.g., ('0.2', '1')"),
    helpText("Note: Lower AUC-ROC = Less detectable (better evasion)"),
    
    # Download sample data button
    downloadButton("download_sample", "Download Sample Format"),
    
    hr(),
    
    # Filters (will be populated after file upload)
    uiOutput("dataset_filter"),
    uiOutput("strategy_filter"),
    uiOutput("model_filter"),
    uiOutput("classifier_filter"),
    
    hr(),
    
    # Human baseline input
    h4("Human Baseline (optional)"),
    numericInput("human_baseline", "Human AUC-ROC:", 
                 value = NULL, min = 0.5, max = 1, step = 0.01),
    helpText("Strategies below this value are harder to detect than human text"),
    
    hr(),
    
    # Download buttons
    downloadButton("download_plot", "Download Plot")
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .content-wrapper, .right-side {
          background-color: #f4f4f4;
        }
        .box {
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        /* Fix table overflow */
        .dataTables_wrapper {
          overflow-x: auto;
        }
        table.dataTable {
          width: 100% !important;
        }
        .box-body {
          overflow-x: auto;
        }
      "))
    ),
    
    tabItems(
      # Visualization Tab
      tabItem(tabName = "viz",
              fluidRow(
                box(
                  width = 12,
                  title = "Detection Performance Visualization",
                  status = "primary",
                  solidHeader = TRUE,
                  plotlyOutput("main_plot", height = "600px")
                )
              ),
              fluidRow(
                box(
                  width = 12,
                  title = "Quick Stats",
                  status = "info",
                  solidHeader = TRUE,
                  verbatimTextOutput("quick_stats")
                )
              )
      ),
      
      # Detection Analysis Tab
      tabItem(tabName = "insights",
              fluidRow(
                box(
                  width = 6,
                  title = "Least Detectable Strategies (Low AUC-ROC)",
                  status = "success",
                  solidHeader = TRUE,
                  div(style = "overflow-x: auto;",
                      tableOutput("least_detectable")
                  )
                ),
                box(
                  width = 6,
                  title = "Most Detectable Strategies (High AUC-ROC)",
                  status = "danger",
                  solidHeader = TRUE,
                  div(style = "overflow-x: auto;",
                      tableOutput("most_detectable")
                  )
                )
              ),
              fluidRow(
                box(
                  width = 6,
                  title = "Detection Summary by Classifier",
                  status = "warning",
                  solidHeader = TRUE,
                  plotlyOutput("classifier_summary", height = "400px")
                ),
                box(
                  width = 6,
                  title = "Detection Summary by Strategy",
                  status = "info",
                  solidHeader = TRUE,
                  plotlyOutput("strategy_summary", height = "400px")
                )
              ),
              fluidRow(
                box(
                  width = 12,
                  title = "Key Insights",
                  status = "primary",
                  solidHeader = TRUE,
                  uiOutput("insights_text")
                )
              )
      ),
      
      # Data Table Tab
      tabItem(tabName = "data",
              fluidRow(
                box(
                  width = 12,
                  title = "Raw Data",
                  status = "primary",
                  solidHeader = TRUE,
                  div(style = "overflow-x: auto;",
                      DTOutput("data_table")
                  )
                )
              )
      )
    )
  )
)

# Define Server
server <- function(input, output, session) {
  
  # Reactive values
  values <- reactiveValues(
    data = NULL,
    filtered_data = NULL
  )
  
  # Download sample data
  output$download_sample <- downloadHandler(
    filename = function() {
      "sample_classifier_data.csv"
    },
    content = function(file) {
      # Create sample data
      sample_data <- data.frame(
        Dataset = c(rep("wikitext", 6), rep("wikinews", 6)),
        Model = rep(c("GPT2-XL", "GPT2-XL", "Qwen 2", "Qwen 2", "Llama 3", "Llama 3"), 2),
        Strategy = rep(c("beam", "temp", "topk", "topp", "CS", "CS"), 2),
        Classifier = rep(c("Logistic Regression", "Random Forest"), 6),
        Hyperparameter = c("5", "0.7", "50", "0.9", "('0.2', '1')", "('0.4', '3')",
                           "10", "0.8", "40", "0.95", "('0.6', '5')", "('0.8', '10')"),
        AUC_ROC = runif(12, 0.6, 0.95)
      )
      write.csv(sample_data, file, row.names = FALSE)
    }
  )
  
  # Load data
  observeEvent(input$file, {
    req(input$file)
    
    tryCatch({
      # Read the Excel file
      data <- read_excel(input$file$datapath)
      
      # Check if required columns exist
      required_cols <- c("Dataset", "Model", "Strategy", "Classifier", "Hyperparameter", "AUC_ROC")
      if (!all(required_cols %in% names(data))) {
        stop("Missing required columns. File must contain: ", paste(required_cols, collapse = ", "))
      }
      
      # Convert AUC_ROC to numeric if it isn't already
      data$AUC_ROC <- as.numeric(data$AUC_ROC)
      
      # Process hyperparameters based on strategy
      # Keep original Hyperparameter column and add parsed columns
      if ("CS" %in% unique(data$Strategy)) {
        # Process CS rows
        cs_rows <- data$Strategy == "CS"
        
        # Initialize new columns
        data$alpha <- NA
        data$k <- NA
        data$Hyperparameter_numeric <- NA
        
        # Parse CS hyperparameters
        if (any(cs_rows)) {
          cs_params <- data$Hyperparameter[cs_rows]
          # Remove parentheses, quotes, and spaces
          cs_params_clean <- gsub("[()'\"]", "", cs_params)
          cs_params_clean <- gsub("\\s+", "", cs_params_clean)
          
          # Split by comma
          param_split <- strsplit(cs_params_clean, ",")
          
          # Extract alpha and k
          data$alpha[cs_rows] <- sapply(param_split, function(x) as.numeric(x[1]))
          data$k[cs_rows] <- sapply(param_split, function(x) as.numeric(x[2]))
        }
        
        # Parse non-CS hyperparameters
        non_cs_rows <- data$Strategy != "CS"
        if (any(non_cs_rows)) {
          data$Hyperparameter_numeric[non_cs_rows] <- as.numeric(data$Hyperparameter[non_cs_rows])
        }
      } else {
        # All non-CS data
        data$Hyperparameter_numeric <- as.numeric(data$Hyperparameter)
        data$alpha <- NA
        data$k <- NA
      }
      
      # Store the processed data
      values$data <- data
      values$filtered_data <- data
      
      # Show success message with data summary
      showNotification(
        paste("Data loaded successfully!", 
              "Rows:", nrow(data),
              "| Strategies:", paste(unique(data$Strategy), collapse = ", ")),
        type = "message",  # Fixed: use "message" instead of "success"
        duration = 5
      )
      
    }, error = function(e) {
      showNotification(
        paste("Error loading file:", e$message), 
        type = "error",
        duration = 10
      )
      print(paste("Detailed error:", e))  # For debugging
    })
  })
  
  # Dynamic filters
  output$dataset_filter <- renderUI({
    req(values$data)
    selectInput("dataset", "Dataset:",
                choices = c("All", unique(values$data$Dataset)),
                selected = "All")
  })
  
  output$strategy_filter <- renderUI({
    req(values$data)
    selectInput("strategy", "Strategy:",
                choices = c("All", unique(values$data$Strategy)),
                selected = "All")
  })
  
  output$model_filter <- renderUI({
    req(values$data)
    selectInput("model", "Model:",
                choices = c("All", unique(values$data$Model)),
                selected = "All",
                multiple = TRUE)
  })
  
  output$classifier_filter <- renderUI({
    req(values$data)
    selectInput("classifier", "Classifier:",
                choices = c("All", unique(values$data$Classifier)),
                selected = "All",
                multiple = TRUE)
  })
  
  # Filter data reactively
  observe({
    req(values$data)
    
    filtered <- values$data
    
    if (!is.null(input$dataset) && input$dataset != "All") {
      filtered <- filtered %>% filter(Dataset == input$dataset)
    }
    
    if (!is.null(input$strategy) && input$strategy != "All") {
      filtered <- filtered %>% filter(Strategy == input$strategy)
    }
    
    if (!is.null(input$model) && !"All" %in% input$model && length(input$model) > 0) {
      filtered <- filtered %>% filter(Model %in% input$model)
    }
    
    if (!is.null(input$classifier) && !"All" %in% input$classifier && length(input$classifier) > 0) {
      filtered <- filtered %>% filter(Classifier %in% input$classifier)
    }
    
    values$filtered_data <- filtered
  })
  
  # Main plot
  output$main_plot <- renderPlotly({
    req(values$filtered_data)
    
    # Check if we have data to plot
    if (nrow(values$filtered_data) == 0) {
      return(plot_ly() %>% layout(title = "No data to display"))
    }
    
    # Define color palette
    classifier_colors <- c(
      "Logistic Regression" = "#2E86AB",
      "Random Forest" = "#A23B72",
      "Naive Bayes" = "#F18F01"
    )
    
    # Fixed: Check if strategy is not NULL before comparison
    if (!is.null(input$strategy) && (input$strategy == "CS" || (input$strategy == "All" && "CS" %in% values$filtered_data$Strategy))) {
      # Heatmap for CS strategy
      cs_data <- values$filtered_data %>%
        filter(Strategy == "CS") %>%
        filter(!is.na(alpha) & !is.na(k)) %>%
        group_by(Model, Classifier, alpha, k) %>%
        summarise(AUC_ROC = mean(AUC_ROC, na.rm = TRUE), .groups = "drop")
      
      if (nrow(cs_data) > 0) {
        p <- plot_ly(
          data = cs_data,
          x = ~alpha,
          y = ~k,
          z = ~AUC_ROC,
          type = "heatmap",
          colorscale = list(
            c(0, "#FF6B6B"),  # Red for low AUC (good - less detectable)
            c(0.5, "#FFF3B2"),  # Yellow for medium
            c(1, "#4ECDC4")  # Teal for high AUC (bad - more detectable)
          ),
          zmin = 0.5,
          zmax = 1.0,
          text = ~paste("Model:", Model, "<br>",
                        "Classifier:", Classifier, "<br>",
                        "Alpha:", alpha, "<br>",
                        "k:", k, "<br>",
                        "AUC-ROC:", round(AUC_ROC, 3), "<br>",
                        "Detection:", ifelse(AUC_ROC < 0.7, "Low", 
                                             ifelse(AUC_ROC < 0.85, "Medium", "High"))),
          hovertemplate = "%{text}<extra></extra>"
        ) %>%
          layout(
            title = "Contrastive Search Detection Heatmap<br><sub>Lower AUC-ROC = Less Detectable (Better)</sub>",
            xaxis = list(title = "Alpha (α)"),
            yaxis = list(title = "k"),
            margin = list(l = 100)
          )
        
        # Add faceting if multiple models/classifiers
        if (length(unique(cs_data$Model)) > 1 || length(unique(cs_data$Classifier)) > 1) {
          p <- cs_data %>%
            group_by(Model, Classifier) %>%
            group_map(~ {
              plot_ly(
                x = ~.x$alpha,
                y = ~.x$k,
                z = ~.x$AUC_ROC,
                type = "heatmap",
                name = paste(.y$Model, "-", .y$Classifier),
                colorscale = list(
                  c(0, "#FF6B6B"),
                  c(0.5, "#FFF3B2"),
                  c(1, "#4ECDC4")
                ),
                zmin = 0.5,
                zmax = 1.0
              )
            }) %>%
            subplot(nrows = length(unique(cs_data$Classifier)), 
                    shareX = TRUE, shareY = TRUE)
        }
      } else {
        p <- plot_ly() %>% layout(title = "No CS data to display")
      }
      
    } else {
      # Line plot for other strategies
      plot_data <- values$filtered_data %>%
        filter(Strategy != "CS") %>%
        filter(!is.na(Hyperparameter_numeric)) %>%
        arrange(Model, Classifier, Strategy, Hyperparameter_numeric)
      
      if (nrow(plot_data) > 0) {
        p <- plot_ly()
        
        # Add lines for each model-classifier-strategy combination
        for (cl in unique(plot_data$Classifier)) {
          cl_data <- plot_data %>% filter(Classifier == cl)
          
          p <- p %>% add_trace(
            data = cl_data,
            x = ~Hyperparameter_numeric,
            y = ~AUC_ROC,
            color = ~Classifier,
            colors = classifier_colors,
            type = 'scatter',
            mode = 'lines+markers',
            name = cl,
            text = ~paste("Model:", Model, "<br>",
                          "Strategy:", Strategy, "<br>",
                          "Hyperparameter:", Hyperparameter_numeric, "<br>",
                          "AUC-ROC:", round(AUC_ROC, 3), "<br>",
                          "Detection:", ifelse(AUC_ROC < 0.7, "Low", 
                                               ifelse(AUC_ROC < 0.85, "Medium", "High"))),
            hovertemplate = "%{text}<extra></extra>",
            line = list(width = 2),
            marker = list(size = 6)
          )
        }
        
        # Add human baseline if provided
        if (!is.null(input$human_baseline) && !is.na(input$human_baseline)) {
          p <- p %>% add_trace(
            y = c(input$human_baseline, input$human_baseline),
            x = c(min(plot_data$Hyperparameter_numeric, na.rm = TRUE), 
                  max(plot_data$Hyperparameter_numeric, na.rm = TRUE)),
            type = 'scatter',
            mode = 'lines',
            name = 'Human Baseline',
            line = list(color = 'red', dash = 'dash', width = 2)
          )
        }
        
        # Add shaded regions for detection levels
        p <- p %>% 
          layout(
            title = "Detection Performance by Hyperparameter<br><sub>Lower AUC-ROC = Less Detectable (Better)</sub>",
            xaxis = list(title = "Hyperparameter Value"),
            yaxis = list(title = "AUC-ROC", range = c(0.5, 1.05)),
            hovermode = 'closest',
            shapes = list(
              list(type = "rect", fillcolor = "green", opacity = 0.1,
                   x0 = 0, x1 = 1, xref = "paper",
                   y0 = 0.5, y1 = 0.7, yref = "y",
                   line = list(width = 0)),
              list(type = "rect", fillcolor = "orange", opacity = 0.1,
                   x0 = 0, x1 = 1, xref = "paper",
                   y0 = 0.7, y1 = 0.85, yref = "y",
                   line = list(width = 0)),
              list(type = "rect", fillcolor = "red", opacity = 0.1,
                   x0 = 0, x1 = 1, xref = "paper",
                   y0 = 0.85, y1 = 1.05, yref = "y",
                   line = list(width = 0))
            ),
            annotations = list(
              list(x = 0.02, y = 0.6, text = "Low Detection", 
                   showarrow = FALSE, xref = "paper", yref = "y"),
              list(x = 0.02, y = 0.775, text = "Medium Detection", 
                   showarrow = FALSE, xref = "paper", yref = "y"),
              list(x = 0.02, y = 0.925, text = "High Detection", 
                   showarrow = FALSE, xref = "paper", yref = "y")
            )
          )
      } else {
        p <- plot_ly() %>% layout(title = "No data to display")
      }
    }
    
    p
  })
  
  # Quick stats
  output$quick_stats <- renderPrint({
    req(values$filtered_data)
    
    cat("Dataset Summary:\n")
    cat("----------------\n")
    cat("Total observations:", nrow(values$filtered_data), "\n")
    cat("Datasets:", paste(unique(values$filtered_data$Dataset), collapse = ", "), "\n")
    cat("Models:", paste(unique(values$filtered_data$Model), collapse = ", "), "\n")
    cat("Strategies:", paste(unique(values$filtered_data$Strategy), collapse = ", "), "\n")
    cat("\nDetection Metrics:\n")
    cat("AUC-ROC range:", 
        round(min(values$filtered_data$AUC_ROC, na.rm = TRUE), 3), "-", 
        round(max(values$filtered_data$AUC_ROC, na.rm = TRUE), 3), "\n")
    cat("Mean AUC-ROC:", round(mean(values$filtered_data$AUC_ROC, na.rm = TRUE), 3), "\n")
    cat("Median AUC-ROC:", round(median(values$filtered_data$AUC_ROC, na.rm = TRUE), 3), "\n")
    
    # Detection categories
    low_detection <- sum(values$filtered_data$AUC_ROC < 0.7, na.rm = TRUE)
    medium_detection <- sum(values$filtered_data$AUC_ROC >= 0.7 & values$filtered_data$AUC_ROC < 0.85, na.rm = TRUE)
    high_detection <- sum(values$filtered_data$AUC_ROC >= 0.85, na.rm = TRUE)
    
    cat("\nDetection Categories:\n")
    cat("Low detection (AUC < 0.7):", low_detection, 
        "(", round(100 * low_detection / nrow(values$filtered_data), 1), "%)\n", sep = "")
    cat("Medium detection (0.7 ≤ AUC < 0.85):", medium_detection,
        "(", round(100 * medium_detection / nrow(values$filtered_data), 1), "%)\n", sep = "")
    cat("High detection (AUC ≥ 0.85):", high_detection,
        "(", round(100 * high_detection / nrow(values$filtered_data), 1), "%)\n", sep = "")
    
    if (!is.null(input$human_baseline) && !is.na(input$human_baseline)) {
      below_human <- sum(values$filtered_data$AUC_ROC < input$human_baseline, na.rm = TRUE)
      cat("\nHuman Comparison:\n")
      cat("Configurations less detectable than human baseline (", input$human_baseline, "):", 
          below_human, 
          "(", round(100 * below_human / nrow(values$filtered_data), 1), "%)\n", sep = "")
    }
  })
  
  # Least detectable strategies table
  output$least_detectable <- renderTable({
    req(values$data)
    
    least_detectable <- values$data %>%
      arrange(AUC_ROC) %>%  # Sort by lowest AUC-ROC
      head(10) %>%
      select(Dataset, Model, Strategy, Classifier, Hyperparameter, AUC_ROC) %>%
      mutate(
        AUC_ROC = round(AUC_ROC, 4),
        Detection_Level = case_when(
          AUC_ROC < 0.7 ~ "Low",
          AUC_ROC < 0.85 ~ "Medium",
          TRUE ~ "High"
        )
      )
    
    least_detectable
  }, width = "100%")
  
  # Most detectable strategies table
  output$most_detectable <- renderTable({
    req(values$data)
    
    most_detectable <- values$data %>%
      arrange(desc(AUC_ROC)) %>%  # Sort by highest AUC-ROC
      head(10) %>%
      select(Dataset, Model, Strategy, Classifier, Hyperparameter, AUC_ROC) %>%
      mutate(
        AUC_ROC = round(AUC_ROC, 4),
        Detection_Level = case_when(
          AUC_ROC < 0.7 ~ "Low",
          AUC_ROC < 0.85 ~ "Medium",
          TRUE ~ "High"
        )
      )
    
    most_detectable
  }, width = "100%")
  
  # Classifier summary plot
  output$classifier_summary <- renderPlotly({
    req(values$data)
    
    summary_data <- values$data %>%
      group_by(Classifier) %>%
      summarise(
        Mean_AUC = mean(AUC_ROC),
        Median_AUC = median(AUC_ROC),
        Max_AUC = max(AUC_ROC),
        Min_AUC = min(AUC_ROC),
        .groups = "drop"
      )
    
    plot_ly(summary_data, x = ~Classifier, y = ~Mean_AUC, 
            type = 'bar', name = 'Mean',
            marker = list(color = '#2E86AB'),
            text = ~paste("Mean:", round(Mean_AUC, 3)),
            textposition = "outside") %>%
      add_trace(y = ~Min_AUC, name = 'Min (Least Detectable)', 
                marker = list(color = '#28A745'),
                text = ~paste("Min:", round(Min_AUC, 3))) %>%
      add_trace(y = ~Max_AUC, name = 'Max (Most Detectable)', 
                marker = list(color = '#DC3545'),
                text = ~paste("Max:", round(Max_AUC, 3))) %>%
      layout(
        title = "Classifier Detection Range",
        yaxis = list(title = 'AUC-ROC', range = c(0.5, 1.05)),
        xaxis = list(title = ''),
        barmode = 'group'
      )
  })
  
  # Strategy summary plot
  output$strategy_summary <- renderPlotly({
    req(values$data)
    
    strategy_data <- values$data %>%
      group_by(Strategy) %>%
      summarise(
        Mean_AUC = mean(AUC_ROC),
        Min_AUC = min(AUC_ROC),
        Max_AUC = max(AUC_ROC),
        .groups = "drop"
      ) %>%
      arrange(Mean_AUC)  # Sort by least detectable
    
    plot_ly(strategy_data, x = ~reorder(Strategy, Mean_AUC), y = ~Mean_AUC, 
            type = 'bar', name = 'Mean Detection',
            marker = list(color = ~Mean_AUC,
                          colorscale = list(
                            c(0, "#28A745"),  # Green for low
                            c(0.5, "#FFC107"),  # Yellow for medium
                            c(1, "#DC3545")  # Red for high
                          ),
                          cmin = 0.5,
                          cmax = 1.0),
            text = ~paste("Mean:", round(Mean_AUC, 3),
                          "<br>Range:", round(Min_AUC, 3), "-", round(Max_AUC, 3)),
            hovertemplate = "%{text}<extra></extra>") %>%
      layout(
        title = "Strategy Effectiveness (Lower = Better)",
        yaxis = list(title = 'Mean AUC-ROC', range = c(0, 1.05)),
        xaxis = list(title = 'Strategy'),
        showlegend = FALSE
      )
  })
  
  # Insights text
  output$insights_text <- renderUI({
    req(values$data)
    
    # Ensure we have data
    if (nrow(values$data) == 0) {
      return(p("No data available for analysis."))
    }
    
    # Calculate insights with safety checks
    least_detectable <- values$data %>%
      slice_min(AUC_ROC, n = 1)
    
    most_detectable <- values$data %>%
      slice_max(AUC_ROC, n = 1)
    
    strategy_effectiveness <- values$data %>%
      group_by(Strategy) %>%
      summarise(Mean_AUC = mean(AUC_ROC, na.rm = TRUE), .groups = "drop") %>%
      arrange(Mean_AUC)
    
    classifier_difficulty <- values$data %>%
      group_by(Classifier) %>%
      summarise(Mean_AUC = mean(AUC_ROC, na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(Mean_AUC))
    
    # Create insights HTML content
    insights_content <- tagList(
      h4("Key Findings:"),
      tags$ul(
        tags$li(
          tags$strong("Least Detectable Configuration: "), 
          sprintf("AUC-ROC of %.4f achieved by %s with %s strategy against %s",
                  least_detectable$AUC_ROC[1],
                  as.character(least_detectable$Model[1]),
                  as.character(least_detectable$Strategy[1]),
                  as.character(least_detectable$Classifier[1]))
        ),
        tags$li(
          tags$strong("Most Detectable Configuration: "),
          sprintf("AUC-ROC of %.4f by %s with %s strategy against %s",
                  most_detectable$AUC_ROC[1],
                  as.character(most_detectable$Model[1]),
                  as.character(most_detectable$Strategy[1]),
                  as.character(most_detectable$Classifier[1]))
        ),
        if (nrow(strategy_effectiveness) > 0) {
          tags$li(
            tags$strong("Most Effective Strategy for Evasion: "),
            sprintf("%s with mean AUC-ROC of %.3f",
                    as.character(strategy_effectiveness$Strategy[1]),
                    strategy_effectiveness$Mean_AUC[1])
          )
        },
        if (nrow(strategy_effectiveness) > 0) {
          tags$li(
            tags$strong("Least Effective Strategy for Evasion: "),
            sprintf("%s with mean AUC-ROC of %.3f",
                    as.character(strategy_effectiveness$Strategy[nrow(strategy_effectiveness)]),
                    strategy_effectiveness$Mean_AUC[nrow(strategy_effectiveness)])
          )
        },
        if (nrow(classifier_difficulty) > 0) {
          tags$li(
            tags$strong("Most Effective Classifier: "),
            sprintf("%s with mean detection rate of %.3f",
                    as.character(classifier_difficulty$Classifier[1]),
                    classifier_difficulty$Mean_AUC[1])
          )
        }
      )
    )
    
    # Add human comparison if baseline provided
    if (!is.null(input$human_baseline) && !is.na(input$human_baseline)) {
      below_human <- sum(values$data$AUC_ROC < input$human_baseline, na.rm = TRUE)
      total <- nrow(values$data)
      percentage <- round(100 * below_human / total, 1)
      
      human_content <- tagList(
        h4("Human Detection Comparison:"),
        tags$ul(
          tags$li(
            sprintf("%d out of %d configurations (%.1f%%) are less detectable than human baseline of %.2f",
                    below_human, total, percentage, input$human_baseline)
          ),
          tags$li(
            sprintf("This suggests that %.1f%% of machine-generated text configurations are harder to detect than human text",
                    percentage)
          )
        )
      )
      
      insights_content <- tagList(insights_content, human_content)
    }
    
    return(insights_content)
  })
  
  # Data table
  output$data_table <- renderDT({
    req(values$filtered_data)
    
    # Prepare display data
    display_data <- values$filtered_data %>%
      mutate(
        AUC_ROC = round(AUC_ROC, 4),
        Detection_Level = case_when(
          AUC_ROC < 0.7 ~ "Low",
          AUC_ROC < 0.85 ~ "Medium",
          TRUE ~ "High"
        )
      ) %>%
      select(Dataset, Model, Strategy, Classifier, Hyperparameter, AUC_ROC, Detection_Level)
    
    datatable(display_data, 
              options = list(
                pageLength = 15, 
                scrollX = TRUE,
                dom = 'Bfrtip',
                buttons = c('copy', 'csv', 'excel'),
                columnDefs = list(
                  list(className = 'dt-center', targets = '_all')
                )
              ),
              filter = 'top',
              rownames = FALSE) %>%
      formatStyle('Detection_Level',
                  backgroundColor = styleEqual(
                    c('Low', 'Medium', 'High'),
                    c('#d4edda', '#fff3cd', '#f8d7da')
                  ))
  })
  
  # Download plot
  output$download_plot <- downloadHandler(
    filename = function() {
      paste("detection_analysis_", Sys.Date(), ".html", sep = "")
    },
    content = function(file) {
      # Save the current plot as HTML
      p <- plotly::last_plot()
      if (!is.null(p)) {
        htmlwidgets::saveWidget(
          plotly::as_widget(p), 
          file = file, 
          selfcontained = TRUE
        )
      }
    }
  )
}

# Run the app
shinyApp(ui = ui, server = server)