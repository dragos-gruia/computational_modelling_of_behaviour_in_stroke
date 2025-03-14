# ============================================================================ #
# Script: CCA and Correlation Analysis for Imaging Data (IDoct Version)
#
# Description:
#   This script performs a canonical correlation analysis (CCA) between modelled
#   cognitive data (IDoct version) and brain imaging metrics. It begins by loading
#   and preprocessing the data—including missing value imputation using random forests.
#   The script then computes correlation matrices for the cognitive data, performs
#   CCA between cognitive measures and imaging volumes (white matter and stroke lesion
#   volumes), and creates scatter plots with regression lines for the first canonical
#   variates as well as bar plots for CCA loadings. The analysis is executed separately
#   for two timepoints (2 and 3), and the resulting plots are combined and saved as PNG
#   files.
#
# Inputs:
#   - patient_data_idoct_withImaging_linked.xlsx : Excel file containing patient
#       imaging and cognitive data.
#
# Outputs:
#   - CCA_corr_idoct.png : Combined scatter plots for canonical variates.
#   - CCA_load_idoct.png : Combined bar plots for CCA loadings.
#
# Author: [Dragos Gruia]
# Date: [3-March-2025]
# ============================================================================ #


# Load required libraries only -------------------------------------------------
library(dplyr)      
library(readxl)      
library(ggplot2)    
library(glue)        
library(missForest)  
library(corrplot)   
library(patchwork)   
library(CCA)         

# Set global options
options(mc.cores = 8)

# Define Functions -------------------------------------------------------------
impute_missing_data <- function(data, max_iter = 25, num_trees = 500) {
  
  # -------------------------------------------------------------------------- #
  # Description : Impute missing values in a numeric matrix or 
  # data frame using missForest.
  #
  # Parameters:
  #   data - A numeric matrix or data frame with missing values.
  #   max_iter - Maximum number of iterations (default: 25).
  #   num_trees - Number of trees in the random forest (default: 500).
  #
  # Returns:
  #   A data frame with imputed values.
  # -------------------------------------------------------------------------- #
  
  imputed_result <- missForest(data, maxiter = max_iter, ntree = num_trees, 
                               verbose = TRUE, variablewise = TRUE)
  return(imputed_result$ximp)
}


load_and_preprocess_data <- function(filepath, column_mapping, timepoint) {
  
  # -------------------------------------------------------------------------- #
  # Description: Load an Excel file, filter out unwanted rows and missing values,
  # impute missing cognitive data, and filter based on the provided timepoint.
  #
  # Parameters:
  #   filepath - Path to the Excel file.
  #   column_mapping - Vector of column names for cognitive data.
  #   timepoint - Numeric value (2 or 3) for filtering the data.
  #
  # Returns:
  #   A preprocessed data frame ready for CCA.
  # -------------------------------------------------------------------------- #
  
  df <- read_excel(filepath) %>%
    # Filter out rows with missing IC3_rs_SRT and unwanted user IDs
    filter(!is.na(IC3_rs_SRT))
  
  # Select cognitive columns, impute missing data, and convert back to data frame
  df_imputed <- df %>%
    select(all_of(column_mapping)) %>%
    as.matrix() %>%
    impute_missing_data() %>%
    as.data.frame()
  
  # Replace original columns with imputed data
  df[, colnames(df_imputed)] <- df_imputed
  
  # Filter data based on the timepoint value
  if (timepoint == 3){
    df = df[df$timepoint >=timepoint,]
  } else {
    df = df[df$timepoint ==timepoint,]
  }
  
  # Remove rows with missing lesion volume
  df <- df %>% filter(!is.na(volume_lesion))
  return(df)
}


create_cca_scatterplot <- function(data, x_col, y_col, timepoint_label) {
  
  # -------------------------------------------------------------------------- #
  # Description: Create a scatter plot for CCA scores with a regression line and annotation.
  #
  # Parameters:
  #   data - Data frame containing x and y scores.
  #   x_col - Column name for the x-axis.
  #   y_col - Column name for the y-axis.
  #   timepoint_label - Title corresponding to the timepoint.
  #
  # Returns:
  #   A ggplot object.
  # -------------------------------------------------------------------------- #
  
  # Calculate correlation coefficient and p-value for annotation
  r_value <- round(cor(data[[x_col]], data[[y_col]], use = "complete.obs"), 2)
  p_value <- round(cor.test(data[[x_col]], data[[y_col]], use = "pairwise.complete.obs")$p.value, 3)
  p_label <- ifelse(p_value == 0, "P<.001", glue("P={p_value}"))
  
  # Build the scatter plot with regression line and annotation text
  plot <- ggplot(data, aes_string(x = x_col, y = y_col)) +
    geom_point(shape = 16, size = 3, color = '#7851A9') +
    geom_smooth(method = "lm", formula = y ~ x, color = '#ffd400', fill = '#fff192') +
    xlab("MRI metrics") +
    ylab("Cognitive data using the modelled Cognitive Index") +
    ggtitle(timepoint_label) +
    theme_classic() +
    theme(text = element_text(size = 16)) +
    annotate("text", 
             x = min(data[[x_col]], na.rm = TRUE), 
             y = max(data[[y_col]], na.rm = TRUE) + 0.3, 
             hjust = 0, vjust = 1, size = 6, 
             label = glue("R-squared={round(r_value^2, 2)}, {p_label}"), 
             parse = FALSE)
  return(plot)
}


create_cca_barplot <- function(data_corr, timepoint_label) {
  
  # -------------------------------------------------------------------------- #
  # Description: Create a bar plot for CCA loadings.
  #
  # Parameters:
  #   data_corr - Data frame containing CCA loadings, variable names, and data types.
  #   timepoint_label - Plot title corresponding to the timepoint.
  #
  # Returns:
  #   A ggplot object.
  # -------------------------------------------------------------------------- #
  
  plot <- ggplot(data_corr, aes(x = cca_corr, y = var_names, fill = `Data Type`)) +
    geom_col(position = "dodge") +
    theme_classic() +
    theme(text = element_text(size = 16)) +
    scale_fill_brewer(palette = "Dark2") +
    ggtitle(timepoint_label) +
    xlab("R-value") +
    ylab("")
  return(plot)
}

# ---------------------------------------------------------------------------- #
# Main Script Execution ------------------------------------------------------
# ---------------------------------------------------------------------------- #

# Define cognitive data column names and timepoint labels ----------------------
column_mapping <- c(
  "IC3_AuditorySustainedAttention", "IC3_BBCrs_blocks", "IC3_Comprehension",
  "IC3_GestureRecognition", "IC3_NVtrailMaking", "IC3_Orientation",
  "IC3_PearCancellation", "IC3_SemanticJudgment", "IC3_TaskRecall",
  "IC3_calculation", "IC3_i4i_IDED", "IC3_i4i_motorControl", "IC3_rs_CRT",
  "IC3_rs_PAL", "IC3_rs_SRT", "IC3_rs_digitSpan", "IC3_rs_oddOneOut",
  "IC3_rs_spatialSpan"
)
timepoint_labels <- c("Acute", "Sub-acute CCA Mode", "Chronic CCA Mode")

# Initialize lists to store scatter and bar plots for each timepoint (2 and 3)
plot_list <- vector("list", 2)
plot_list_bar <- vector("list", 2)

# Loop through the selected timepoints (2 and 3) -------------------------------
for (timepoint in 2:3) {
  
  # Load and preprocess data from the Excel file -------------------------------
  df_moca <- load_and_preprocess_data("patient_data_idoct_withImaging_linked.xlsx", 
                                      column_mapping, timepoint)
  
  # Extract cognitive and imaging subsets for CCA
  df_cog <- df_moca %>% select(all_of(column_mapping))
  df_img <- df_moca %>% select(volume_wmh, volume_lesion)
  
  # Display correlation plots for cognitive data -------------------------------
  corr_values <- cor(df_cog)
  corrplot(corr_values, method = "shade", order = "AOE", diag = FALSE)
  corrplot.mixed(corr_values, order = "AOE")
  
  # Perform Canonical Correlation Analysis (CCA) -------------------------------
  cca_result <- cc(df_cog, df_img)
  
  # Combine the CCA scores into one data frame for plotting
  data_x <- as.data.frame(cca_result$scores$xscores)
  colnames(data_x) <- c("x_score1", "x_score2")
  data_y <- as.data.frame(cca_result$scores$yscores)
  colnames(data_y) <- c("y_score1", "y_score2")
  data_combined <- cbind(data_x, data_y)
  
  # Create and store the scatter plot for the first canonical variate ----------
  plot_list[[timepoint - 1]] <- create_cca_scatterplot(data_combined, 
                                                       "x_score1", "y_score1", 
                                                       timepoint_labels[timepoint])
  
  # Prepare data for the CCA loadings bar plot ---------------------------------
  cor_x <- as.data.frame(cca_result$scores$corr.X.xscores[, 1])
  colnames(cor_x) <- "cca_corr"
  cor_y <- as.data.frame(cca_result$scores$corr.Y.yscores[, 1])
  colnames(cor_y) <- "cca_corr"
  
  var_names <- c(
    "Volume - White Matter", "Volume - Stroke Lesion", "Auditory Attention",
    "Blocks", "Language Comprehension", "Gesture Recognition", "Trail-making",
    "Orientation", "Pear Cancellation", "Semantic Judgement", "Task Recall",
    "Calculations", "Rule Learning", "Motor Control", "CRT", 
    "Paired Associates Learning", "SRT", "Digit Span", "Odd one out", "Spatial Span"
  )
  
  # Combine imaging and cognitive loadings into one data frame
  data_corr <- rbind(cor_y, cor_x)
  data_corr$var_names <- var_names
  data_corr$`Data Type` <- "Cognitive"
  data_corr[c(1, 2), "Data Type"] <- "Imaging"
  rownames(data_corr) <- NULL
  
  # Create and store the bar plot for CCA loadings -----------------------------
  p_bar <- create_cca_barplot(data_corr, timepoint_labels[timepoint])
  if (timepoint == 2) {
    p_bar <- p_bar + theme(legend.position = "none")
  }
  plot_list_bar[[timepoint - 1]] <- p_bar
  
  # (Optional) Create plot for the second canonical variate if needed ----------
  # Here we calculate and annotate the second variate (not stored for later use).
  r_value <- round(cor(cca_result$scores$xscores[, 2], cca_result$scores$yscores[, 2], use = "complete.obs"), 2)
  p_value <- round(cor.test(cca_result$scores$xscores[, 2], cca_result$scores$yscores[, 2], use = "pairwise.complete.obs")$p.value, 3)
  p_label <- ifelse(p_value == 0, "P<.001", glue("P={p_value}"))
  
  g_f2 <- ggplot(data_combined, aes(x_score2, y_score2)) +
    geom_point(shape = 16, size = 3, color = '#7851A9') +
    geom_smooth(method = "lm", formula = y ~ x, color = '#ffd400', fill = '#fff192') +
    xlab("MRI metrics") +
    ylab("Cognitive data using the modelled Cognitive Index") +
    ggtitle(timepoint_labels[timepoint]) +
    theme_classic() +
    theme(text = element_text(size = 16)) +
    annotate("text", x = min(data_combined$x_score2, na.rm = TRUE), 
             y = max(data_combined$y_score2, na.rm = TRUE) + 0.3, 
             hjust = 0, vjust = 1, size = 6, 
             label = glue("R-squared={round(r_value^2, 2)}, {p_label}"), 
             parse = FALSE)
  
  # Uncomment the next line to display the second variate plot if needed
  # print(g_f2)
}

# Combine scatter plots for timepoints 2 and 3 and save as PNG -----------------
combined_plot <- plot_list[[1]] | plot_list[[2]]
combined_plot
ggsave("CCA_corr_idoct.png", plot = combined_plot, device = "png", width = 12, height = 8, units = "in")

# Combine bar plots for CCA loadings and save as PNG ---------------------------
combined_plot_bar <- plot_list_bar[[1]] | plot_list_bar[[2]]
combined_plot_bar
ggsave("CCA_load_idoct.png", plot = combined_plot_bar, device = "png", width = 12, height = 8, units = "in")