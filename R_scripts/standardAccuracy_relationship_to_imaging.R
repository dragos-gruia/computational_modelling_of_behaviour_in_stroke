# ============================================================================ #
# Script: PCA and Correlation Analysis for Imaging Data
#
# Description:
#   This script performs a principal component analysis (PCA) on cognitive data 
#   extracted from patient imaging records and evaluates the correlation 
#   between a global cognitive index derived from PCA and imaging volumes 
#   (stroke lesion and white matter lesion volumes). The analysis is performed 
#   separately for two timepoints (2 and 3), and the resulting plots are 
#   combined and saved as a PNG file.
#
# Inputs:
#   - patient_data_withImaging_linked.xlsx : Excel file containing patient 
#       imaging and cognitive data.
#
# Outputs:
#   - pca_results_imaging_raw_log.png : Combined correlation plots.
#
# Author: [Dragos Gruia]
# Date: [3-March-2025]
# ============================================================================ #

# Load required libraries ------------------------------------------------------
library(dplyr)
library(magrittr)
library(pcaMethods)
library(readxl)
library(ggplot2)
library(patchwork)
library(glue)

# Set options for parallel processing
options(mc.cores = 8)

# Define parameters -----------------------------------------------------------
timepoint_stage <- c("Acute", "Sub-acute", "Chronic")
column_mapping <- c(
  "IC3_AuditorySustainedAttention", "IC3_BBCrs_blocks", "IC3_Comprehension",
  "IC3_GestureRecognition", "IC3_NVtrailMaking", "IC3_Orientation",
  "IC3_PearCancellation", "IC3_SemanticJudgment", "IC3_TaskRecall",
  "IC3_calculation", "IC3_i4i_IDED", "IC3_i4i_motorControl",
  "IC3_rs_CRT", "IC3_rs_PAL", "IC3_rs_SRT", "IC3_rs_digitSpan",
  "IC3_rs_oddOneOut", "IC3_rs_spatialSpan"
)


# Define functions -------------------------------------------------------------

process_timepoint <- function(tp) {
  # -------------------------------------------------------------------------- #
  # Description:
  #   Processes patient imaging data for a specified timepoint, performs PCA on
  #   the cognitive measures, and creates correlation plots between the derived 
  #   global cognitive index ("g") and imaging volumes.
  #
  # Arguments:
  #   tp: Numeric value indicating the timepoint (e.g., 2 or 3).
  #
  # Returns:
  #   A ggplot object combining two vertically stacked correlation plots.
  # -------------------------------------------------------------------------- #
  
  # Read patient imaging data
  df_moca <- read_excel("patient_data_withImaging_linked.xlsx")
  
  # Filter data based on the timepoint criteria:
  df_moca <- if (tp == 3) {
    df_moca %>% filter(timepoint >= 3)
  } else {
    df_moca %>% filter(timepoint == tp)
  }
  
  # Exclude rows with missing cognitive outcomes and unwanted user IDs
  df_moca <- df_moca %>%
    filter(!is.na(IC3_rs_SRT))
  
  # Save user IDs for later merging
  user_ids <- df_moca$user_id
  
  # Subset the cognitive measures and order columns alphabetically
  df_cognition <- df_moca %>%
    select(all_of(column_mapping)) %>%
    select(order(colnames(.)))
  
  # Perform PCA using the 'bpca' method with 17 principal components
  pca_method <- "bpca"
  pca_result <- pcaMethods::pca(df_cognition, method = pca_method, 
                                center = TRUE, nPcs = 17)
  print(summary(pca_result))
  eigenvalues <- pca_result@sDev^2
  print(eigenvalues)
  
  # Extract PCA factor scores
  factor_scores <- scores(pca_result)
  
  # Re-attach user IDs to the cognitive data for merging
  df_cognition$user_id <- user_ids
  
  # Retrieve additional imaging metrics from the same Excel file
  df_imaging <- read_excel("patient_data_withImaging_linked.xlsx") %>%
    select(user_id, volume_lesion, volume_wmh, MOCA, IADL)
  
  # Merge cognitive and imaging data, transform imaging volumes, and append PCA scores
  df_patients <- left_join(df_cognition, df_imaging, by = "user_id") %>%
    mutate(
      volume_lesion = log((volume_lesion / 1000) + 1),
      volume_wmh   = log((volume_wmh   / 1000) + 1)
    ) %>%
    bind_cols(as.data.frame(factor_scores)) %>%
    # Create the global cognitive index 'g' from PC1 (standardized and sign-reversed)
    mutate(g = -scale(PC1, center = TRUE, scale = TRUE)[, 1]) %>%
    select(-user_id)
  
  create_corr_plot <- function(metric, x_label) {
    # ------------------------------------------------------------------------ #
    # Description:
    #   Creates a correlation plot for a specified imaging metric versus the 
    #   global cognitive index 'g'.
    #
    # Arguments:
    #   metric: Character string representing the column name of the imaging metric.
    #   x_label: Label for the x-axis in the plot.
    #
    # Returns:
    #   A ggplot object of the correlation plot.
    # ------------------------------------------------------------------------ #
    
    r_value <- round(cor(df_patients[[metric]], df_patients$g, 
                         use = "pairwise.complete.obs"), 2)
    p_val   <- round(cor.test(df_patients[[metric]], df_patients$g, 
                              use = "pairwise.complete.obs")$p.value, 3)
    p_label <- ifelse(p_val == 0, "P<.001", glue("P={p_val}"))
    
    ggplot(df_patients, aes_string(x = metric, y = "g")) +
      geom_point(shape = 16, size = 3, color = "#7851A9") +
      geom_smooth(method = lm, formula = y ~ x, color = "#ffd400", fill = "#fff192") +
      labs(x = x_label, y = "G standard accuracy metrics") +
      theme_classic() +
      theme(text = element_text(size = 16)) +
      coord_cartesian(ylim = c(min(df_patients$g, na.rm = TRUE),
                               max(df_patients$g, na.rm = TRUE) + 0.3)) +
      annotate("text",
               x = min(df_patients[[metric]], na.rm = TRUE),
               y = max(df_patients$g, na.rm = TRUE) + 0.3,
               hjust = 0, vjust = 1, size = 6,
               label = glue("R-squared={round(r_value^2, 2)}, {p_label}"),
               parse = FALSE)
  }
  
  # Create correlation plots for lesion volume and WMH volume
  plot_lesion <- create_corr_plot("volume_lesion", "Stroke lesion volume") +
    ggtitle(glue("{timepoint_stage[tp]} Phase"))
  plot_wmh <- create_corr_plot("volume_wmh", "White matter lesion volume")
  
  # Combine the two plots vertically
  combined_plot_tp <- plot_lesion / plot_wmh
  return(combined_plot_tp)
}

# Analyse data from both timepoints --------------------------------------------
timepoints <- c(2, 3)
plots_list <- lapply(timepoints, process_timepoint)
combined_plot_all <- plots_list[[1]] | plots_list[[2]]

# Display and save the final combined plot -------------------------------------
print(combined_plot_all)
ggsave("pca_results_imaging_raw_log.png", plot = combined_plot_all,
       device = "png", width = 12, height = 8, units = "in")
