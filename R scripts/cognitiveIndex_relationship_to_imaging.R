# ============================================================================ #
# Script: PCA and Correlation Analysis for Imaging Data (IDoct Version)
#
# Description:
#   This script performs a principal component analysis (PCA) on cognitive data
#   extracted from patient imaging records (IDoct version) and evaluates the
#   correlation between a global cognitive index derived from PCA and imaging
#   volumes (stroke lesion and white matter lesion volumes). The analysis is
#   performed separately for two timepoints (2 and 3), and the resulting plots
#   are combined and saved as a PNG file. Additionally, p-values from imaging
#   analyses are adjusted for multiple comparisons using the FDR method.
#
# Inputs:
#   - patient_data_idoct_withImaging_linked.xlsx : Excel file containing patient
#       imaging and cognitive data.
#
# Outputs:
#   - pca_results_imaging_idoct_log.png : Combined correlation plots.
#   - Adjusted p-values printed to the console.
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

# Set options for parallel processing ------------------------------------------
options(mc.cores = 8)

# Define parameters ------------------------------------------------------------
timepoint_stage <- c("Acute", "Sub-acute", "Chronic")
column_mapping <- c(
  "IC3_AuditorySustainedAttention", "IC3_BBCrs_blocks", "IC3_Comprehension",
  "IC3_GestureRecognition", "IC3_NVtrailMaking", "IC3_Orientation",
  "IC3_PearCancellation", "IC3_SemanticJudgment", "IC3_TaskRecall",
  "IC3_calculation", "IC3_i4i_IDED", "IC3_i4i_motorControl",
  "IC3_rs_CRT", "IC3_rs_PAL", "IC3_rs_SRT", "IC3_rs_digitSpan",
  "IC3_rs_oddOneOut", "IC3_rs_spatialSpan"
)

# Load patient imaging and cognitive data -------------------------------------
imaging_data <- read_excel("patient_data_idoct_withImaging_linked.xlsx")

# Define function to process a given timepoint -------------------------------
process_timepoint <- function(tp) {
  
  # ---------------------------------------------------------------------------#
  # Description:
  #   Processes patient imaging data for a specified timepoint, performs PCA on
  #   cognitive measures, and creates correlation plots between the derived
  #   global cognitive index ("g") and imaging volumes.
  #
  # Arguments:
  #   tp: Numeric value indicating the timepoint (e.g., 2 or 3).
  #
  # Returns:
  #   A ggplot object combining two vertically stacked correlation plots.
  # ---------------------------------------------------------------------------#
  
  # Filter data: remove rows with missing cognitive outcomes and unwanted user IDs
  df_moca_full <- imaging_data %>%
    filter(!is.na(IC3_rs_SRT))
  
  # Filter based on timepoint criteria
  if (tp == 3) {
    df_moca_full <- df_moca_full %>% filter(timepoint >= tp)
  } else {
    df_moca_full <- df_moca_full %>% filter(timepoint == tp)
  }
  
  # Create a cognitive-only data frame for PCA, ordering columns alphabetically
  df_moca_cog <- df_moca_full %>%
    select(all_of(column_mapping)) %>%
    select(order(colnames(.)))
  
  # Perform PCA using the 'bpca' method with 6 principal components
  pca_method <- "bpca"
  resPCA <- pcaMethods::pca(df_moca_cog, method = pca_method, center = TRUE,
                            scale = "none", nPcs = 6)
  factor_scores <- scores(resPCA)
  
  # Combine imaging variables with the PCA factor scores
  df_patients <- df_moca_full %>%
    select(user_id, volume_lesion, volume_wmh) %>%
    mutate(
      volume_lesion = log((volume_lesion / 1000) + 1),
      volume_wmh    = log((volume_wmh / 1000) + 1)
    ) %>%
    bind_cols(as.data.frame(factor_scores)) %>%
    filter(volume_lesion != 0) %>%
    # Create the global cognitive index 'g' from PC1 (standardized and sign-reversed)
    mutate(g = -scale(PC1, center = TRUE, scale = TRUE)[, 1]) %>%
    select(-user_id)
  
  # Define function to create a correlation plot for a given metric
  create_corr_plot <- function(metric, x_label) {
    # -------------------------------------------------------------------------#
    # Description:
    #   Creates a correlation plot between the imaging metric and the global
    #   cognitive index 'g'.
    #
    # Arguments:
    #   metric: Character string representing the column name of the imaging metric.
    #   x_label: Label for the x-axis in the plot.
    #
    # Returns:
    #   A ggplot object of the correlation plot.
    # -------------------------------------------------------------------------#
    cor_test <- cor.test(df_patients[[metric]], df_patients$g, use = "pairwise.complete.obs")
    r_squared <- round(cor_test$estimate, 2)^2
    r_squared <- ifelse(r_squared < 0.01, 0.01, r_squared)
    p_value   <- round(cor_test$p.value, 3)
    p_label   <- ifelse(p_value == 0, "P<.001", glue("P={p_value}"))
    
    ggplot(df_patients, aes_string(x = metric, y = "g")) +
      geom_point(shape = 16, size = 3, color = "#7851A9") +
      geom_smooth(method = "lm", formula = y ~ x, color = "#ffd400", fill = "#fff192") +
      labs(x = x_label, y = "G modelled Cognitive Index") +
      theme_classic() +
      theme(text = element_text(size = 16)) +
      coord_cartesian(ylim = c(min(df_patients$g, na.rm = TRUE),
                               max(df_patients$g, na.rm = TRUE) + 0.3)) +
      annotate("text",
               x = min(df_patients[[metric]], na.rm = TRUE),
               y = max(df_patients$g, na.rm = TRUE) + 0.3,
               hjust = 0, vjust = 1, size = 6,
               label = glue("R-squared={round(r_squared, 2)}, {p_label}"))
  }
  
  # Create correlation plots for stroke lesion volume and white matter lesion volume
  plot_lesion <- create_corr_plot("volume_lesion", "Stroke lesion volume") +
    ggtitle(glue("{timepoint_stage[tp]} Phase"))
  plot_wmh <- create_corr_plot("volume_wmh", "White matter lesion volume")
  
  # Combine the two plots vertically
  combined_plot_tp <- plot_lesion / plot_wmh
  return(combined_plot_tp)
}

# Process data for each timepoint and store the resulting plots ----------------
timepoints <- c(2, 3)
plots_list <- lapply(timepoints, process_timepoint)
combined_plot_all <- plots_list[[1]] | plots_list[[2]]

# Display and save the final combined plot ------------------------------------
print(combined_plot_all)
ggsave("pca_results_imaging_idoct_log.png", plot = combined_plot_all, device = "png",
       width = 12, height = 8, units = "in")

# Adjust p-values using FDR correction and print the results -------------------
p_imaging <- c(0.022, 0.566, 0.008, 0.002, 0.122, 0.097, 0.037, 0.01)
adjusted_p <- p.adjust(p_imaging, method = "fdr")
print(adjusted_p)