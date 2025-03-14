# ============================================================================ #
# Script: PCA and Correlation Analysis for MOCA and IADL Data (IDoct Version)
#
# Description:
#   This script performs a principal component analysis (PCA) on cognitive data
#   extracted from healthy and patient imaging records (IDoct version) and
#   evaluates the correlation between a global cognitive index derived from PCA
#   and two clinical measures (MOCA and IADL scores). The analysis is conducted
#   separately for each timepoint (Acute, Sub-acute, Chronic) and the resulting
#   plots are combined and saved as a PNG file.
#
# Inputs:
#   - clean_healthy_idoct.xlsx : Excel file containing healthy cognitive data.
#   - patient_data_idoct_withImaging_linked.xlsx : Excel file containing patient
#       imaging, MOCA, and IADL data.
#
# Outputs:
#   - pca_results_moca_iadl_idoct.png : Combined correlation plots.
#
# Author: [Dragos Gruia]
# Date: [3-March-2025]
# ============================================================================ #

# Load required libraries ------------------------------------------------------
library(dplyr)
library(readxl)
library(ggplot2)
library(pcaMethods)
library(patchwork)
library(corrplot)
library(glue)

# Set options for parallel processing ------------------------------------------
options(mc.cores = 8)

# Define global parameters -----------------------------------------------------
timepoint_stage <- c("Acute", "Sub-acute", "Chronic")
column_mapping <- c("IC3_AuditorySustainedAttention", "IC3_BBCrs_blocks", "IC3_Comprehension",
                    "IC3_GestureRecognition", "IC3_NVtrailMaking", "IC3_Orientation", "IC3_PearCancellation",
                    "IC3_SemanticJudgment", "IC3_TaskRecall", "IC3_calculation", "IC3_i4i_IDED",
                    "IC3_i4i_motorControl", "IC3_rs_CRT", "IC3_rs_PAL", "IC3_rs_SRT", "IC3_rs_digitSpan",
                    "IC3_rs_oddOneOut", "IC3_rs_spatialSpan")

# Initialize list to store combined plots for each timepoint ------------------
combined_plot <- list()

# Loop over each timepoint -----------------------------------------------------
for (tp in seq_along(timepoint_stage)) {
  
  # ---------------------------------------------------------------------------#
  # 1. Load and process healthy data ---------------
  #    - Select the specified cognitive columns and filter out rows where all 
  #      columns are NA.
  # ---------------------------------------------------------------------------#
  healthy_data <- read_excel("clean_healthy_idoct.xlsx") %>%
    select(all_of(column_mapping)) %>%
    filter(rowSums(is.na(.)) != ncol(.))
  
  # ---------------------------------------------------------------------------#
  # 2. Load and filter patient data for the current timepoint -----------
  #    - Read the patient data, filter based on timepoint, and remove entries
  #      with missing cognitive data (IC3_rs_SRT).
  # ---------------------------------------------------------------------------#
  patient_df <- read_excel("patient_data_idoct_withImaging_linked.xlsx")
  patient_df <- if (tp == 3) {
    filter(patient_df, timepoint >= tp)
  } else {
    filter(patient_df, timepoint == tp)
  } %>% 
    filter(!is.na(IC3_rs_SRT))
  
  # Preserve the user_id for later joins (if needed)
  users_patients <- patient_df$user_id
  
  # ---------------------------------------------------------------------------#
  # 3. Subset cognitive columns for PCA ----------------------------------------
  # ---------------------------------------------------------------------------#
  patient_cog <- patient_df %>% select(all_of(column_mapping))
  
  # ---------------------------------------------------------------------------#
  # 4. Perform PCA using the "bpca" method ------------------------------------
  #    - PCA is performed on the patient cognitive data with 17 principal components.
  # ---------------------------------------------------------------------------#
  pca_method <- "bpca"
  resPCA <- pcaMethods::pca(patient_cog, method = pca_method, center = TRUE, nPcs = 17)
  summary(resPCA)
  factor_scores <- scores(resPCA)
  
  # ---------------------------------------------------------------------------#
  # 5. Merge PCA scores with MOCA and IADL data --------------------------------
  #    - Combine the PCA factor scores with clinical data (MOCA and IADL).
  #    - Compute the standardized global cognitive index "g" (reverse sign of PC1).
  # ---------------------------------------------------------------------------#
  df_patients <- patient_df %>%
    select(user_id, MOCA, IADL) %>%
    cbind(factor_scores) %>%
    mutate(g = -((PC1 - mean(PC1, na.rm = TRUE)) / sd(PC1, na.rm = TRUE))) %>%
    select(-user_id)
  
  # ---------------------------------------------------------------------------#
  # 6. Generate plots for correlations between "g" and MOCA/IADL ---------------
  # ---------------------------------------------------------------------------#
  
  # Plot: G vs MOCA 
  r_moca <- round(cor(df_patients$MOCA, df_patients$g, use = "pairwise.complete.obs"), 2)
  p_moca <- round(cor.test(df_patients$MOCA, df_patients$g, use = "pairwise.complete.obs")$p.value, 3)
  p_moca <- ifelse(p_moca == 0, "P<.001", glue("P={p_moca}"))
  
  g_moca <- ggplot(df_patients, aes(x = MOCA, y = g)) +
    geom_point(shape = 16, size = 3, color = "#7851A9") +
    geom_smooth(method = "lm", formula = y ~ x, color = "#ffd400", fill = "#fff192") +
    labs(x = "MOCA score", y = "G modelled Cognitive Index",
         title = glue("{timepoint_stage[tp]} Phase")) +
    theme_classic() +
    theme(text = element_text(size = 16)) +
    annotate("text",
             x = min(df_patients$MOCA, na.rm = TRUE),
             y = max(df_patients$g, na.rm = TRUE) + 0.3,
             hjust = 0, vjust = 1, size = 6,
             label = glue("R-squared={round(r_moca^2, 2)}, {p_moca}"),
             parse = FALSE)
  
  # Plot: G vs IADL 
  r_iadl <- round(cor(df_patients$IADL, df_patients$g, use = "pairwise.complete.obs"), 2)
  p_iadl <- round(cor.test(df_patients$IADL, df_patients$g, use = "pairwise.complete.obs")$p.value, 3)
  p_iadl <- ifelse(p_iadl == 0, "P<.001", glue("P={p_iadl}"))
  
  g_iadl <- ggplot(df_patients, aes(x = IADL, y = g)) +
    geom_point(shape = 16, size = 3, color = "#7851A9") +
    geom_smooth(method = "lm", formula = y ~ x, color = "#ffd400", fill = "#fff192") +
    labs(x = "IADL score", y = "G modelled Cognitive Index") +
    coord_cartesian(ylim = c(min(df_patients$g, na.rm = TRUE),
                             max(df_patients$g, na.rm = TRUE) + 0.3)) +
    theme_classic() +
    theme(text = element_text(size = 16)) +
    annotate("text",
             x = min(df_patients$IADL, na.rm = TRUE),
             y = max(df_patients$g, na.rm = TRUE) + 0.3,
             hjust = 0, vjust = 1, size = 6,
             label = glue("R-squared={round(r_iadl^2, 2)}, {p_iadl}"),
             parse = FALSE)
  
  # ---------------------------------------------------------------------------#
  # 7. Combine the two plots vertically for the current timepoint --------------
  # ---------------------------------------------------------------------------#
  combined_plot[[tp]] <- g_moca / g_iadl
}

# -----------------------------------------------------------------------------#
# 8. Combine plots across all timepoints and save the final figure -------------
# -----------------------------------------------------------------------------#
combined_plot_all <- combined_plot[[1]] | combined_plot[[2]] | combined_plot[[3]]
print(combined_plot_all)
ggsave("pca_results_moca_iadl_idoct.png", plot = combined_plot_all,
       device = "png", width = 12, height = 8, units = "in")