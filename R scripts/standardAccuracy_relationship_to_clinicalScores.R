# ============================================================================ #
# Script: PCA and Correlation Analysis for MOCA/IADL Data with Demographic Merging
#
# Description:
#   This script performs a principal component analysis (PCA) on patient cognitive 
#   data extracted from imaging records and merges it with demographic outcomes 
#   (MOCA and IADL scores) after incorporating additional outcome measures. The 
#   analysis is conducted for three timepoints (Acute, Sub-acute, Chronic), and 
#   correlation plots between the standardized global cognitive index "g" (derived 
#   from PC1) and MOCA/IADL scores are generated for each timepoint. The resulting 
#   plots are combined and saved as a PNG file.
#
# Inputs:
#   - clean_demographics_and_cognition_relabelled.csv : CSV file containing 
#       demographics and cognition data.
#   - IC3_rs_SRT_outcomes.csv : CSV file containing outcome data for IC3_rs_SRT.
#   - IC3_NVtrailMaking_outcomes.csv : CSV file containing outcome data for 
#       IC3_NVtrailMaking.
#   - patient_data_withImaging_linked.xlsx : Excel file containing patient imaging 
#       and clinical data.
#
# Outputs:
#   - pca_results_moca_iadl_raw.png : Combined correlation plots across timepoints.
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
library(glue)

# Set options for parallel processing ------------------------------------------
options(mc.cores = 8)

# -----------------------------------------------------------------------------#
# 1. Define Global Variables and Load Common Datasets -------------------------
# -----------------------------------------------------------------------------#

# Timepoint labels and cognitive column mapping
timepoint_stage <- c("Acute", "Sub-acute", "Chronic")
column_mapping <- c("IC3_AuditorySustainedAttention", "IC3_BBCrs_blocks", "IC3_Comprehension",
                    "IC3_GestureRecognition", "IC3_NVtrailMaking", "IC3_Orientation", "IC3_PearCancellation",
                    "IC3_SemanticJudgment", "IC3_TaskRecall", "IC3_calculation", "IC3_i4i_IDED",
                    "IC3_i4i_motorControl", "IC3_rs_CRT", "IC3_rs_PAL", "IC3_rs_SRT", "IC3_rs_digitSpan",
                    "IC3_rs_oddOneOut", "IC3_rs_spatialSpan")

# Load demographics and cognition data; remove specified columns
df_demog <- read.csv("clean_demographics_and_cognition_relabelled.csv") %>%
  select(-IC3_rs_SRT, -IC3_NVtrailMaking)

# Load outcome data for SRT and NV trail making and rename the outcome column
srt_outcomes <- read.csv("IC3_rs_SRT_outcomes.csv") %>%
  select(user_id, Accuracy) %>%
  rename(IC3_rs_SRT = Accuracy)

trail_outcomes <- read.csv("IC3_NVtrailMaking_outcomes.csv") %>%
  select(user_id, Accuracy) %>%
  rename(IC3_NVtrailMaking = Accuracy)

# Merge outcome data into the demographics dataframe
df_demog <- df_demog %>%
  left_join(srt_outcomes, by = "user_id") %>%
  left_join(trail_outcomes, by = "user_id")

# Subset cognitive columns and remove rows that are entirely missing
df_all_temp <- df_demog %>%
  select(all_of(column_mapping)) %>%
  filter(rowSums(is.na(.)) != ncol(.))

# Load patient data (imaging and clinical) once
patient_data_full <- read_excel("patient_data_withImaging_linked.xlsx")

# Initialize list to store combined plots for each timepoint
combined_plots <- list()

# -----------------------------------------------------------------------------#
# 2. Process Data for Each Timepoint and Generate Correlation Plots -----------
# -----------------------------------------------------------------------------#

for (tp in seq_along(timepoint_stage)) {
  
  # 2.1. Filter patient data based on the current timepoint
  df_moca <- if (tp == 3) {
    patient_data_full %>% filter(timepoint >= tp)
  } else {
    patient_data_full %>% filter(timepoint == tp)
  }
  
  # Remove rows with missing IC3_rs_SRT and preserve user IDs for merging
  df_moca <- df_moca %>% filter(!is.na(IC3_rs_SRT))
  user_ids <- df_moca$user_id
  
  # 2.2. Subset to cognitive columns for PCA
  df_moca_cog <- df_moca %>% select(all_of(column_mapping))
  
  # 2.3. Perform PCA using the 'bpca' method with 17 principal components
  resPCA <- pcaMethods::pca(df_moca_cog, method = "bpca", center = TRUE, nPcs = 17)
  factor_scores <- scores(resPCA)
  
  # 2.4. Merge PCA scores with MOCA and IADL data
  df_patients <- df_moca %>%
    select(user_id, MOCA, IADL) %>%
    left_join(data.frame(user_id = user_ids, factor_scores), by = "user_id") %>%
    # Create the standardized global cognitive index "g" (reverse sign of PC1)
    mutate(g = as.numeric(-scale(PC1))) %>%
    select(-user_id)
  
  # ---------------------------------------------------------------------------#
  # 3. Generate Correlation Plot: Global Cognitive Index (g) vs MOCA ---------
  # ---------------------------------------------------------------------------#
  
  # Compute correlation and p-value for MOCA
  cor_moca <- cor(df_patients$MOCA, df_patients$g, use = "pairwise.complete.obs")
  test_moca <- cor.test(df_patients$MOCA, df_patients$g, use = "pairwise.complete.obs")
  r_value <- round(cor_moca, 2)
  p_value <- ifelse(round(test_moca$p.value, 3) == 0, "P<.001", glue("P={round(test_moca$p.value, 3)}"))
  
  g_moca <- ggplot(df_patients, aes(x = MOCA, y = g)) +
    geom_point(shape = 16, size = 3, color = "#7851A9") +
    geom_smooth(method = "lm", formula = y ~ x, color = "#ffd400", fill = "#fff192") +
    labs(x = "MOCA score", y = "G standard accuracy metrics",
         title = glue("{timepoint_stage[tp]} Phase")) +
    theme_classic() +
    coord_cartesian(ylim = c(min(df_patients$g, na.rm = TRUE),
                             max(df_patients$g, na.rm = TRUE) + 0.3)) +
    theme(text = element_text(size = 16)) +
    annotate("text",
             x = min(df_patients$MOCA, na.rm = TRUE),
             y = max(df_patients$g, na.rm = TRUE) + 0.3,
             hjust = 0, vjust = 1, size = 6,
             label = glue("R-squared={round(r_value^2, 2)}, {p_value}"),
             parse = FALSE)
  
  # ---------------------------------------------------------------------------#
  # 4. Generate Correlation Plot: Global Cognitive Index (g) vs IADL ---------
  # ---------------------------------------------------------------------------#
  
  # Compute correlation and p-value for IADL
  cor_iadl <- cor(df_patients$IADL, df_patients$g, use = "pairwise.complete.obs")
  test_iadl <- cor.test(df_patients$IADL, df_patients$g, use = "pairwise.complete.obs")
  r_value <- round(cor_iadl, 2)
  p_value <- ifelse(round(test_iadl$p.value, 3) == 0, "P<.001", glue("P={round(test_iadl$p.value, 3)}"))
  
  g_iadl <- ggplot(df_patients, aes(x = IADL, y = g)) +
    geom_point(shape = 16, size = 3, color = "#7851A9") +
    geom_smooth(method = "lm", formula = y ~ x, color = "#ffd400", fill = "#fff192") +
    labs(x = "IADL score", y = "G standard accuracy metrics") +
    coord_cartesian(ylim = c(min(df_patients$g, na.rm = TRUE),
                             max(df_patients$g, na.rm = TRUE) + 0.3)) +
    theme_classic() +
    theme(text = element_text(size = 16)) +
    annotate("text",
             x = min(df_patients$IADL, na.rm = TRUE),
             y = max(df_patients$g, na.rm = TRUE) + 0.3,
             hjust = 0, vjust = 1, size = 6,
             label = glue("R-squared={round(r_value^2, 2)}, {p_value}"),
             parse = FALSE)
  
  # ---------------------------------------------------------------------------#
  # 5. Combine the two plots vertically for the current timepoint -----------
  # ---------------------------------------------------------------------------#
  combined_plots[[tp]] <- g_moca / g_iadl
}

# -----------------------------------------------------------------------------#
# 6. Combine Plots Across All Timepoints and Save the Final Figure ------------
# -----------------------------------------------------------------------------#
combined_plot_all <- combined_plots[[1]] | combined_plots[[2]] | combined_plots[[3]]
ggsave("pca_results_moca_iadl_raw.png", plot = combined_plot_all,
       device = "png", width = 12, height = 8, units = "in")
print(combined_plot_all)