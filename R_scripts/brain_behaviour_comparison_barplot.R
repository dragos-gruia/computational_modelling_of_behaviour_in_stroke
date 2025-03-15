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
timepoints = c(2,3)
timepoint_stage <- c("Sub-acute", "Chronic")
column_mapping <- c("IC3_AuditorySustainedAttention", "IC3_BBCrs_blocks", "IC3_Comprehension",
                    "IC3_GestureRecognition", "IC3_NVtrailMaking", "IC3_Orientation", "IC3_PearCancellation",
                    "IC3_SemanticJudgment", "IC3_TaskRecall", "IC3_calculation", "IC3_i4i_IDED",
                    "IC3_i4i_motorControl", "IC3_rs_CRT", "IC3_rs_PAL", "IC3_rs_SRT", "IC3_rs_digitSpan",
                    "IC3_rs_oddOneOut", "IC3_rs_spatialSpan")

input_data_types = c("patient_data_withImaging_linked.xlsx", "patient_data_idoct_withImaging_linked.xlsx")

# Initialize list to store plots and information for each timepoint ------------
combined_plot <- list()
r_estimates <- c()
p_estimates <- c()
labels_list <- c()

for (input_file in input_data_types) {
  
  # Loop over each timepoint -----------------------------------------------------
  for (tp in timepoints) {
    
    # ---------------------------------------------------------------------------#
    # 2. Load and filter patient data for the current timepoint -----------
    #    - Read the patient data, filter based on timepoint, and remove entries
    #      with missing cognitive data (IC3_rs_SRT).
    # ---------------------------------------------------------------------------#
    patient_df <- read_excel(input_file) %>%
      filter(!is.na(IC3_rs_SRT))
    
    patient_df <- if (tp == 3) {
      filter(patient_df, timepoint >= tp)
    } else {
      filter(patient_df, timepoint == tp)
    } 
    
    
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
    # 5. Merge PCA scores with imaging -------------------------------------------
    # ---------------------------------------------------------------------------#
    
    # Combine imaging variables with the PCA factor scores
    df_patients <- patient_df %>%
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
    
    for (imaging_var in c('volume_lesion', 'volume_wmh')) {
      cor_test <- cor.test(df_patients[[imaging_var]], df_patients$g, use = "pairwise.complete.obs")
      r_squared <- round(cor_test$estimate^2, 2)
      r_squared <- ifelse(r_squared < 0.01, 0.01, r_squared)
      p_value   <- round(cor_test$p.value, 3)
      r_estimates <- c(r_estimates, as.numeric(r_squared))
      p_estimates <- c(p_estimates, as.numeric(p_value))
      if (imaging_var == 'volume_lesion') {
        labels <- glue("{timepoint_stage[tp-1]} stroke lesion") 
      } else {
        labels <- glue("{timepoint_stage[tp-1]} WMH lesion") 
      }
      labels_list <- c(labels_list, labels)
    }
    
  }  

}   

data_type = c(rep('Standard accuracy', length(labels_list)/2), rep('Modelled Cognitive Index', length(labels_list)/2))
adjusted_p <- round(p.adjust(p_estimates, method = "fdr"),2)
adjusted_p <- ifelse(adjusted_p < 0.05, "*", " ")

df_plot <- as.data.frame(cbind(r_estimates,labels_list,data_type))
df_plot$data_type = as.factor(df_plot$data_type)
df_plot$r_estimates = as.numeric(df_plot$r_estimates)
df_plot$labels_list = factor(df_plot$labels_list, levels = rev(sort(unique(df_plot$labels_list))))

plot_comparison <- ggplot(df_plot, aes(x = labels_list, y = r_estimates, fill = data_type)) +
  geom_bar(stat = "identity", position = position_dodge2(width = 1, padding = 0.02)) +
  geom_text(aes(label = adjusted_p), 
            vjust = 0.25,
            size=15,
            position = position_dodge(width = 0.9)) +
  theme_classic() +
  theme(text = element_text(size = 16), axis.text.x = element_text(angle = 40, hjust = 1), legend.position = c(0.85, 0.85)) +
  scale_fill_manual(name = 'Cognitive Performance', values = c("#7851A9", "#ffd400")) +
  #ggtitle(timepoint_label) +
  ylab("R-squared") +
  scale_y_continuous(limits = c(0, 0.25)) +  # Sets y-axis from 0 to 100
  xlab("")
  
plot_comparison 

ggsave("imaging_comparison.png", plot = plot_comparison,
       device = "png", width = 12, height = 8, units = "in")