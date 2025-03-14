

"""
================================================================================
Author: Dragos-Cristian Gruia
Last Modified: 14/03/2025
================================================================================
"""

import os
import IDoCT_model
import task_difficulty
import performance_distributions
import effect_of_demographics
import effect_of_impaired_hand
import path_check


def main(root_path_controls, root_path_patients, task_names, figure_output_directory, column_mapping=None):
    
    """
    Runs the complete analysis pipeline.

    This function orchestrates the end-to-end workflow by performing the following steps:
      1. Ensures that the specified figure output directory exists.
      2. Runs the iDoCT model on control and patient data via `IDoCT_model.main_wrapper` to obtain model outcome directories.
      3. Generates task difficulty scale scatter plots from control data using `task_difficulty.create_group_difficulty_scale`.
      4. Plots performance distributions from patient data using `performance_distributions.plot_group_distributions`.
      5. Runs regression analyses to examine the effects of demographics using `effect_of_demographics.run_group_regression_weights_comparison`.
      6. Plots the effects of hand impairment on outcomes using `effect_of_impaired_hand.plot_group_handImpairment_effect`.

    Parameters
    ----------
    root_path_controls : str
        The directory path containing the control CSV files used by the iDoCT model.
    root_path_patients : str
        The directory path containing the patient CSV files used for applying the iDoCT model.
    task_names : list of str
        A list of task file prefixes (e.g., "IC3_calculation", "IC3_Comprehension"). For each task,
        the pipeline will process the corresponding files.
    figure_output_directory : str
        The directory where all generated figures (including difficulty scales, performance distributions,
        regression plots, and hand impairment plots) will be saved.
    column_mapping : dict, optional
        A mapping from task names to descriptive titles. If provided, these titles will be used to label
        subplots or figures instead of raw task names.

    Returns
    -------
    None
        This function does not return any value; it executes the processing pipeline and saves the resulting
        figures to the specified output directory.

    Raises
    ------
    (Any exceptions raised by internal functions such as file not found or directory errors will propagate.)

    Examples
    --------
    >>> root_path_controls = "/path/to/control_data"
    >>> root_path_patients = "/path/to/patient_data"
    >>> task_names = ["IC3_calculation", "IC3_Comprehension"]
    >>> figure_output_directory = "/path/to/output_figures"
    >>> column_mapping = {"IC3_calculation": "Calculation Task", "IC3_Comprehension": "Comprehension Task"}
    >>> main(root_path_controls, root_path_patients, task_names, figure_output_directory, column_mapping)
    """    
    
    # Ensure that the output directory for figures exists; if not, create it.
    path_check.ensure_directory(figure_output_directory)
    
    # Run the iDoCT model on control and patient data; obtain directories with model outcome files.
    model_outcomes_directory_controls, model_outcomes_directory_patients = IDoCT_model.main_wrapper(root_path_controls, root_path_patients, task_names)
    
    # Create group difficulty scale plots using control data and save the figure to the output directory.
    task_difficulty.create_group_difficulty_scale(model_outcomes_directory_controls, task_names, figure_output_directory, column_mapping)
 
    # Plot performance distributions from patient data and save the figure to the output directory.
    performance_distributions.plot_group_distributions(model_outcomes_directory_patients, task_names, figure_output_directory, column_mapping)
    
    # Run regression analyses to assess the effect of demographics on performance, create figures, and save the results.
    effect_of_demographics.run_group_regression_weights_comparison(model_outcomes_directory_patients,task_names,figure_output_directory, column_mapping)
    
    # Plot the effect of impaired hand on outcomes from patient data, plot figures, and save the figure.
    effect_of_impaired_hand.plot_group_handImpairment_effect(model_outcomes_directory_patients,task_names,figure_output_directory, column_mapping)



