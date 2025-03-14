

"""
================================================================================
Author: Dragos-Cristian Gruia
Last Modified: 14/03/2025
================================================================================
"""


import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm
import numpy as np
import pandas as pd
import os
import itertools
from statsmodels.stats.multitest import fdrcorrection
from calculate_motor_impairment import *

def plot_group_handImpairment_effect(
    root_path, 
    task_names,
    output_path,
    column_mapping=None,
    dv=['AS_scaled', 'Accuracy'],                                          
    demographic_file='patient_data_cleaned_linked.xlsx', 
    patient_group=None,
    labels_dv=None,
    random_effects=['ID', 'timepoint'],
    independent_vars=[
        'age', 'gender', 'english_secondLanguage',
        'education_Alevels', 'education_bachelors',
        'education_postBachelors',
        'NIHSS at admission or 2 hours after thrombectomy/thrombolysis',
        'timepoint', 'impaired_hand'
    ],
    var_of_interest='impaired_hand',
    file_extention='_outcomes.csv'):

    """
    Analyzes and plots the effect of hand impairment on multiple dependent variables.

    This function reads in one or more CSV files containing task outcome measures,
    merges each with demographic and motor information, and fits either a mixed-effects
    or OLS regression model. It then plots bar charts of the standardized beta
    coefficients for a specified predictor variable (by default, 'impaired_hand'). 
    Each subplot corresponds to a single task, and each bar within the subplot 
    represents one of the dependent variables specified in `dv`.

    Parameters
    ----------
    root_path : str
        The root directory path where the task outcome files are located. Each
        task name from `task_names` is appended to this path along with `file_extention`
        to form the full file path to the CSV file.
    task_names : list of str
        A list of task name strings (without file extensions). Each task name is
        combined with `root_path` and `file_extention` to create the file path for 
        the corresponding CSV file containing outcome measures. For example, if a
        task name is "IC3_rs_SRT", the file path will be constructed as:
        os.path.join(root_path, "IC3_rs_SRT" + file_extention).
    output_path : str
        Directory path where the resulting figure (`handImpairment_effects.png`)
        will be saved.
    dv : list of str
        A list of dependent variable column names. The function models each DV
        against `independent_vars` and displays the effect size of `var_of_interest`.
    demographic_file : str
        File name (CSV or Excel) containing demographic data. Must be located in
        the '../trial_data/' folder relative to each task file path. This file must
        contain the 'user_id' column for merging.
    patient_group : {'acute', 'chronic', None}, optional
        - If 'acute', retains only observations with `timepoint == 1`.
        - If 'chronic', retains only observations with `timepoint != 1`.
        - If None, no filtering by `timepoint` is applied.
        Defaults to None.
    labels_dv : list of str, optional
        Custom labels for each dependent variable to be used in the plot legend.
        If provided, the list must be the same length as `dv`.
    column_mapping : dict, optional
        A mapping of task name strings to more descriptive labels. If provided and
        contains an entry for a given task name, that label is used as the subplot
        title. Otherwise, the raw task name is used.
    random_effects : list of str, optional
        Columns specifying the random effects for a mixed model. If:
          - `len(random_effects) == 1`: Only one grouping factor is used (e.g., ID).
          - `len(random_effects) == 2`: The second grouping factor is modeled as a
            random slope. For example, `random_effects=['ID', 'timepoint']`.
          - If None or empty, an ordinary least-squares (OLS) model is fitted instead
            of a mixed model.  
        Defaults to ['ID', 'timepoint'].
    independent_vars : list of str, optional
        A list of fixed-effect predictors in the model, which must include 
        `var_of_interest`. Defaults to:
        [
            'age', 'gender', 'english_secondLanguage',
            'education_Alevels', 'education_bachelors',
            'education_postBachelors',
            'NIHSS at admission or 2 hours after thrombectomy/thrombolysis',
            'timepoint', 'impaired_hand'
        ].
    var_of_interest : str, optional
        The primary predictor whose effect size is being plotted (default 'impaired_hand').
    file_extention : str, optional
        Suffix to append to each task name to form the CSV file name. Defaults to
        '_outcomes.csv'. Ensure that the file extension string correctly matches the
        naming convention of your task files.

    Returns
    -------
    None
        The function saves a figure named 'handImpairment_effects.png' in the specified
        `output_path` and does not return any value.

    Raises
    ------
    FileNotFoundError
        If a CSV file constructed from a task name or the demographic file does not exist.
    ValueError
        If the `demographic_file` is not in a supported format (i.e., not .csv, .xls, or .xlsx).
    KeyError
        If required columns (e.g., `user_id`, or columns specified in `dv` or `independent_vars`)
        are missing from the input data.

    Notes
    -----
    - The function internally merges each task DataFrame with demographic information 
      (from `demographic_file`) and motor information (obtained via `get_motor_information`),
      using `user_id` as the common key.
    - For tasks not named 'IC3_rs_SRT' or 'IC3_NVtrailMaking', the function assigns the main
      outcome measure to `df['Accuracy']`.
    - The function log-transforms the variable 
      'NIHSS at admission or 2 hours after thrombectomy/thrombolysis' (after adding 1)
      and standardizes `age`.
    - The dependent variables in `dv` are standardized before modeling.
    - An intercept column is added to the design matrix.
    - P-values for the effect of `var_of_interest` on each dependent variable are corrected for
      multiple comparisons using the Benjamini-Hochberg (FDR) procedure. Statistically significant
      results (adjusted p-value < 0.05) are marked with an asterisk.
    - The bar plots display the absolute standardized beta coefficients, with a hatching pattern
      applied to the second half of the bars in each subplot.
    - Subplots are arranged in a grid with up to three columns per row.

    Examples
    --------
    >>> root_path = "/path/to/task_files"
    >>> task_names = [
    ...     "IC3_rs_SRT",
    ...     "IC3_calculation"
    ... ]
    >>> output_path = "/path/to/output"
    >>> dv = ["Accuracy", "ReactionTime"]
    >>> demographic_file = "demographics.csv"
    >>> column_mapping = {
    ...     "IC3_rs_SRT": "Simple Reaction Task",
    ...     "IC3_calculation": "Calculation Task"
    ... }
    >>> plot_group_handImpairment_effect(
    ...     root_path=root_path,
    ...     task_names=task_names,
    ...     output_path=output_path,
    ...     dv=dv,
    ...     demographic_file=demographic_file,
    ...     patient_group='acute',
    ...     labels_dv=['Acc (%)', 'RT (ms)'],
    ...     column_mapping=column_mapping
    ... )
    """
    
    def load_demographics(task_path):

        """
        Loads demographic data from a CSV or Excel file in `../trial_data/`.

        Parameters
        ----------
        task_path : str
            The file path of a task CSV. The demographic file is assumed to be located
            in `../trial_data/` relative to this path.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing demographic information merged on 'user_id'.
        """

        base_dir = os.path.dirname(task_path)
        demo_path = os.path.join(base_dir, '..', 'trial_data', demographic_file)
        ext = demographic_file.split('.')[-1].lower()
        if ext == 'csv':
            return pd.read_csv(demo_path)
        elif ext in ['xls', 'xlsx']:
            return pd.read_excel(demo_path)
        else:
            raise ValueError("Unsupported demographic file format.")

    fig_rows = len(task_names)/3 if (len(task_names)%3 == 0) else int(len(task_names)/3) + 1
    fig_cols = 3 if (len(task_names) >=3) else len(task_names)

    # Set up the figure and subplots
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(25, 8 * len(task_names)/3))
    fig.subplots_adjust(hspace=0.2)

    # Generate all subplot coordinates (Cartesian product)
    # e.g., (0,0), (0,1), (0,2), (1,0), ...
    subplot_coordinates = list(itertools.product(range(fig_rows), range(fig_cols)))

    # Prepare a color palette and a simple placeholder for the x-axis categories
    custom_colors = ['#ffd11a', '#62428a', '#ffd11a', '#62428a']
    x_labels = ['']  # Single category on the x-axis, used as a placeholder

    # Loop through each task path
    for i, task in enumerate(task_names):
        
        task_path = os.path.join(root_path, task + file_extention)

        # Subplot coordinate selection
        row_idx, col_idx = subplot_coordinates[i]
        
        if fig_rows ==1:
            ax_sub = axes[col_idx]
        else:
            ax_sub = axes[row_idx, col_idx]

        # Read the CSV for the task
        df_task = pd.read_csv(task_path).reset_index(drop=True)

        # Merge with demographics and motor info (assuming get_motor_information is defined elsewhere)
        
        df_dem = load_demographics(task_path)
        
        base_dir = os.path.dirname(task_path)
        motor_path = os.path.join(base_dir, '..', 'trial_data')
        df_motor = get_motor_information(motor_path)

        df = (df_task
              .merge(df_dem, how='left', on='user_id')
              .merge(df_motor, how='left', on='user_id'))

        # Log-transform NIHSS and standardize age
        df['NIHSS at admission or 2 hours after thrombectomy/thrombolysis'] = np.log(
            df['NIHSS at admission or 2 hours after thrombectomy/thrombolysis'] + 1
        )
        df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()

        # If the task is not SRT or NVtrailMaking, assign raw outcome to 'Accuracy'
        if task not in ['IC3_rs_SRT', 'IC3_NVtrailMaking']:
            df['Accuracy'] = df[task]

        # Drop rows missing DV or independent vars
        df.dropna(subset=dv, inplace=True)
        df.dropna(subset=independent_vars, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Optional filtering by patient group
        if patient_group == 'acute':
            df = df[df.timepoint == 1].reset_index(drop=True)
        elif patient_group == 'chronic':
            df = df[df.timepoint != 1].reset_index(drop=True)

        # Prepare design matrix
        X = df[independent_vars].copy()
        df["Intercept"] = 1

        effSize = []
        pvalues = []

        # Loop over each dependent variable
        for dv_measure in dv:
            # Standardize the DV
            Y = df[dv_measure]
            Y = (Y - Y.mean()) / Y.std()

            # Choose model type (MixedLM if random effects are specified, else OLS)
            if random_effects:
                if len(random_effects) == 1:
                    model = sm.MixedLM(endog=Y, exog=X, groups=df[random_effects[0]])
                elif len(random_effects) == 2:
                    model = sm.MixedLM(
                        endog=Y,
                        exog=X,
                        groups=df[random_effects[0]],
                        exog_re=df[['Intercept', random_effects[1]]].copy()
                    )
            else:
                model = sm.OLS(Y, X)

            # Fit model
            results = model.fit()

            # Extract effect size (beta for var_of_interest)
            beta = results.params[var_of_interest]
            pval = results.pvalues[var_of_interest]

            # Compute standardized beta
            predictor_std = X[var_of_interest].std()
            residual_std = np.sqrt(results.scale)
            standardized_beta = beta * (predictor_std / residual_std)

            effSize.append(abs(standardized_beta))
            pvalues.append(pval)

        # FDR-correct the p-values
        _, pvals_adj = fdrcorrection(pvalues, alpha=0.05, method='indep')
        pvals_adj = np.round(pvals_adj, 2)

        # Plot barplot of effect sizes
        sn.barplot(
            y=effSize,
            x=x_labels * len(dv),  # repeat placeholder to match number of bars
            palette=custom_colors,
            hue=dv,
            ax=ax_sub
        )
        ax_sub.axhline(y=0, linestyle='-', color='black', linewidth=1)
        ax_sub.set_ylim(-0.1, 0.8)

        # Manage legend
        if labels_dv:
            handles, legend_labels = ax_sub.get_legend_handles_labels()
            ax_sub.legend(handles, labels_dv, fontsize=12)
        else:
            ax_sub.get_legend().remove()

        # Title or custom label for the subplot
        if column_mapping:
            title_label = column_mapping.get(task, task)
        else:
            title_label = task

        ax_sub.text(
            0.5, 1.15,
            title_label,
            transform=ax_sub.transAxes,
            fontsize=20,
            va='top', ha='center',
            bbox = dict(boxstyle='round', facecolor = 'white')
        )

        # Y-axis labeling
        if col_idx == 0:
            ax_sub.set_ylabel('Standardised beta coefficient', fontsize=16)
        else:
            ax_sub.set_ylabel('')

        # Add numeric values above bars
        for idx, bar in enumerate(ax_sub.patches):
            height = bar.get_height()
            label_fmt = "{:.2f}".format(height)
            if pvals_adj[idx % len(dv)] < 0.05:  # significance check
                label_fmt += "*"
            ax_sub.text(
                x=bar.get_x() + bar.get_width() / 2,
                y=height + (0.02 if height > 0 else -0.01),
                s=label_fmt,
                ha="center",
                fontsize=16,
                fontweight='semibold' if "*" in label_fmt else 'regular'
            )

        # Apply hatching to the second half of the bars
        c=0
        for bar in ax_sub.patches:
            if c>=len(custom_colors)/2:
                bar.set_hatch('//')  # Possible hatches: '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
            c+=1

    # Remove any figure-level legend that might appear after all subplots
    fig.legend().remove()

    # Save the figure
    save_path = os.path.join(output_path, 'handImpairment_effects.png')
    fig.savefig(save_path, format='png', transparent=False)