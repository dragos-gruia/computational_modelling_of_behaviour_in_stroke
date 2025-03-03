import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm
import numpy as np
import pandas as pd
import warnings
import os
import itertools
from statsmodels.stats.multitest import fdrcorrection


def plot_group_handImpairment_effect(
    path_files,
    output_path,
    dv,
    demographic_file,
    patient_group=None,
    labels_dv=None,
    column_mapping=None,
    random_effects=['ID', 'timepoint'],
    independent_vars=[
        'age', 'gender', 'english_secondLanguage',
        'education_Alevels', 'education_bachelors',
        'education_postBachelors',
        'NIHSS at admission or 2 hours after thrombectomy/thrombolysis',
        'timepoint', 'impaired_hand'
    ],
    var_of_interest='impaired_hand'
):

    """
    Analyzes and plots the effect of hand impairment on multiple dependent variables.

    This function reads in one or more CSV files containing task outcome measures,
    merges each with demographic and motor information, and fits either a mixed-effects
    or OLS regression model. It then plots bar charts of the standardized beta
    coefficients for a specified predictor variable (by default, 'impaired_hand'). 
    Each subplot corresponds to a single task file, and each bar within the subplot 
    represents one of the dependent variables specified in `dv`.

    Parameters
    ----------
    path_files : list of str
        A list of CSV file paths. Each file should contain:
          - A set of outcome measures (columns) identified by the task name or other
            relevant headers.
          - A shared key column `user_id` for merging with demographic and motor info data.
    output_path : str
        Directory path where the resulting figure (`handImpairment_effects.png`)
        will be saved.
    dv : list of str
        A list of dependent variable column names. The function will model each DV
        against `independent_vars` and display the effect size of `var_of_interest`.
    demographic_file : str
        File name (CSV or Excel) containing demographic data. Must be located in
        the same folder as `../trial_data/` relative to each task file path. 
        This file must share the `user_id` column for merging.
    patient_group : {'acute', 'chronic', None}, optional
        - If 'acute', retains only observations with `timepoint == 1`.
        - If 'chronic', retains only observations with `timepoint != 1`.
        - If None, no filtering by `timepoint` is applied.
        Defaults to None.
    labels_dv : list of str, optional
        Custom labels for each dependent variable in the legend. If provided, must
        be the same length as `dv`.
    column_mapping : dict, optional
        A mapping of task names (parsed from the file name) to more descriptive labels.
        If provided and includes an entry for a given task name, it will be used
        as the plot title for that task’s subplot. Otherwise, the raw name is used.
    random_effects : list of str, optional
        Columns specifying the random effects for a mixed model. If:
          - `len(random_effects) == 1`: Only one grouping factor is used (e.g., ID).
          - `len(random_effects) == 2`: The second grouping factor is modeled as a
            random slope. For instance, `random_effects=['ID', 'timepoint']`.
          - If None or empty, an ordinary least-squares (OLS) model is fitted instead
            of a mixed model.  
        Defaults to ['ID', 'timepoint'].
    independent_vars : list of str, optional
        A list of fixed-effect predictors in the model, which must include `var_of_interest`.
        Defaults to:
        [
            'age', 'gender', 'english_secondLanguage',
            'education_Alevels', 'education_bachelors',
            'education_postBachelors',
            'NIHSS at admission or 2 hours after thrombectomy/thrombolysis',
            'timepoint', 'impaired_hand'
        ].
    var_of_interest : str, optional
        The primary predictor whose effect size is being plotted (default 'impaired_hand').

    Returns
    -------
    None
        The function saves a figure named 'handImpairment_effects.png' in the given
        `output_path` and does not return any value.

    Raises
    ------
    FileNotFoundError
        If a CSV file in `path_files` or the `demographic_file` does not exist.
    ValueError
        If the `demographic_file` is not a .csv or .xlsx.
    KeyError
        If required columns (e.g., `user_id`, or columns in `dv` or `independent_vars`)
        are missing from the input data.

    Notes
    -----
    - The function internally merges each task DataFrame with demographic info
      (in `demographic_file`) and motor information (via `get_motor_information`),
      using `user_id` as the common key.
    - For tasks not named 'IC3_rs_SRT' or 'IC3_NVtrailMaking', the script assigns
      the main outcome measure to `df['Accuracy']`.
    - The function log-transforms `NIHSS at admission or 2 hours after thrombectomy/thrombolysis`
      and standardizes `age`.
    - The dependent variables in `dv` are also standardized before modeling.
    - P-values for the effect of `var_of_interest` on each dependent variable
      are corrected for multiple comparisons using the Benjamini-Hochberg (FDR)
      procedure. Statistically significant results (< 0.05) are marked with an asterisk.

    Examples
    --------
    >>> path_files = [
    ...     "/path/to/IC3_rs_SRT_outcomes.csv",
    ...     "/path/to/IC3_calculation_outcomes.csv"
    ... ]
    >>> output_path = "/path/to/output"
    >>> dv = ["Accuracy", "ReactionTime"]
    >>> demographic_file = "demographics.csv"
    >>> column_mapping = {
    ...     "IC3_rs_SRT": "Simple Reaction Task",
    ...     "IC3_calculation": "Calculation Task"
    ... }
    >>> plot_group_handImpairment_effect(
    ...     path_files=path_files,
    ...     output_path=output_path,
    ...     dv=dv,
    ...     demographic_file=demographic_file,
    ...     patient_group='acute',
    ...     labels_dv=['Acc (%)', 'RT (ms)'],
    ...     column_mapping=column_mapping
    ... )

    """

    fig_rows = len(path_files)/3 if len(path_files)%3 == 0 else fig_rows = int(len(path_files)/3) + 1
    fig_cols = 3 if len(path_files) >=3 else fig_cols = len(path_files)

    # Set up the figure and subplots
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(25, 8 * len(path_files)/3))
    fig.subplots_adjust(hspace=0.2)

    # Generate all subplot coordinates (Cartesian product)
    # e.g., (0,0), (0,1), (0,2), (1,0), ...
    subplot_coordinates = list(itertools.product(range(fig_rows), range(fig_cols)))

    # Prepare a color palette and a simple placeholder for the x-axis categories
    custom_colors = ['#ffd11a', '#62428a', '#ffd11a', '#62428a']
    x_labels = ['']  # Single category on the x-axis, used as a placeholder

    # Loop through each task path
    for i, task_path in enumerate(path_files):
        # Subplot coordinate selection
        row_idx, col_idx = subplot_coordinates[i]
        ax_sub = axes[row_idx, col_idx]

        # Extract the task name from the file path
        task_name = task_path.split('/')[-1].split('_outcomes')[0]

        # Read the CSV for the task
        df_task = pd.read_csv(task_path).reset_index(drop=True)

        # Merge with demographic data
        os.chdir('/'.join(task_path.split('/')[:-1]))

        if demographic_file.endswith('.csv'):
            df_dem = pd.read_csv(f'../trial_data/{demographic_file}')
        elif demographic_file.endswith('.xlsx'):
            df_dem = pd.read_excel(f'../trial_data/{demographic_file}')
        else:
            print('Incorrect demographic file type. Must be CSV or XLSX.')
            return

        # Merge with motor info (assuming get_motor_information is defined elsewhere)
        df_motor = get_motor_information('../trial_data/')

        df = (df_task
              .merge(df_dem, how='left', on='user_id')
              .merge(df_motor, how='left', on='user_id'))

        # Log-transform NIHSS and standardize age
        df['NIHSS at admission or 2 hours after thrombectomy/thrombolysis'] = np.log(
            df['NIHSS at admission or 2 hours after thrombectomy/thrombolysis'] + 1
        )
        df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()

        # If the task is not SRT or NVtrailMaking, assign raw outcome to 'Accuracy'
        if task_name not in ['IC3_rs_SRT', 'IC3_NVtrailMaking']:
            df['Accuracy'] = df[task_name]

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
            title_label = column_mapping.get(task_name, task_name)
        else:
            title_label = task_name

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