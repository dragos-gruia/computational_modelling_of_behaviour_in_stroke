

import warnings
import os
import itertools

import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm
import numpy as np
import pandas as pd

from statsmodels.stats.multitest import fdrcorrection


def plot_group_device_effect(
    path_files,
    output_path,
    dv,
    demographic_file,
    labels_dv=None,
    column_mapping=None,
    random_effects=None,
    independent_vars=None,
    var_of_interest='device_tablet'
):
    """
    Analyzes and plots the effect of device on multiple task outcome variables.

    This function reads one or more CSV files containing outcome measures for different
    tasks, merges each with demographic data (based on `user_id`), and fits a regression
    model (mixed-effects by default) to estimate the standardized beta coefficient for
    a specified predictor variable (by default, 'device_tablet'). It then produces a
    grid of bar charts, where each row corresponds to one file/task and each bar
    represents the standardized effect size of 'device_tablet' on a given dependent variable.
    Statistically significant effects (after FDR correction) are marked with an asterisk.

    Parameters
    ----------
    path_files : list of str
        A list of file paths (CSV) containing outcome measures for one or more tasks.
        Each file must include:
          - A `user_id` column to merge with demographic data.
          - One or more columns matching the names in `dv` (i.e., the dependent variables).
    output_path : str
        Directory path where the resulting figure (`device_effects.png`) will be saved.
    dv : list of str
        A list of dependent variables (columns in the CSV) to be modeled against
        `independent_vars`.
    demographic_file : str
        File name (CSV or Excel) containing demographic information. This file is
        assumed to be located in `../trial_data/` relative to each task file. Must
        share a `user_id` column for merging.
    labels_dv : list of str, optional
        Custom labels for the dependent variables. These labels are used in the legend.
        Must be the same length as `dv`. If None, the legend will use `dv` directly.
    column_mapping : dict, optional
        A mapping of task file name prefixes (e.g., "IC3_rs_SRT") to more descriptive
        task labels. If provided and contains an entry for a given task name, the subplot
        title uses that entry instead of the raw filename prefix.
    random_effects : list of str, optional
        Columns specifying the random effects for a mixed-effects model. Default is
        `['ID', 'timepoint']`. If `random_effects` is None or empty, an OLS model is used.
        By default, if two random effects are provided, the second is treated as a random
        slope.
    independent_vars : list of str, optional
        Predictor (fixed-effect) variables to include in the model. Must include
        `var_of_interest`. Defaults to:
        [
            'age', 'gender', 'english_secondLanguage', 'education_Alevels',
            'education_bachelors', 'education_postBachelors',
            'NIHSS at admission or 2 hours after thrombectomy/thrombolysis',
            'timepoint', 'device_tablet'
        ].
    var_of_interest : str, optional
        The predictor whose effect is highlighted in the plots, by default 'device_tablet'.

    Returns
    -------
    None
        This function does not return anything. It saves a multi-panel figure to
        `output_path` with the filename `device_effects.png`.

    Raises
    ------
    FileNotFoundError
        If a path in `path_files` or the demographic file cannot be found.
    ValueError
        If the demographic file type is not supported (only .csv or .xlsx are allowed).
    KeyError
        If required columns (e.g., `user_id`, `var_of_interest`, or a variable in `dv`
        or `independent_vars`) are not found in the data.

    Notes
    -----
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
    ...     "IC3_calculation_outcomes.csv",
    ...     "IC3_Comprehension_outcomes.csv"
    ... ]
    >>> output_path = "./results"
    >>> dv = ["Accuracy", "DT_scaled"]
    >>> demographic_file = "demographics.csv"
    >>> labels_dv = ["Calculation Accuracy", "Delay Time"]
    >>> column_map = {
    ...     "IC3_calculation": "Calculation Task",
    ...     "IC3_Comprehension": "Comprehension Task"
    ... }
    >>> plot_group_device_effect(
    ...     path_files=path_files,
    ...     output_path=output_path,
    ...     dv=dv,
    ...     demographic_file=demographic_file,
    ...     labels_dv=labels_dv,
    ...     column_mapping=column_map
    ... )

    """

    # Set defaults if not provided
    if random_effects is None:
        random_effects = ['ID', 'timepoint']
        
    if independent_vars is None:
        independent_vars = [
            'age', 'gender', 'english_secondLanguage', 'education_Alevels',
            'education_bachelors', 'education_postBachelors',
            'NIHSS at admission or 2 hours after thrombectomy/thrombolysis',
            'timepoint', 'device_tablet'
        ]

    # Set up figure dimensions
    fig_rows = len(path_files)/3 if len(path_files)%3 == 0 else fig_rows = int(len(path_files)/3) + 1
    fig_cols = 3 if len(path_files) >=3 else fig_cols = len(path_files)
    
    fig, axes = plt.subplots(
        fig_rows,
        fig_cols,
        figsize=(25, 8 * len(path_files) / 3)
    )
    fig.subplots_adjust(hspace=0.2)

    # Generate subplot coordinates
    subplot_coordinates = list(itertools.product(range(fig_rows), range(fig_cols)))

    # Loop over task files
    for i, task_path in enumerate(path_files):
        # Determine subplot location
        row_idx, col_idx = subplot_coordinates[i]
        ax_sub = axes[row_idx, col_idx]

        # Load the CSV
        if not os.path.isfile(task_path):
            raise FileNotFoundError(f"Task file not found: {task_path}")

        df = pd.read_csv(task_path).reset_index(drop=True)

        # Extract task name
        task_name = task_path.split('/')[-1].split('_outcomes')[0]
        print(task_name)

        # Move into directory containing the file, then up to trial_data for demographics
        os.chdir('/'.join(task_path.split('/')[:-1]))

        # Load demographics
        if demographic_file.endswith('.csv'):
            df_dem = pd.read_csv(f'../trial_data/{demographic_file}')
        elif demographic_file.endswith('.xlsx'):
            df_dem = pd.read_excel(f'../trial_data/{demographic_file}')
        else:
            raise ValueError("Incorrect demographic file type. Must be CSV or XLSX.")

        if not set(['user_id']).issubset(df.columns) or not set(['user_id']).issubset(df_dem.columns):
            raise KeyError("The column 'user_id' is required in both task and demographic data.")

        # Merge with demographics
        df_merged = df.merge(df_dem, how='left', on='user_id')

        # Log-transform NIHSS if present
        if 'NIHSS at admission or 2 hours after thrombectomy/thrombolysis' in df_merged.columns:
            df_merged['NIHSS at admission or 2 hours after thrombectomy/thrombolysis'] = np.log(
                df_merged['NIHSS at admission or 2 hours after thrombectomy/thrombolysis'] + 1
            )

        # If not SRT/NVtrailMaking, map raw outcome to 'Accuracy'
        if task_name not in ['IC3_rs_SRT', 'IC3_NVtrailMaking'] and task_name in df_merged.columns:
            df_merged['Accuracy'] = df_merged[task_name]

        # Standardize age if present
        if 'age' in df_merged.columns:
            df_merged['age'] = (df_merged['age'] - df_merged['age'].mean()) / df_merged['age'].std()

        # Drop NaNs in DV and IV
        df_merged.dropna(subset=dv, inplace=True)
        df_merged.dropna(subset=independent_vars, inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

        # Prepare model matrix
        X = df_merged[independent_vars].copy()
        eff_sizes = []
        p_vals = []
        placeholders = []

        # For each dependent variable
        for dv_measure in dv:
            if dv_measure not in df_merged.columns:
                raise KeyError(f"Dependent variable '{dv_measure}' not found in the dataset.")

            Y = df_merged[dv_measure]
            Y_std = (Y - Y.mean()) / Y.std()

            # Add intercept
            df_merged["Intercept"] = 1

            # Construct model
            if random_effects:
                # If the second random effect is provided, treat it as a random slope
                if len(random_effects) == 2:
                    model = sm.MixedLM(
                        endog=Y_std,
                        exog=X,
                        groups=df_merged[random_effects[0]],
                        exog_re=df_merged[['Intercept', random_effects[1]]].copy()
                    )
                else:
                    model = sm.MixedLM(
                        endog=Y_std,
                        exog=X,
                        groups=df_merged[random_effects[0]]
                    )
            else:
                # If no random effects, fall back to OLS
                X_const = sm.add_constant(X, has_constant='add')
                model = sm.OLS(Y_std, X_const)

            # Fit model
            results = model.fit()

            # Get the predictor coefficient and p-value
            if var_of_interest not in results.params:
                raise KeyError(f"Variable of interest '{var_of_interest}' not in model parameters.")

            beta = results.params[var_of_interest]
            pvalue = results.pvalues[var_of_interest]

            # Standardized beta
            predictor_std = X[var_of_interest].std()
            residual_std = np.sqrt(results.scale)
            std_beta = beta * (predictor_std / residual_std)

            eff_sizes.append(abs(std_beta))
            placeholders.append('')
            p_vals.append(pvalue)

        # Multiple comparison correction (FDR)
        _, p_vals_corrected = fdrcorrection(p_vals, alpha=0.05, method='indep')
        p_vals_corrected = np.round(p_vals_corrected, 2)

        # Plot bar chart for effect sizes
        custom_colors = ['#ffd11a', '#62428a', '#ffd11a', '#62428a']
        sn.barplot(y=eff_sizes, x=placeholders, palette=custom_colors, hue=dv, ax=ax_sub)

        # Hatching for second half of bars
        half_len = len(custom_colors) / 2
        for c_idx, bar in enumerate(ax_sub.patches):
            if c_idx >= half_len:
                bar.set_hatch('//')

        ax_sub.axhline(y=0, linestyle='-', color='black', linewidth=1)
        ax_sub.set_ylim(-0.1, 1.1)

        # If labels for DV are provided
        if labels_dv is not None:
            handles, _ = ax_sub.get_legend_handles_labels()
            ax_sub.legend(handles, labels_dv, fontsize=12)
        else:
            ax_sub.get_legend().remove()

        # Subplot title using column mapping if available
        title_text = column_mapping.get(task_name, task_name) if column_mapping else task_name
        ax_sub.text(
            0.5, 1.1, title_text,
            transform=ax_sub.transAxes,
            fontsize=20,
            va='top',
            ha='center',
            bbox=dict(boxstyle='round', facecolor='white')
        )

        # Y-axis label only on the first column
        if col_idx == 0:
            ax_sub.set_ylabel('Standardised beta coefficient', fontsize=16)
        else:
            ax_sub.set_ylabel('')

        # Hide any figure-wide legend if present
        fig.legend().remove()

        # Annotate bars with numeric values and significance
        for bar_idx, bar in enumerate(ax_sub.patches):
            height = bar.get_height()
            # If the bar height is NaN, interpret as 0
            if np.isnan(height):
                height = 0.0

            # Prepare label text
            label_str = f"{height:.2f}"
            if p_vals_corrected[bar_idx % len(dv)] < 0.05:
                label_str += "*"

            ax_sub.text(
                x=bar.get_x() + (bar.get_width() / 2),
                y=height + (0.02 if height > 0 else height - 0.01),
                s=label_str,
                ha="center",
                fontsize=16,
                fontweight='semibold' if "*" in label_str else 'regular'
            )

    # Save final figure
    save_path = os.path.join(output_path, 'device_effects.png')
    fig.savefig(save_path, format='png', transparent=False)