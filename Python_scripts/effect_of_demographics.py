

"""
================================================================================
Author: Dragos-Cristian Gruia
Last Modified: 14/03/2025
================================================================================
"""


import os
import math
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def run_group_regression_weights_comparison(
    root_path, 
    task_names, 
    output_path, 
    column_mapping=None, 
    dv=['AS_scaled', 'Accuracy'],                                          
    demographic_file='patient_data_cleaned_linked.xlsx', 
    random_effects=['ID', 'timepoint'], 
    independent_vars=['age', 'gender', 'english_secondLanguage',
                        'education_Alevels', 'education_bachelors',
                        'education_postBachelors', 
                        'NIHSS at admission or 2 hours after thrombectomy/thrombolysis', 
                        'timepoint'],
    x_labels=['Age', 'Gender', 'English - second language',
            'Education - A levels', 'Education - Bachelors',
            'Education - Postgraduate', 'NIHSS Score', 'Timepoint'],
    file_extention='_outcomes.csv'):
    
    """
    Generates a grid of subplots comparing linear mixed effects regression coefficients for two dependent variables
    across multiple tasks.

    This function constructs file paths for task outcome CSV files by combining a root directory, a task name,
    and a file extension. Each CSV file is merged with demographic data and then used to fit regression models 
    (MixedLM if random effects are specified, otherwise OLS) for two dependent variables. The first dependent variable 
    is modeled directly while the second is standardized prior to modeling. For each task, the function plots two sets 
    of regression coefficients (with corresponding confidence intervals) side by side with a slight vertical offset.

    Parameters
    ----------
    root_path : str
        The directory path where the task outcome CSV files are located. Each task file is constructed by appending
        a task name (from `task_names`) and `file_extention` to this path.
    task_names : list of str
        A list of task name strings (without file extensions). Each task name is combined with `root_path` and 
        `file_extention` to create the full path for the corresponding CSV file. Each file must include a 
        `user_id` column for merging with demographic data.
    output_path : str
        Directory where the resulting figure (`demographic_effects.png`) will be saved.
    dv : list of str
        A list of exactly two dependent variable column names (e.g., ['AS_scaled', 'Accuracy']). The first dependent
        variable is modeled directly, whereas the second is standardized internally before modeling.
    demographic_file : str
        File name (CSV or Excel) containing demographic information. This file is assumed to reside in a 
        `../trial_data/` directory relative to each task file's location and must share a `user_id` column with the task data.
    column_mapping : dict, optional
        A dictionary mapping task names (as provided in `task_names`) to more descriptive strings for subplot titles.
        If None, the raw task name is used.
    random_effects : list of str, optional
        A two-element list specifying random effects for the regression model. The first element is used as the grouping
        variable (e.g., 'ID') and the second as the random slope (e.g., 'timepoint'). If omitted or empty, an OLS model is used.
    independent_vars : list of str, optional
        A list of predictor column names to be used as fixed effects in the regression. Defaults to:
        [
            'age', 'gender', 'english_secondLanguage',
            'education_Alevels', 'education_bachelors',
            'education_postBachelors',
            'NIHSS at admission or 2 hours after thrombectomy/thrombolysis',
            'timepoint'
        ]
    x_labels : list of str, optional
        A list of display labels corresponding to `independent_vars` for the y-axis in the plots. Defaults to:
        [
            'Age', 'Gender', 'English - second language',
            'Education - A levels', 'Education - Bachelors',
            'Education - Postgraduate', 'NIHSS Score', 'Timepoint'
        ]
    file_extention : str, optional
        Suffix to append to each task name to form the CSV file name. Defaults to '_outcomes.csv'. Ensure that this matches
        the naming convention of your task files.

    Returns
    -------
    None
        Saves a multi-subplot figure named 'demographic_effects.png' in the specified `output_path` directory. 
        No value is returned.

    Raises
    ------
    FileNotFoundError
        If a task CSV file (constructed from a task name) or the specified `demographic_file` does not exist.
    ValueError
        If the `demographic_file` is neither CSV nor Excel (XLS/XLSX), or if `dv` does not contain exactly two dependent variables.
    KeyError
        If required columns (e.g., `user_id`, columns in `dv`, or columns in `independent_vars`) are missing from the data.

    Notes
    -----
    - For tasks other than 'IC3_rs_SRT' and 'IC3_NVtrailMaking', the outcome measure (the column matching the task name) is 
      assigned to `df['Accuracy']`.
    - Age is standardized (subtract mean, divide by standard deviation) if present.
    - The NIHSS score (from 'NIHSS at admission or 2 hours after thrombectomy/thrombolysis') is log-transformed after adding 1.
    - Rows with missing values in any of the columns specified in `dv` or `independent_vars` are dropped.
    - Mixed-effects regression (MixedLM) is used if `random_effects` are provided; otherwise, an OLS model is applied.
    - The inner function `plot_regression` plots the coefficients and their confidence intervals on a given axis, applying a 
      vertical offset (if specified) to distinguish between the two sets of regression results.
    - A background (invisible) barplot is drawn when alignment is needed to maintain consistent spacing for y-axis labels.
    - The x-axis is set to range from -2 to 2 with a red dashed vertical line at 0 for reference.
    - Y-axis labels (from `x_labels`) are displayed only on subplots in the first column.
    - Unused subplot axes are hidden if the number of tasks does not fill the grid.

    Examples
    --------
    >>> task_names = [
    ...     "task1",
    ...     "task2"
    ... ]
    >>> output_path = "/path/to/output"
    >>> dv = ["Accuracy", "ReactionTime"]
    >>> demographic_file = "demographics.csv"
    >>> column_mapping = {"task1": "Task 1 Label", "task2": "Task 2 Label"}
    >>> run_group_regression_weights_comparison(
    ...     root_path="/path/to/task_files",
    ...     task_names=task_names,
    ...     output_path=output_path,
    ...     dv=dv,
    ...     demographic_file=demographic_file,
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
    
    def run_regression(df, indep_vars, dv_col, random_effects, standardize_dv=False):

        """
        Fits a regression model (MixedLM if random effects are provided, else OLS)
        and returns model coefficients and confidence intervals.

        Parameters
        ----------
        df : pandas.DataFrame
            The merged task + demographic DataFrame.
        indep_vars : list of str
            Columns in `df` to use as predictors.
        dv_col : str
            The dependent variable column in `df`.
        random_effects : list of str
            List of columns used for grouping (e.g., ['ID', 'timepoint']).
            If empty or None, an OLS model is used.
        standardize_dv : bool, optional
            If True, the dependent variable is standardized before fitting the model.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns:
            ['predictor', 'coefficient', 'std_coeff', 'std_error', 'ci_lower', 'ci_upper'].

        Notes
        -----
        - The function automatically adds an 'Intercept' column for MixedLM grouping.
        - If `len(random_effects) == 2`, the second element is used as a random slope.
        """

        X = df[indep_vars].copy()
        Y = df[dv_col].copy()
        if standardize_dv:
            Y = (Y - Y.mean()) / Y.std()
        df["Intercept"] = 1  # For mixed-effects model random intercept
        if random_effects:
            model = sm.MixedLM(endog=Y, exog=X, groups=df[random_effects[0]], 
                               exog_re=df[['Intercept', random_effects[1]]].copy())
        else:
            model = sm.OLS(Y, X)
        results = model.fit()
        outputs = []
        for predictor in X.columns:
            coef = results.params[predictor]
            std_err = results.bse[predictor]
            tval = results.tvalues[predictor]
            ci = results.conf_int().loc[predictor]
            outputs.append((predictor, coef, tval, std_err, ci[0], ci[1]))
        results_df = pd.DataFrame(outputs, 
                                  columns=['predictor', 'coefficient', 'std_coeff', 'std_error', 'ci_lower', 'ci_upper'])
        # Remove constant if present
        return results_df[results_df['predictor'] != 'const']
    
    def plot_regression(ax, results_df, offset=0, color='#62428a', cap_width=0.1, aligment_needed = True):

        """
        Plots regression coefficients, confidence intervals, and vertical caps onto a given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The subplot axis on which to draw.
        results_df : pandas.DataFrame
            A DataFrame containing columns ['predictor', 'coefficient', 'std_coeff',
            'std_error', 'ci_lower', 'ci_upper'].
        offset : float, optional
            Vertical offset to apply to the points, allowing multiple sets of coefficients
            to appear side by side. Defaults to 0.0.
        color : str, optional
            Color for the points and confidence interval lines.
        cap_width : float, optional
            The half-height of the vertical cap lines at each confidence interval boundary.
        aligment_needed : bool, optional
            If True, a background barplot is drawn first to help align different coefficient sets.
            This is useful if the results of multiple regressions are overlapped on the plot.

        Returns
        -------
        None
            Modifies the `ax` in place.

        Notes
        -----
        - This function typically is called twice per task: once for the standard DV and once for
          the modelled DV. The `offset` parameter visually separates them.
        - The barplot drawn when `aligment_needed` is True is invisible (white color)
          but preserves the correct spacing for the y-axis labels.
        """
        
        if aligment_needed:
            # Background barplot for standardized coefficients (for alignment)
            sns.barplot(y=results_df['predictor'], x=results_df['std_coeff'], orient='h',
                        color='white', ax=ax, capsize=0)
            
        # Plot the coefficients
        y_positions = np.arange(len(results_df)) + offset
        sns.scatterplot(x=results_df['coefficient'], y=y_positions, marker='o', s=200, ax=ax, color=color)
        for idx, row in results_df.iterrows():
            y_val = idx + offset
            ax.hlines(y=y_val, xmin=row['ci_lower'], xmax=row['ci_upper'], colors=color, linewidth=2)
            ax.vlines(x=row['ci_lower'], ymin=y_val - cap_width/2, ymax=y_val + cap_width/2, colors=color, linewidth=2)
            ax.vlines(x=row['ci_upper'], ymin=y_val - cap_width/2, ymax=y_val + cap_width/2, colors=color, linewidth=2)
    
    # Set up the subplot grid (3 columns; rows determined by the number of files)
    n_files = len(task_names)
    n_cols = 3
    n_rows = math.ceil(n_files / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(25, 8 * n_rows))
    fig.subplots_adjust(hspace=0.3)
    axs = axs.flatten()
    
    for idx, task in enumerate(task_names):
        
        ax = axs[idx]
        task_path = os.path.join(root_path, task + file_extention)
        
        # Load the task data and extract task name from filename
        df = pd.read_csv(task_path).reset_index(drop=True)
        
        # Merge with demographic information (without changing working directory)
        df_dem = load_demographics(task_path)
        df = df.merge(df_dem, how='left', on='user_id')
        
        # Transform NIHSS if present
        nihss_col = 'NIHSS at admission or 2 hours after thrombectomy/thrombolysis'
        if nihss_col in df.columns:
            df[nihss_col] = np.log(df[nihss_col] + 1)
        
        # For tasks other than specified ones, assign the task outcome to 'Accuracy'
        if task not in ['IC3_rs_SRT', 'IC3_NVtrailMaking']:
            df['Accuracy'] = df[task]
        
        # Standardize age
        if 'age' in df.columns:
            df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()
        
        # Drop missing values for dependent and independent variables
        df = df.dropna(subset=dv + independent_vars).reset_index(drop=True)
        
        # Run regressions for the two dependent variables
        results_df_AS = run_regression(df, independent_vars, dv[0], random_effects, standardize_dv=False)
        results_df_raw = run_regression(df, independent_vars, dv[1], random_effects, standardize_dv=True)
        
        # Plot regression coefficients with a small vertical offset for the second model
        vertical_gap = 0.2
        plot_regression(ax, results_df_AS, offset=0, color='#62428a', aligment_needed=True)
        plot_regression(ax, results_df_raw, offset=vertical_gap, color='#ffcc00', aligment_needed=False)
        
        # Format the subplot
        ax.set_xlim(-2, 2)
        ax.grid(True, linewidth=1)
        xticks = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.axvline(x=0, linestyle='--', color='red', linewidth=1)
        ax.set_xlabel('Standardised Beta Coefficients', fontsize=16)
        
        # Display y-axis labels only on the first column subplots
        if idx % n_cols == 0:
            ax.set_yticks(np.arange(len(x_labels)))
            ax.set_yticklabels(x_labels, fontsize=12)
            ax.set_ylabel('')
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        
        # Annotate the subplot with task name or mapped display name
        if column_mapping is not None:
            display_text = column_mapping.get(task, task)
            ax.text(0.5, 1.1, display_text, transform=ax.transAxes, fontsize=20, 
                    va='top', ha='center', bbox=dict(boxstyle='round', facecolor='white'))
        else:
            ax.text(-0.1, 1.35, task, transform=ax.transAxes, fontsize=20, 
                    va='top', ha='center')
    
    # Hide any unused axes if n_files does not fill the grid
    for j in range(idx + 1, len(axs)):
        axs[j].axis('off')
    
    fig.savefig(os.path.join(output_path, 'demographic_effects.png'), format='png', transparent=False)