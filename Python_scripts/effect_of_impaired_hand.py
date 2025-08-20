

"""
================================================================================
Author: Dragos-Cristian Gruia
Last Modified: 20/08/2025
================================================================================
"""


import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm
import numpy as np
import pandas as pd
import os
import itertools
import scipy.stats as stats
from pymer4.models import glmer
from rpy2.robjects import pandas2ri
import polars as pl
from statsmodels.stats.multitest import fdrcorrection
from sklearn.preprocessing import PowerTransformer
from calculate_motor_impairment import *


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

    This function reads in a CSV file containing task outcome measures,
    merges each with demographic and motor information, and fits either a linear mixed-effects
    or binomial mixed effects regression model. It then plots bar charts of the standardized
    coefficients for a specified predictor variable (by default, 'impaired_hand'). 
    Each subplot corresponds to a single task, and each bar within the subplot 
    represents one of the dependent variables specified in `dv`.

    Parameters
    ----------
    path_files : list of str
        List of file paths containing outcome measures (CSV format).
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
        Columns specifying the random effects for a mixed model.
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

    Returns
    -------
    None
        The function saves a figure named 'handImpairment_effects.png' in the specified
        `output_path` and does not return any value.

    """
    
    fig_rows = 6
    fig_cols = 3

    # Set up the figure and subplots
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(25, 8 * len(path_files)/3))
    fig.subplots_adjust(hspace=0.2)

    # Generate all subplot coordinates (Cartesian product)
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
        X['age_squared'] = X['age']**2
        df["Intercept"] = 1

        effSize = []
        pvalues = []
        norm_values = []
        
        # Loop over each dependent variable
        for dv_measure in dv:
            
            Y = df[dv_measure].copy()
            model = sm.MixedLM(
                        endog=Y,
                        exog=X,
                        groups=df[random_effects[0]],
                        exog_re=df[['Intercept', random_effects[1]]].copy()
                    )
            results = model.fit()
            residuals = results.resid / np.sqrt(results.scale)
            _,p_shapiro = stats.shapiro(residuals)

            if p_shapiro >= 0.05:
                
                beta = results.params[var_of_interest]
                pval = results.pvalues[var_of_interest]

                # Compute standardized beta
                predictor_std = X[var_of_interest].std()
                residual_std = np.sqrt(results.scale)
                standardized_beta = beta * (predictor_std / residual_std)
                norm_values.append('True')

           
            elif dv_measure =='Accuracy':
                
                df_sub = X.copy()
                df_sub = df_sub.rename(columns={"NIHSS at admission or 2 hours after thrombectomy/thrombolysis": "NIHSS"})
                df_sub['ID'] = df['ID'].astype(int)
                df_sub[dv_measure] = df[dv_measure].astype(int)
                df_sub['number_trials'] = df[dv_measure].max().astype(int)
                
                df_sub = (
                    df_sub
                    .assign(
                        outcome=[
                            [1]*r + [0]*(n - r)
                            for r, n in zip(df_sub[dv_measure], df_sub["number_trials"])
                        ]
                    )
                    .explode("outcome", ignore_index=True)
                )            
                
                df_sub = pl.from_pandas(df_sub)  
    
                model = glmer("outcome ~ age + age_squared + gender + english_secondLanguage + education_Alevels + education_bachelors + education_postBachelors + NIHSS + impaired_hand + timepoint + (1 + timepoint | ID)", data=df_sub, family="binomial")
                model.set_factors('ID')
                model.fit(summary=True)
                
                fe = model.fixef
                sd = df_sub[var_of_interest].std(ddof=1)
                standardized_beta = fe[var_of_interest] * sd
                standardized_beta = standardized_beta[0]
                pval = (
                    model.result_fit
                    .filter(pl.col('term') == var_of_interest)
                    .select('p_value')
                    .item()
                ) 
                norm_values.append('False')
                
            else:
                        
                pt = PowerTransformer(method="yeo-johnson", standardize=True)  
                df[dv_measure] = pt.fit_transform(df[[dv_measure]])    
                Y = df[dv_measure].copy()
                model = sm.MixedLM(
                        endog=Y,
                        exog=X,
                        groups=df[random_effects[0]],
                        exog_re=df[['Intercept', random_effects[1]]].copy()
                    )
                results = model.fit()
                beta = results.params[var_of_interest]
                pval = results.pvalues[var_of_interest]

                # Compute standardized beta
                predictor_std = X[var_of_interest].std()
                residual_std = np.sqrt(results.scale)
                standardized_beta = beta * (predictor_std / residual_std)
                norm_values.append('True')
            
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
        ax_sub.set_ylabel('Standardised beta coefficient', fontsize=16)

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
            ax_sub.set_ylabel('Regression coefficient', fontsize=16)
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
            
            ax_sub.text(
                x=bar.get_x() + bar.get_width() / 2,
                y=-0.05,
                s='Binomial' if (idx ==0) & (norm_values[idx]=='False') else 'Linear',
                ha="center",
                fontsize=14,
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

