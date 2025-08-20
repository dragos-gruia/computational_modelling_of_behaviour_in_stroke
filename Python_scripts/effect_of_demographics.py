

"""
================================================================================
Author: Dragos-Cristian Gruia
Last Modified: 20/08/2025
================================================================================
"""


import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm
import scipy.stats as stats
from pymer4.models import glmer
from rpy2.robjects import pandas2ri
import polars as pl
from sklearn.preprocessing import PowerTransformer
from calculate_motor_impairment import *


def plot_group_regression_weights_comparison(
    path_files, 
    output_path,
    dv,
    demographic_file,
    column_mapping=None,
    random_effects=['ID','timepoint'],
    independent_vars = [
        'age','gender','english_secondLanguage','education_Alevels',
        'education_bachelors','education_postBachelors', 
        'NIHSS at admission or 2 hours after thrombectomy/thrombolysis', 'timepoint'
    ],
    x_labels = [
        'Age', 'Gender', 'English - second language',
        'Education - A levels','Education - Bachelors',
        'Education - Postgraduate', 'NIHSS Score', 'Timepoint'
    ]
):
    
    """
    Generates a grid of subplots comparing mixed effects regression coefficients for two dependent variables
    across multiple tasks.

    This function constructs file paths for task outcome CSV files by combining a root directory, a task name,
    and a file extension. Each CSV file is merged with demographic data and then used to fit regression models 
    for two dependent variables. For each task, the function plots two sets of regression coefficients
    (with corresponding confidence intervals) side by side with a slight vertical offset.

    Parameters
    ----------
    path_files : list of str
        List of file paths containing outcome measures (CSV format).
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
        variable (e.g., 'ID') and the second as the random slope (e.g., 'timepoint').
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

    Returns
    -------
    None
        Saves a multi-subplot figure named 'demographic_effects.png' in the specified `output_path` directory. 
        No value is returned. 
    """
     
    fig, ax = plt.subplots(int(len(path_files)/3), 3, figsize=(25, 8*len(path_files)/3))  # Adjust the figure size if necessary
    fig.subplots_adjust(hspace=0.3)
        
    # Define the ranges for the two values
    first_range = range(6)  # 0 to 5
    second_range = range(3) # 0 to 2
    
    # Generate all possible combinations (Cartesian product)
    subplot_coordinates = list(itertools.product(first_range, second_range))

    i=0
    for task_path in path_files:
        
        panel_number= subplot_coordinates[i]
        
        # Load task data
        df = pd.read_csv(task_path)
        df = df.reset_index(drop=True)
        task_name = task_path.split('/')[-1].split('_outcomes')[0]
        
        # Merge demographic characteristics with the AS and DT metrics
        os.chdir('/'.join(task_path.split('/')[0:-1]))
        
        if demographic_file.split('.')[-1] == 'csv':
            df_dem = pd.read_csv(f'../trial_data/{demographic_file}')
        elif demographic_file.split('.')[-1] == 'xlsx':
            df_dem = pd.read_excel(f'../trial_data/{demographic_file}')
        else:
            print('Incorrect demgographic file names')
            return 0
        
        df = df.merge(df_dem, how='left', on='user_id')
        
        # Assign raw primary outcome to 'Accuracy' column
        if 'NIHSS at admission or 2 hours after thrombectomy/thrombolysis' in df.columns:
            df.loc[:,'NIHSS at admission or 2 hours after thrombectomy/thrombolysis'] = np.log(df['NIHSS at admission or 2 hours after thrombectomy/thrombolysis']+1)
            
        if (task_name != 'IC3_rs_SRT') & (task_name !='IC3_NVtrailMaking'):
            df['Accuracy'] = df[task_name]
        
        outputs = []    
        df['age'] = (df['age'] - df['age'].mean())/df['age'].std()
        df = df.dropna(subset=dv).reset_index(drop=True)
        df = df.dropna(subset=independent_vars).reset_index(drop=True)
        df["Intercept"] = 1
        
        X = df.loc[:,independent_vars]
        X.insert(
            loc=1,
            column='age_squared',
            value = X['age']**2    
        )
        
        # Run regression and check for residual distribution
        Y = df[[dv[0]]].copy()
        model = sm.MixedLM(
                        endog=Y,
                        exog=X,
                        groups=df[random_effects[0]],
                        exog_re=df[['Intercept', random_effects[1]]].copy()
                    )
        results = model.fit()
        residuals = results.resid / np.sqrt(results.scale)
        _,p_shapiro = stats.shapiro(residuals)

        if p_shapiro < 0.05:
                
            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            Y = pt.fit_transform(df[[dv[0]]])    
            model = sm.MixedLM(
                endog=Y,
                exog=X,
                groups=df[random_effects[0]],
                exog_re=df[['Intercept', random_effects[1]]].copy()
            )        
                
        results = model.fit()
        predictors = X.columns
        
        for predictor_name in predictors:
            
            coefficient = results.params[predictor_name]
            standardized_beta = coefficient

            # Calculate confidence intervals
            CI = results.conf_int().loc[predictor_name,:]
            ci_lower = (CI[0])
            ci_upper = (CI[1])
            
            outputs.append((predictor_name, coefficient, standardized_beta, ci_lower,ci_upper))
            
            
        results_df_AS = pd.DataFrame(outputs, columns=['predictor', 'coefficient', 'std_coeff','ci_lower','ci_upper'])
        results_df_AS = results_df_AS.drop(results_df_AS[results_df_AS['predictor'] == 'const'].index)
          
        # Linear regression model for Accuracy
        outputs = []
        X = df.loc[:,independent_vars]
        X.insert(
            loc=1,
            column='age_squared',
            value = X['age']**2    
        )

        Y = df[[dv[1]]].copy()
        Y = (Y - Y.mean())/Y.std()
        
        model = sm.MixedLM(
                        endog=Y,
                        exog=X,
                        groups=df[random_effects[0]],
                        exog_re=df[['Intercept', random_effects[1]]].copy()
                    )   
        results = model.fit()
        residuals = results.resid / np.sqrt(results.scale)
        _,p_shapiro = stats.shapiro(residuals)

        if p_shapiro < 0.05:
                  
            df_sub = X.copy()
            df_sub = df_sub.rename(columns={"NIHSS at admission or 2 hours after thrombectomy/thrombolysis": "NIHSS"})
            predictors = df_sub.columns

            df_sub['ID'] = df['ID'].astype(int)
            df_sub[dv[1]] = df[dv[1]].astype(int)
            df_sub['number_trials'] = df[dv[1]].max().astype(int)
            df_sub = (
                df_sub
                .assign(
                    outcome=[
                        [1]*r + [0]*(n - r)
                        for r, n in zip(df_sub[dv[1]], df_sub["number_trials"])
                    ]
                )
                .explode("outcome", ignore_index=True)
            )            
            df_sub = pl.from_pandas(df_sub)  

            model = glmer("outcome ~ age + age_squared + gender + english_secondLanguage + education_Alevels + education_bachelors + education_postBachelors + NIHSS + timepoint + (1 + timepoint | ID)", data=df_sub, family="binomial")
            model.set_factors('ID')  
            model.fit(summary=True)
            fe = model.fixef

            for predictor_name in predictors:
                
                coefficient = fe[predictor_name][0]
                predictor_std = df_sub[predictor_name].std(ddof=1)
                standardized_beta = coefficient * predictor_std 
                
                # Calculate confidence intervals
                CI = model.result_fit
                ci_lower = (
                    CI
                    .filter(pl.col('term') == predictor_name)
                    .select('conf_low')
                    .item()
                ) * predictor_std                
                ci_upper = (
                    CI
                    .filter(pl.col('term') == predictor_name)
                    .select('conf_high')
                    .item()
                ) * predictor_std  
                
                outputs.append((predictor_name, coefficient, standardized_beta, ci_lower,ci_upper))
        
        else: 
            for predictor_name in predictors:
                coefficient = results.params[predictor_name]
                standardized_beta = coefficient

                # Calculate confidence intervals
                CI = results.conf_int().loc[predictor_name,:]
                ci_lower = (CI[0])
                ci_upper = (CI[1])
                
                outputs.append((predictor_name, coefficient, standardized_beta, ci_lower,ci_upper))

            
        results_df_raw = pd.DataFrame(outputs, columns=['predictor', 'coefficient', 'std_coeff','ci_lower','ci_upper'])
        results_df_raw = results_df_raw.drop(results_df_raw[results_df_raw['predictor'] == 'const'].index)
            
        # Plotting coefficients
        cap_width = 0.1
        vertical_gap = 0.2

        sn.barplot(y=results_df_AS['predictor'], x=results_df_AS['std_coeff'], orient='h',
                color='white', ax=ax[panel_number], capsize=0)

        sn.scatterplot(y=np.arange(results_df_AS.shape[0]),
                marker='o', s=200, ax =ax[panel_number],
                x=results_df_AS['std_coeff'], color='#62428a')
        
        for count, point in results_df_AS.iterrows():
            ax[panel_number].hlines(y=count, 
                    xmin=point['ci_lower'],
                    xmax=point['ci_upper'],
                    colors='#62428a',
                    linewidth=2)
            
            # Add small vertical caps at the ends of CI lines
            ax[panel_number].vlines(x=point['ci_lower'], 
                    ymin=count - cap_width/2,
                    ymax=count + cap_width/2,
                    colors='#62428a',
                    linewidth=2)
            ax[panel_number].vlines(x=point['ci_upper'], 
                    ymin=count - cap_width/2,
                    ymax=count + cap_width/2,
                    colors='#62428a',
                    linewidth=2)
            

        sn.scatterplot(y=np.arange(results_df_raw.shape[0])+vertical_gap,
                marker='o', s=200, ax =ax[panel_number],
                x=results_df_raw['std_coeff'], color='#ffcc00')
        
        for count, point in results_df_raw.iterrows():
            ax[panel_number].hlines(y=count + vertical_gap, 
                    xmin=point['ci_lower'],
                    xmax=point['ci_upper'],
                    colors='#ffcc00',
                    linewidth=2)
            
            # Add small vertical caps at the ends of CI lines
            ax[panel_number].vlines(x=point['ci_lower'], 
                    ymin=count - cap_width/2 + vertical_gap,
                    ymax=count + cap_width/2 + vertical_gap,
                    colors='#ffcc00',
                    linewidth=2)
            ax[panel_number].vlines(x=point['ci_upper'], 
                    ymin=count- cap_width/2 + vertical_gap,
                    ymax=count + cap_width/2 + vertical_gap,
                    colors='#ffcc00',
                    linewidth=2)              

        
        ax[panel_number].set_xlim(-2, 2)
        ax[panel_number].grid(True, linewidth=1)
        ax[panel_number].set_xticks([-4,-3,-2,-1, 0, 1, 2,3,4])
        ax[panel_number].set_xticklabels([-4,-3,-2,-1, 0, 1,2,3,4])
        ax[panel_number].axvline(x=0, linestyle='--', color='red', linewidth=1)
        ax[panel_number].set_xlabel('Regression Coefficients', fontsize=16)

        if panel_number[1] == 0:
            ax[panel_number].set_yticklabels(x_labels, fontsize=12)
            ax[panel_number].set_ylabel('')
        else:
            ax[panel_number].set_yticklabels('', fontsize=12)
            ax[panel_number].set_ylabel('')
        
        if column_mapping != None:
            ax[panel_number].text(0.5, 1.1, column_mapping[task_name], transform=ax[panel_number].transAxes, fontsize=20, va='top', ha='center', bbox = dict(boxstyle='round', facecolor = 'white'))
        else:
            ax[panel_number].text(-0.1, 1.35, task_name, transform=ax[panel_number].transAxes, fontsize=20, va='top', ha='center')
        
        i = i +1
                    
    fig.savefig(f'{output_path}/demographic_effects.png', format='png', transparent=False)

