

"""
================================================================================
Author: Dragos-Cristian Gruia
Last Modified: 14/03/2025
================================================================================
"""


from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def plot_group_distributions(root_path, task_names, output_path, column_mapping=None, demographic_file='patient_data_cleaned_linked.xlsx', file_extention='_outcomes.csv'):
    
    """
    Plots and saves a bespoke multi-panel figure of task-based distributions.

    This function generates a multi-panel figure displaying histograms for four different metrics 
    (Standard Accuracy, Modelled Cognitive Index, Median Reaction Time, and Response Delay Time) for each 
    specified task. For each task, the function constructs the file path by joining a root directory, a task name 
    (from `task_names`), and a file extension (`file_extention`). It then loads the task data (supporting CSV and Excel formats), 
    merges it with demographic data (loaded from a file in '../trial_data/'), and produces histograms for selected columns. 
    The function customizes the appearance of each subplot, and assigns descriptive labels using an optional `column_mapping`.

    Parameters
    ----------
    root_path : str
        The directory path where the task outcome files are stored. Each task file is constructed by joining this path with 
        a task name (from `task_names`) appended with `file_extention`.
    task_names : list of str
        A list of task name strings (without file extensions). Each task name is appended with `file_extention` and joined with 
        `root_path` to form the complete file path for the task outcome data.
    output_path : str
        Directory where the resulting figure ('task_distributions.png') will be saved.
    column_mapping : dict, optional
        A dictionary mapping task names to descriptive labels for the y-axis in the first column. If provided, these labels 
        override the default task name.
    demographic_file : str, optional
        Filename for the demographic data (CSV or Excel). This file is expected to be located in the '../trial_data/' 
        directory relative to each task file. Default is 'patient_data_cleaned_linked.xlsx'.
    file_extention : str, optional
        Suffix to append to each task name to form the full file name of the task outcomes. Default is '_outcomes.csv'.

    Returns
    -------
    None
        The function saves a figure named 'task_distributions.png' in the specified `output_path` and does not return any value.

    Raises
    ------
    FileNotFoundError
        If any of the task or demographic files cannot be found.
    ValueError
        If an unsupported file extension is encountered for either the task data or the demographic data.
    
    Notes
    -----
    - Task data can be in CSV or Excel format; demographic data is similarly supported.
    - Demographic data is merged with the task data on the common key 'user_id'.
    - The function calls `create_unique_ticks` to generate evenly spaced x-tick positions for each histogram based on its bin edges.
    - The figure is arranged with one row per task and four columns corresponding to the four metrics, with column titles 
      applied only to the top row.
    - The figure size scales with the number of tasks.

    Examples
    --------
    >>> plot_group_distributions(
    ...     root_path='/path/to/task_files',
    ...     task_names=['task1', 'task2'],
    ...     output_path='/path/to/output',
    ...     demographic_file='demographics.csv',
    ...     column_mapping={'task1': 'Task 1 Label', 'task2': 'Task 2 Label'}
    ... )
    """
    
    
    n_tasks = len(task_names)
    fig, axes = plt.subplots(n_tasks, 4, figsize=(30, 5 * n_tasks))
    # Ensure axes is 2D even if there is one task
    if n_tasks == 1:
        axes = np.expand_dims(axes, axis=0)

    # Column titles for the top row
    col_titles = ["Standard Accuracy", "Modelled Cognitive Index", "Median Reaction Time", "Response Delay Time"]

    for i, task in enumerate(task_names):
        
        task_path = os.path.join(root_path, task + file_extention)
        
        # Load task data (supporting CSV or Excel)
        ext_task = os.path.splitext(task_path)[1].lower()
        if ext_task == '.csv':
            df = pd.read_csv(task_path)
        elif ext_task in ['.xls', '.xlsx']:
            df = pd.read_excel(task_path)
        else:
            raise ValueError(f"Unsupported task file format: {task_path}")

        df = df.replace([np.inf, -np.inf], np.nan).copy()

        # Construct the demographic file path relative to the task file directory
        task_dir = os.path.dirname(task_path)
        demo_path = os.path.join(task_dir, '..', 'trial_data', demographic_file)
        
        ext_demo = os.path.splitext(demographic_file)[1].lower()
        if ext_demo == '.csv':
            df_dem = pd.read_csv(demo_path)
        elif ext_demo in ['.xls', '.xlsx']:
            df_dem = pd.read_excel(demo_path)
        else:
            raise ValueError("Unsupported demographic file format: " + demographic_file)

        # Merge task data with demographic data on 'user_id'
        df = df.merge(df_dem, how='left', on='user_id')

        # If task is not in the exceptions, assign primary outcome to 'Accuracy'
        if task not in ['IC3_rs_SRT', 'IC3_NVtrailMaking']:
            df['Accuracy'] = df[task]

        # Determine ylabel for the first column using column_mapping if provided
        ylabel_col0 = column_mapping.get(task, task) if column_mapping else task
        is_top_row = (i == 0)

        # Plot the four metrics
        plot_histogram(axes[i, 0], df, "Accuracy", col_titles[0] if is_top_row else "", ylabel_col0, "#ffcc00", is_top_row)
        plot_histogram(axes[i, 1], df, "AS_scaled", col_titles[1] if is_top_row else "", "", "#62428a", is_top_row)
        plot_histogram(axes[i, 2], df, "MedianRT", col_titles[2] if is_top_row else "", "", "#ffcc00", is_top_row)
        plot_histogram(axes[i, 3], df, "DT_scaled", col_titles[3] if is_top_row else "", "", "#62428a", is_top_row)

    print('Group distributions were created')
    fig.savefig(os.path.join(output_path, "task_distributions.png"), format="png", transparent=False)
    

def create_unique_ticks(ax, nticks=5):
    
    """
    Generate a specified number of evenly spaced tick positions based on histogram bin edges.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object containing the histogram.
    nticks : int, optional
        Number of tick positions to generate. Defaults to 5.

    Returns
    -------
    numpy.ndarray
        Array of tick positions.
    """
    
    patches = ax.patches
    if not patches:
        return np.array([])
    x_min = min(p.get_x() for p in patches)
    x_max = max(p.get_x() + p.get_width() for p in patches)
    return np.linspace(x_min, x_max, nticks)

def plot_histogram(ax, df, column, title, ylabel, color, is_top_row):
    
    """
    Plot a histogram on the provided axis with custom styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The subplot on which to plot.
    df : pandas.DataFrame
        DataFrame containing the data.
    column : str
        Column name to plot.
    title : str
        Title to display (only on the top row).
    ylabel : str
        Y-axis label.
    color : str
        Color of the histogram.
    is_top_row : bool
        Whether this subplot is in the top row.
    """
    
    sns.histplot(data=df, x=column, edgecolor='white', color=color, ax=ax)
    if is_top_row:
        ax.set_title(title, fontsize=22)
    else:
        ax.set_title("", fontsize=20)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=20 if is_top_row else 16)
    ax.set_xticks(create_unique_ticks(ax))
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)    
    
    
if __name__ == "__main__":
    plot_group_distributions()