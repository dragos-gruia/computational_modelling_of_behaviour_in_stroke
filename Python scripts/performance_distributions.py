
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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

def plot_group_distributions(path_files, output_path, demographic_file, column_mapping=None):
    
    """
    Plots and saves a bespoke multi-panel figure of task-based distributions.

    Parameters
    ----------
    path_files : list of str
        List of file paths for task outcome measures (CSV or Excel).
    output_path : str
        Directory where the figure ('task_distributions.png') will be saved.
    demographic_file : str
        Filename for the demographic data (CSV or Excel). The file is expected to be located in 
        '../trial_data/' relative to each task file.
    column_mapping : dict, optional
        Mapping from task names to descriptive labels for the y-axis in the first column.
    
    Raises
    ------
    FileNotFoundError
        If any file cannot be found.
    ValueError
        If an unsupported file extension is encountered.
    
    Notes
    -----
    - The function internally calls `create_unique_ticks` to define unique x-tick 
      locations for each histogram based on the histogram's bin edges.
    - If the task name is not `IC3_rs_SRT` or `IC3_NVtrailMaking`, the script assigns 
      the primary outcome to `df['Accuracy']`.
    - The merged dataframe expects a common key named `user_id`.
    - The figure size scales with the number of rows (i.e. tasks) but can be further adjusted if necessary

    Examples
    --------
    >>> plot_group_distributions(
    ...     path_files=['/path/to/task1_outcomes.csv', '/path/to/task2_outcomes.csv'],
    ...     output_path='/path/to/output',
    ...     demographic_file='demographics.csv',
    ...     column_mapping={'task1': 'Task 1 Label', 'task2': 'Task 2 Label'}
    ... )
    """
    
    n_tasks = len(path_files)
    fig, axes = plt.subplots(n_tasks, 4, figsize=(30, 5 * n_tasks))
    # Ensure axes is 2D even if there is one task
    if n_tasks == 1:
        axes = np.expand_dims(axes, axis=0)

    # Column titles for the top row
    col_titles = ["Standard Accuracy", "Modelled Cognitive Index", "Median Reaction Time", "Response Delay Time"]

    for i, task_path in enumerate(path_files):
        # Load task data (supporting CSV or Excel)
        ext_task = os.path.splitext(task_path)[1].lower()
        if ext_task == '.csv':
            df = pd.read_csv(task_path)
        elif ext_task in ['.xls', '.xlsx']:
            df = pd.read_excel(task_path)
        else:
            raise ValueError(f"Unsupported task file format: {task_path}")

        df = df.replace([np.inf, -np.inf], np.nan).copy()

        # Extract task name from the filename
        task_name = os.path.basename(task_path).split('_outcomes')[0]

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
        if task_name not in ['IC3_rs_SRT', 'IC3_NVtrailMaking']:
            df['Accuracy'] = df[task_name]

        # Determine ylabel for the first column using column_mapping if provided
        ylabel_col0 = column_mapping.get(task_name, task_name) if column_mapping else task_name
        is_top_row = (i == 0)

        # Plot the four metrics
        plot_histogram(axes[i, 0], df, "Accuracy", col_titles[0] if is_top_row else "", ylabel_col0, "#ffcc00", is_top_row)
        plot_histogram(axes[i, 1], df, "AS_scaled", col_titles[1] if is_top_row else "", "", "#62428a", is_top_row)
        plot_histogram(axes[i, 2], df, "MedianRT", col_titles[2] if is_top_row else "", "", "#ffcc00", is_top_row)
        plot_histogram(axes[i, 3], df, "DT_scaled", col_titles[3] if is_top_row else "", "", "#62428a", is_top_row)

    fig.savefig(os.path.join(output_path, "task_distributions.png"), format="png", transparent=False)
    
    
if __name__ == "__main__":
    plot_group_distributions()