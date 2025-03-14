
"""
================================================================================
Author: Dragos-Cristian Gruia
Last Modified: 14/03/2025
================================================================================
"""

import matplotlib.pyplot as plt
import seaborn as sn
import os
import pandas as pd

def create_group_difficulty_scale(root_path, task_names, output_path, column_mapping=None, sort_by_difficulty=True, file_extention='_trialDifficulty.csv'):
     
    """
    Plots and saves scatter plots of task-specific trial difficulties.

    This function reads one or more CSV files containing trial difficulty data – each file is expected to include 
    a 'DS' column representing a scaled measure of difficulty from the IDoCT model and an 'Original_difficulty' column describing the 
    difficulty label for each trial. The function loads the data for each task, and optionally sorts the data by the 'DS' column if `sort_by_difficulty` is True. 
    The function applies task-specific formatting to the 'Original_difficulty' labels via the `format_trial_difficulty` 
    helper function, and creates a scatter plot with 'Original_difficulty' on the x-axis and 'DS' on the y-axis.
    The resulting figure contains one subplot per task and is saved in the specified `output_path` as `task_trialDifficulty.svg`.

    Parameters
    ----------
    root_path : str
        The directory path where the task outcome CSV files are stored. Each task file is constructed by joining 
        this path with a task name (from `task_names`) appended with `file_extention`.
    task_names : list of str
        A list of task name strings (without file extensions). Each task name is appended with `file_extention` and 
        joined with `root_path` to form the full path to the CSV file containing trial difficulty data.
    output_path : str
        The directory path where the generated figure (`task_trialDifficulty.svg`) will be saved.
    column_mapping : dict, optional
        A mapping from task names (extracted from the file name) to descriptive titles. For example, 
        `{"IC3_Comprehension": "Comprehension Task"}`. If provided and a key matching the task name is found, 
        the subplot title will use the descriptive title instead of the raw task name. Defaults to None.
    sort_by_difficulty : bool, optional
        If True, the rows in each CSV file are sorted by the 'DS' column before plotting; if False, the original 
        order of trials is retained. Defaults to True.
    file_extention : str, optional
        The file extension to append to each task name to form the complete file name. Defaults to '_trialDifficulty.csv'.

    Returns
    -------
    None
        The function does not return any value; it saves a multi-subplot figure to `output_path`.

    Raises
    ------
    FileNotFoundError
        If any of the task files (constructed from `root_path` and a task name) cannot be found.
    ValueError
        If a CSV file does not contain the expected columns ('DS', 'Original_difficulty').

    Notes
    -----
    - Each subplot’s x-axis labels are rotated 90 degrees for readability, and the y-axis displays the scaled 
      difficulty values from 'DS'.
    - The figure size is proportional to the number of tasks (i.e., one subplot per task), and subplots are 
      vertically spaced.

    Examples
    --------
    >>> # Suppose we have two trial difficulty files: 'IC3_Comprehension_trialDifficulty.csv'
    >>> # and 'IC3_rs_PAL_trialDifficulty.csv', located in a directory:
    >>> root_path = "/path/to/task_files"
    >>> task_names = ["IC3_Comprehension", "IC3_rs_PAL"]
    >>> output_path = "/path/to/output"
    >>> column_mapping = {
    ...     "IC3_Comprehension": "Comprehension Task",
    ...     "IC3_rs_PAL": "Paired Association Learning Task"
    ... }
    >>> create_group_difficulty_scale(
    ...     root_path=root_path,
    ...     task_names=task_names,
    ...     output_path=output_path,
    ...     column_mapping=column_mapping,
    ...     sort_by_difficulty=True
    ... )
    """
    
    
    fig, axes = plt.subplots(len(task_names), 1, figsize=(30, 15*len(task_names)))  # Adjust the figure size if necessary
    fig.subplots_adjust(hspace=1.5)
        
    panel_number = 0
    
    for task in task_names:
        
        # Load task data
        
        task_path = os.path.join(root_path, task + file_extention)

        df = pd.read_csv(task_path)
        df = df.replace([float('inf'), -float('inf')], float('nan'))
        
        task_name = task_path.split('/')[-1].split('_trialDifficulty')[0]
        os.chdir('/'.join(task_path.split('/')[0:-1]))        
        
        # Sort by trial difficulty
        
        if sort_by_difficulty:
            df = df.sort_values(by='DS')
          
        # Format the way difficulty is presented for a subset of tasks
    
        df = format_trial_difficulty(df, task_name)
        
        # Plot figures
        
        sn.scatterplot(data=df, x='Original_difficulty', y='DS', marker='o', s=800, color = '#62428a', ax=axes[panel_number])
        
        axes[panel_number].tick_params(axis='x', labelrotation = 90, labelsize=30)
        axes[panel_number].set_xlabel("")  
        axes[panel_number].set_ylabel('Scaled Difficulty', fontsize=30)    
        axes[panel_number].set_ylim(df['DS'].min()-0.05,df['DS'].max()+0.05)   
        axes[panel_number].grid(linestyle='--', linewidth=1.5)

        if column_mapping != None:
            axes[panel_number].set_title(column_mapping[task_name], fontsize=40)      
        else:
            axes[panel_number].set_title(task_name, fontsize=40)       


        panel_number = panel_number + 1
        
    # Save figures
    print('Task difficulty plots were created.')
    fig.savefig(f'{output_path}/task_trialDifficulty.svg', format='svg', transparent=False)


def format_trial_difficulty(df, task_name):
    
    """
    Formats the 'Original_difficulty' column for a specific task.

    This function modifies or groups the 'Original_difficulty' column in the
    input DataFrame (`df`) depending on the task name. Some tasks simply sort
    by numeric difficulty levels, while others re-organise difficulty labels or
    group trials into categories (e.g., short vs. complex sentences).

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing at least:
            - 'DS' (float): A scaled difficulty metric.
            - 'Original_difficulty' (str): A descriptor for each trial's difficulty.
        Other columns may be present but are not modified unless specified by
        task-specific logic.
    task_name : str
        The identifier for the task. Various transformations are triggered
        depending on the value of `task_name`. Some recognized tasks include:
        'IC3_Orientation', 'IC3_Comprehension', 'IC3_rs_PAL', 'IC3_rs_digitSpan',
        'IC3_i4i_IDED', etc.

    Returns
    -------
    df : pandas.DataFrame
        The input DataFrame with an updated (or grouped/sorted) 'Original_difficulty'
        column. For certain tasks, the rows may be aggregated or re-sorted.

    Raises
    ------
    KeyError
        If the 'Original_difficulty' column is not present in `df` before processing.
    TypeError
        If the type of 'Original_difficulty' is incompatible with the transformations
        performed (e.g., attempting to cast non-numeric strings to int).

    Notes
    -----
    - Depending on `task_name`, this function may group entries by 'Original_difficulty'
      and average 'DS' (e.g., in 'IC3_Comprehension'), replace label strings, or transform
      numeric codes into descriptive text.
    - If `task_name` does not match a recognized case, the function will leave the
      DataFrame unchanged.
    
    """
            
    if (task_name == 'IC3_Orientation') | (task_name == 'IC3_TaskRecall'):
        df['Original_difficulty'] = (df.index +1).astype(str)
        df['Original_difficulty'] = 'Question ' + df['Original_difficulty']
        df = df.sort_values(by='Original_difficulty')
        
    elif task_name == 'IC3_rs_PAL':
        df = df.sort_values(by='Original_difficulty')
        df['Original_difficulty'] = df['Original_difficulty'].astype(int).astype(str)
        df['Original_difficulty'] = df['Original_difficulty'] + ' Associations Recalled'
        
    elif task_name == 'IC3_rs_digitSpan':
        df = df.sort_values(by='Original_difficulty')
        df['Original_difficulty'] = df['Original_difficulty'].astype(int).astype(str)
        df['Original_difficulty'] = df['Original_difficulty'] + ' Numbers Recalled'
        
    elif task_name == 'IC3_rs_spatialSpan':
        df = df.sort_values(by='Original_difficulty')
        df['Original_difficulty'] = df['Original_difficulty'].astype(int).astype(str)
        df['Original_difficulty'] = df['Original_difficulty'] + ' Positions Recalled'
        
    elif task_name == 'IC3_Comprehension':
        
        single_word = (df['Original_difficulty'].apply(lambda x: x.count(' '))==0)
        relational_sentences = (df['Original_difficulty'].apply(lambda x: x.count(' '))<6) & (df['Original_difficulty'].apply(lambda x: x.count(' '))>0)
        complex_sentences = (df['Original_difficulty'].apply(lambda x: x.count(' '))>=6)
        
        df.loc[single_word,'Original_difficulty'] = 'Single Word'
        df.loc[complex_sentences,'Original_difficulty'] = 'Gramatically complex sentenses'
        df.loc[relational_sentences,'Original_difficulty'] = 'Short relational sentenses'
        df = df.groupby('Original_difficulty').agg({'DS':'mean'})
        df = df.sort_values(by='DS')
        
    elif task_name == 'IC3_i4i_IDED':
        df['Original_difficulty'] = df['Original_difficulty'].replace({'Exploitation 1D': 'Exploitation 1 (Rule known)', 'Exploration 1D':'Exploration 1 (Rule unknown)', 'Exploitation ID reversal 1': 'Exploitation 2 (Rule known)', 
                                            'Exploration ID reversal 1':'Exploration 2 (Rule unknown)', 'Exploitation 2D':'Exploitation 3 (Rule known)', 'Exploration 2D':'Exploration 3 (Rule unknown)',
                                            'Exploitation 2D overlapped':'Exploitation 4 (Rule known)','Exploration 2D overlapped':'Exploration 4 (Rule unknown)', 'Exploitation ID reversal 2':'Exploitation 5 (Rule known)',
                                            'Exploration ID reversal 2':'Exploration 5 (Rule unknown)', 'Exploitation ID shift':'Exploitation 6 (Rule known)','Exploration ID shift':'Exploration 6 (Rule unknown)',
                                            'Exploitation ID reversal 3':'Exploitation 7 (Rule known)','Exploration ID reversal 3':'Exploration 7 (Rule unknown)',
                                            'Exploitation ED shift (to lines)':'Exploitation 8 (Rule known)','Exploration ED shift (to lines)':'Exploration 8 (Rule unknown)', 'Exploitation ID reversal (to lines)':'Exploitation 9 (Rule known)','Exploration ED shift (to lines)':'Exploration 9 (Rule unknown)'})

    elif task_name == 'IC3_PearCancellation':
        df['Original_difficulty'] = df['Original_difficulty'].str.replace('Square Location', 'Position', regex=True)
        
    elif task_name == 'IC3_rs_SRT':
        df['Original_difficulty'] = 'ISI ' + df['Original_difficulty']
        
    elif task_name == 'IC3_AuditorySustainedAttention':
        df['Original_difficulty'] = df['Original_difficulty'].str.replace('Up', 'Target 1', regex=True)
        df['Original_difficulty'] = df['Original_difficulty'].str.replace('Come', 'Target 2', regex=True)
        df['Original_difficulty'] = df['Original_difficulty'].str.replace('Hi', 'Target 3', regex=True)

    elif task_name == 'IC3_i4i_motorControl':
        df = df.sort_values(by='Original_difficulty')
        df['Original_difficulty'] = df['Original_difficulty'].astype(int).astype(str)
        df['Original_difficulty'] = 'Distance from target - ' + df['Original_difficulty']
        
    elif task_name == 'IC3_calculation':
        df['Original_difficulty'] = df['Original_difficulty'].str.replace('Sum', 'Addition -', regex=True)
        df['Original_difficulty'] = df['Original_difficulty'].str.replace('Subtract', 'Subtraction -', regex=True)
        
    elif task_name == 'IC3_GestureRecognition':
        df['Original_difficulty'] = df['Original_difficulty'].replace({'audio/IC3_gesture/bulbS.gif': 'Transitive Gesture 1','audio/IC3_gesture/comeOverS.gif':'Intransitive Gesture 1',
                                                                        'audio/IC3_gesture/drinkCupS.gif':'Transitive Gesture 2','audio/IC3_gesture/goodS.gif':'Intransitive Gesture 2',
                                                                        'audio/IC3_gesture/goodbyeS.gif':'Intransitive Gesture 3','audio/IC3_gesture/goodluckS.gif':'Intransitive Gesture 4',
                                                                        'audio/IC3_gesture/keysS.gif':'Transitive Gesture 3','audio/IC3_gesture/lighterS.gif':'Transitive Gesture 4'})
        
    return df