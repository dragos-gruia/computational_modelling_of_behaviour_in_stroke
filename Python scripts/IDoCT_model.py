

"""
Collection of functions used to estimate motor delay from cognitive tasks using
a fixed-point equation model.

The main workflow involves two phases:
1. Running the model on a control cohort (`run_idoct_model_controls`), deriving
   parameters like maximum AT (answer time) and maximum RT (reaction time).
2. Applying the derived model parameters to patient data (`apply_idoct_model_patients`)
   to compute the cognitive index (AS) and response delay time (DT).
   
The model has been developed by Valentina Giunchiglia and is presented in detail in the original publication:
 
Giunchiglia, V., Gruia, D.C., Lerede, A., Trender, W., Hellyer, P. and Hampshire, A., 2024. 
An iterative approach for estimating domain-specific cognitive abilities from large scale online cognitive data. 
NPJ Digital Medicine, 7(1), p.328.  

Here we adapt the use of the model so that it can applied in patient populations.
"""

import os
import pandas as pd
import numpy as np
from time import time

def main_wrapper(root_path_controls, root_path_patients, task_names, input_directory= 'idoct_input',  output_directory = 'idoct_output'):
    
    """
    Runs the iDoct model on control data, then applies it to patient data for each task.

    For each task in `file_names`, this function does the following:
    1. Constructs file paths for accuracy, RT (reaction time), and trial-definition CSVs.
    2. Runs the IDoCT model on the control cohort to compute maximum answer time (`at_max`)
       and maximum reaction time (`rt_max`).
    3. Applies the derived model to patient data, scaling the patient's outcome measures
       accordingly.

    Parameters
    ----------
    root_path_controls : str
        The directory where the control CSV files (accuracy, RT, trial definition) are located.
    root_path_patients : str
        The directory where the patients CSV files (accuracy, RT, trial definition) are located.
    task_names : list of str
        The list of task file prefixes (e.g., 'IC3_calculation', 'IC3_Comprehension'). For each task,
        the function will look for files named like `{task_name}_accuracy.csv`, `{task_name}_rt.csv`,
        and `{task_name}_trialdef.csv`.
    input_directory : str
        The directory containing patient CSV files (accuracy, RT, trial definition).
    output_directory : str
        The directory for model results (e.g., scaled outcomes and difficulty files).


    Returns
    -------
    None
        This function does not return any data structure; it prints a success message
        for each processed task and saves the outcomes to files in specified directories.

    Examples
    --------
    >>> root_path_controls = "/path/to/control_data"
    >>> root_path_patients = "/path/to/control_output"
    >>> task_names = ["IC3_calculation", "IC3_Comprehension"]
    >>> main_wrapper(root_path_controls, root_path_patients, task_names)
    """

    input_directory_controls = os.path.join(root_path_controls, input_directory)
    input_directory_patients = os.path.join(root_path_patients, input_directory)
    output_directory_controls = os.path.join(root_path_controls, output_directory)
    output_directory_patients = os.path.join(root_path_patients, output_directory)

    for i in range(len(task_names)):
        
        # Set up the input and run the idoct model on the control sample
        task_name = task_names[i]
        acc_file = f'{task_name}_accuracy.csv'
        rt_file = f'{task_name}_rt.csv'
        trialdef_file = f'{task_name}_trialdef.csv'
        
        df,df_difficulty,at_max,rt_max = run_idoct_model_controls(input_directory_controls, acc_file,rt_file,trialdef_file, task_name, output_directory_controls)
        
        # Set up the input and apply the idoct model on the patient sample
        difficulty_controls = f'{task_name}_trialDifficulty.csv'
        outcomes_controls = f'{task_name}_outcomes.csv'

        df_pats = apply_idoct_model_patients(output_directory_controls, input_directory_patients, acc_file,rt_file,trialdef_file, difficulty_controls, outcomes_controls, at_max, rt_max, task_name, output_directory_patients)

        print(f'Model has been successfully applied to {task_name}')


def run_idoct_model_controls(path, acc_file,rt_file,trialdef_file, task_name, output_directory):
    
    
    """
    Computes the iDoct model parameters for a control cohort and saves the results.

    This function reads accuracy, reaction time, and trial definition data for controls,
    encodes the trial definitions, and fits a fixed-point equation model to extract user
    ability (AS) and delay time (DT). It also estimates maximum answer time (AT) and maximum
    reaction time (RT) for subsequent use with patient data.

    Parameters
    ----------
    path : str
        Directory containing the control CSV files (`acc_file`, `rt_file`, `trialdef_file`).
    acc_file : str
        Filename of the accuracy CSV (e.g. 'IC3_calculation_accuracy.csv').
    rt_file : str
        Filename of the reaction time CSV (e.g. 'IC3_calculation_rt.csv').
    trialdef_file : str
        Filename of the trial definition CSV (e.g. 'IC3_calculation_trialdef.csv').
    task_name : str
        The name or prefix of the task (e.g. 'IC3_calculation').
    output_directory : str, optional
        The relative output folder where the results (`{task_name}_outcomes.csv`
        and `{task_name}_trialDifficulty.csv`) are saved. Defaults to 'idoct_output'.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing user-level metrics such as 'AS' (ability), 'DT' (delay time),
        'MedianRT', 'MeanRT', 'Accuracy', and scaled versions 'AS_scaled' and 'DT_scaled'.
    df_difficulty : pandas.DataFrame
        A DataFrame containing task item difficulties (`D`) and scaled difficulties (`DS`)
        for each unique trial definition (e.g., question/word).
    at_max : float
        The maximum answer time (AT) estimated from the model.
    rt_max : float
        The maximum reaction time (RT) observed for this control group.

    Notes
    -----
    - Output columns:
      - `AS`: user cognitive index
      - `DT`: user response delay time
      - `MedianRT`, `MeanRT`: median and mean reaction times across trials
      - `Accuracy`: total number of correct responses
      - `AS_scaled`, `DT_scaled`: standardized versions of `AS` and `DT`
      - `D`, `DS`: raw and scaled difficulty indices
      - `Original_difficulty`: the original label for each trial difficulty item

    Examples
    --------
    >>> df_out, df_diff, atmax, rtmax = run_idoct_model_controls(
    ...     path="/path/to/controls",
    ...     acc_file="IC3_calculation_accuracy.csv",
    ...     rt_file="IC3_calculation_rt.csv",
    ...     trialdef_file="IC3_calculation_trialdef.csv",
    ...     task_name="IC3_calculation"
    ... )
    """
    
    #Load the data
    os.chdir(path)
    df_acc = pd.read_csv(acc_file)
    df_rt = pd.read_csv(rt_file)
    df_trialdef = pd.read_csv(trialdef_file)
    
    user_ids = df_trialdef['user_id'].tolist()
    df_trialdef = df_trialdef.drop(columns=['user_id'])
    df_acc = df_acc.drop(columns=['user_id'])
    df_rt = df_rt.drop(columns=['user_id'])
    
    #Encode data
    lookupTable, int_word_mat = ordinal_encode(df_trialdef)
    rt_mat = df_rt.astype(float).values.copy()
    correct_mat = df_acc.astype(float).copy()
    word_mat = int_word_mat.astype(float).copy()
    #print(f'RT df shape {rt_mat.shape}, Accuracy df shape {correct_mat.shape}, Trial df shape {word_mat.shape}')
    
    #Run the model 
    (perf, word_complexity, complexity_mat, vocabulary, word_indexer) = word_complexities_from_rt(
    rt_mat, correct_mat, word_mat, niter=50
    )
    
    # Calculate ability and delay time 
    niter = 10
    rt_max = np.nanmax(rt_mat)
    abilities_0 = np.ones((rt_mat.shape[0],))
    dt_0 = np.nanmin(rt_mat, axis = 1)
    at_0 = rt_mat - dt_0[:, None]
    
    complexity_mat = complexity_mat.astype(float)
    correct_mat = correct_mat.astype(bool)
    perf_0 = ((1 - (at_0/rt_max) - (dt_0[:, None]/rt_max))*complexity_mat)*correct_mat
    for j in range(niter):
    
        true_ability = get_abilities(perf_0).to_numpy()
        
        # Get Answer time (AT) from formula
        at_new = (1 - true_ability[:, None])*(complexity_mat)*(rt_mat - dt_0[:, None])
        dt_new = rt_mat - at_new
        
        # Estimate Delay Time of a user as the avg of all DTs
        dt_newm = np.nanmean(dt_new, axis = 1)

        perf_new = (
            (1 - (at_new/rt_max) - (dt_newm[:, None]/rt_max)) *
            complexity_mat *
            correct_mat
        )
        d_perf = np.abs(perf_new - perf_0) / perf_0
        d_ab = np.abs(true_ability - abilities_0) / abilities_0
        
        abilities_0 = true_ability.copy()
        perf_0 = perf_new.copy()

    at_max = np.nanmax(at_new)
    #print(f'AT_max {at_max}, RT_max {rt_max}')
    perf_at = ((1 - (at_new/at_max))*complexity_mat)*correct_mat
    pa_ability = get_abilities(perf_at).to_numpy()
    
    word_complexities = word_complexity.squeeze()
    complexity_mat_3D = np.where(word_indexer, word_complexities[:, None, None], np.nan)
    maxtarget = np.nanmax(word_complexities)
    mintarget = np.nanmin(word_complexities)
    weighted_complexity_3D = complexity_mat_3D * (pa_ability)[None, :, None]
    weighted_complexities = np.nanmean(weighted_complexity_3D, axis = (1,2))
    weighted_complexities = weighted_complexities-np.nanmin(weighted_complexities)
    weighted_complexities = weighted_complexities/np.nanmax(weighted_complexities)
    weighted_complexities = weighted_complexities*(maxtarget-mintarget)
    weighted_complexities = weighted_complexities+mintarget
    
    df = pd.DataFrame({"user_id" : user_ids, "AS": pa_ability,"DT": dt_newm, 'MedianRT':np.nanmedian(df_rt, axis = 1), 'MeanRT':np.nanmean(df_rt, axis = 1), 'Accuracy': np.nansum(df_acc, axis = 1)}, index = df_trialdef.index)
    if len(lookupTable) > len(word_complexity):
        lookupTable = lookupTable[1:]
        
    df_difficulty = pd.DataFrame({ "D": word_complexity.squeeze(), 'DS': weighted_complexities, "Original_difficulty": lookupTable,}).reset_index()
    df['AS_scaled'] = (df['AS'] - df['AS'].mean())/df['AS'].std()
    df['DT_scaled'] = (df['DT'] - df['DT'].mean())/df['DT'].std()
    
    df.to_csv(f'{output_directory}/{task_name}_outcomes.csv')
    df_difficulty.to_csv(f'{output_directory}/{task_name}_trialDifficulty.csv')
    
    return df,df_difficulty,at_max,rt_max



def apply_idoct_model_patients(path_controls, path_patients, acc_file,rt_file,trialdef_file, difficulty_controls, outcomes_controls, at_max_controls, rt_max_controls, task_name, output_directory):

    """
    Applies a previously fitted iDoct model (from controls) to patient data, saving
    scaled outcome measures.

    This function loads task data for patients (accuracy, reaction time, trial definitions),
    and uses the difficulty scale and model outputs from the control data to compute
    patient-level ability (AS) and delay time (DT). The final patient dataset is saved with
    scaled metrics (`AS_scaled` and `DT_scaled`), anchored to the control distribution.

    Parameters
    ----------
    path_controls : str
        Directory containing the control model outputs (including the difficulty file 
        `difficulty_controls` and outcomes file `outcomes_controls`).
    path_patients : str
        Directory with patient CSV files (`acc_file`, `rt_file`, `trialdef_file`).
    acc_file : str
        Patient accuracy CSV filename (e.g. 'IC3_calculation_accuracy.csv').
    rt_file : str
        Patient reaction time CSV filename (e.g. 'IC3_calculation_rt.csv').
    trialdef_file : str
        Patient trial definition CSV filename (e.g. 'IC3_calculation_trialdef.csv').
    difficulty_controls : str
        Control difficulty CSV filename (e.g. 'IC3_calculation_trialDifficulty.csv') with
        columns "D" (raw difficulty) and "DS" (scaled).
    outcomes_controls : str
        Control outcomes CSV filename (e.g. 'IC3_calculation_outcomes.csv') with columns
        'AS' and 'DT' to standardize patient values.
    at_max_controls : float
        The maximum answer time (AT) derived from the control dataset.
    rt_max_controls : float
        The maximum reaction time (RT) from the control dataset.
    task_name : str
        The name/prefix of the task (e.g. 'IC3_calculation').
    output_directory : str, optional
        Directory where the resulting patient outcomes CSV is saved. Defaults to 'idoct_output'.

    Returns
    -------
    df : pandas.DataFrame
        The resulting patient data with columns: "AS", "DT", "MedianRT", "MeanRT", "Accuracy",
        "AS_scaled", "DT_scaled", and "user_id". Scaled metrics are anchored to the control
        data distribution.

    Notes
    -----
    - The iDoct approach is identical to the control workflow but uses `at_max_controls` and
      `rt_max_controls` to maintain consistency with the control group.
    - After computing raw ability/delay for patients, the final `AS_scaled` and `DT_scaled`
      columns are standardized using the control group's mean and standard deviation.

    Examples
    --------
    >>> df_pat = apply_idoct_model_patients(
    ...     path_controls="/path/to/control_outputs",
    ...     path_patients="/path/to/patient_data",
    ...     acc_file="IC3_calculation_accuracy.csv",
    ...     rt_file="IC3_calculation_rt.csv",
    ...     trialdef_file="IC3_calculation_trialdef.csv",
    ...     difficulty_controls="IC3_calculation_trialDifficulty.csv",
    ...     outcomes_controls="IC3_calculation_outcomes.csv",
    ...     at_max_controls=2.5,
    ...     rt_max_controls=4.0,
    ...     task_name="IC3_calculation"
    ... )
    """

    #Load the control data
    os.chdir(path_controls)
    D = pd.read_csv(difficulty_controls)
    df_controls = pd.read_csv(outcomes_controls)

    #Load the patient data
    os.chdir(path_patients)
    df_acc = pd.read_csv(acc_file)
    df_rt = pd.read_csv(rt_file)
    df_trialdef = pd.read_csv(trialdef_file)

    user_ids = df_trialdef['user_id'].tolist()
    df_trialdef = df_trialdef.drop(columns=['user_id'])
    df_acc = df_acc.drop(columns=['user_id'])
    df_rt = df_rt.drop(columns=['user_id'])

    #Encode data
    lookupTable, int_word_mat = ordinal_encode(df_trialdef)
    rt_mat = df_rt.astype(float).values.copy()
    correct_mat = df_acc.astype(float).copy()
    word_mat = int_word_mat.astype(float).copy()
    #print(f'RT df shape {rt_mat.shape}, Accuracy df shape {correct_mat.shape}, Trial df shape {word_mat.shape}')

    #Apply the model to patients and use the difficulty scales of controls
    word_complexity = D["D"].to_numpy()
    vocabulary = D["index"].to_numpy()

    vocabulary = vocabulary[:, None, None] # equivalent to np.expand_dims
    word_indexer = word_mat[None, :, :] == vocabulary#.squeeze()
    word_apprearance = np.sum(word_indexer, axis=(1,2))

    complexity_mat = np.sum(np.where(word_indexer, word_complexity[:, None, None], 0), axis=0)

    # Set the RT and AT max of controls
    at_max = at_max_controls
    rt_max= rt_max_controls

    # Calculate ability and delay time 
    niter = 10
    abilities_0 = np.ones((rt_mat.shape[0],))
    dt_0 = np.nanmin(rt_mat, axis = 1)
    at_0 = rt_mat - dt_0[:, None]
    complexity_mat = complexity_mat.astype(float)
    correct_mat = correct_mat.astype(bool)
    perf_0 = ((1 - (at_0/rt_max) - (dt_0[:, None]/rt_max))*complexity_mat)*correct_mat

    for j in range(niter):

        true_ability = get_abilities(perf_0).to_numpy()
        # Get Answer time (AT) from formula
        at_new = (1 - true_ability[:, None])*(complexity_mat)*(rt_mat - dt_0[:, None])
        dt_new = rt_mat - at_new
        # Estimate Delay Time of a user as the avg of all DTs
        dt_newm = np.nanmean(dt_new, axis = 1)
        perf_new = (
            (1 - (at_new/rt_max) - (dt_newm[:, None]/rt_max)) *
            complexity_mat *
            correct_mat
        )
        d_perf = np.abs(perf_new - perf_0) / perf_0
        d_ab = np.abs(true_ability - abilities_0) / abilities_0
        
        abilities_0 = true_ability.copy()
        perf_0 = perf_new.copy()

    perf_at = ((1 - (at_new/at_max))*complexity_mat)*correct_mat
    pa_ability = get_abilities(perf_at).to_numpy()


    df = pd.DataFrame({"user_id" : user_ids, "AS": pa_ability,"DT": dt_newm, 'MedianRT':np.nanmedian(df_rt, axis = 1), 'MeanRT':np.nanmean(df_rt, axis = 1), 'Accuracy': np.nansum(df_acc, axis = 1)}, index = df_trialdef.index)
    if len(lookupTable) > len(word_complexity):
        lookupTable = lookupTable[1:]
        
        
    df['AS_scaled'] = (df['AS'] - df_controls['AS'].mean())/df_controls['AS'].std()
    df['DT_scaled'] = (df['DT'] - df_controls['DT'].mean())/df_controls['DT'].std()
    
    df.to_csv(f'{output_directory}/{task_name}_outcomes.csv')

    return df


def word_complexities_from_rt(rt_mat, correct_mat, word_mat, niter = 200):
    
    """
    Computes word (or trial) complexities based on reaction times and correctness.

    This function initializes a complexity matrix and iteratively updates it using
    the `update_complexity_estimate` routine. The final complexities are used to
    interpret how 'difficult' each trial or item is, factoring in user correctness
    and reaction times.

    Parameters
    ----------
    rt_mat : numpy.ndarray
        2D array (user x trials) of reaction times.
    correct_mat : numpy.ndarray
        2D boolean array indicating correct responses (True) or incorrect/missing (False).
    word_mat : numpy.ndarray
        2D array indicating the discrete 'word'/item ID for each trial.
    niter : int, optional
        Number of iterations for refining the complexity estimates. Default is 200.

    Returns
    -------
    perf : numpy.ndarray
        Final performance matrix derived at the last iteration.
    word_complexity : numpy.ndarray
        1D array of word/item complexity scores corresponding to each unique 'word' index.
    complexity_mat : numpy.ndarray
        2D array of complexity values aligned with (user x trials).
    vocabulary : numpy.ndarray
        3D array representing unique item indices (expanded for matching shapes).
    word_indexer : numpy.ndarray
        Boolean 3D array (word, user, trial) used to map each 'word' to its occurrences.

    Notes
    -----
    - The approach is iterative, attempting to converge on stable complexity estimates for each item.
    - The `update_complexity_estimate` function handles each step of the iteration.

    Examples
    --------
    >>> perf, word_complexity, complexity_mat, vocab, indexer = word_complexities_from_rt(
    ...     rt_mat, correct_mat, word_mat, niter=50
    ... )
    """
    
    # Initialise empty matrices
    tau_max = np.nanmax(rt_mat)
    complexity_mat = np.ones_like(rt_mat)
    vocabulary = np.unique(word_mat)
    vocabulary = vocabulary[ ~np.isnan(vocabulary) ]
    
    # Get the 3D word indexer array
    vocabulary = vocabulary[:, None, None] # equivalent to np.expand_dims
    word_indexer = word_mat[None, :, :] == vocabulary
    word_apprearance = np.nansum(word_indexer, axis=(1,2))

    last_complexity_estimate = np.ones_like(vocabulary)
    perf_messages = []
    for i in range(niter):
        #print(i)
        t_start = time()
        (perf, rt_mat, tau_max, correct_mat, complexity_mat, 
        word_indexer, word_apprearance, 
        last_complexity_estimate, avg_delta_complexity) = update_complexity_estimate(
            rt_mat, tau_max, correct_mat, complexity_mat,
            word_indexer, word_apprearance, last_complexity_estimate
        )
        t = time() - t_start
        #print(f"[{i:4d}] avg(∂C) = {avg_delta_complexity:.15f}" 
        #                     f" | Elapsed: {t:.5f}s") 
    word_complexity = last_complexity_estimate
    return perf, word_complexity, complexity_mat, vocabulary, word_indexer


def update_complexity_estimate(rt_mat, tau_max, correct_mat, complexity_mat,
                               word_indexer, word_apprearance, last_complexity_estimate):
    
    """
    Single iteration of the word complexity estimation procedure.

    Parameters
    ----------
    rt_mat : numpy.ndarray
        2D array of reaction times (user x trials).
    tau_max : float
        The maximum reaction time observed in the dataset.
    correct_mat : numpy.ndarray
        2D boolean array indicating correctness of each trial.
    complexity_mat : numpy.ndarray
        2D array of current complexity estimates (user x trials).
    word_indexer : numpy.ndarray
        3D boolean array for indexing each word/item in each trial for each user.
    word_apprearance : numpy.ndarray
        1D array indicating how many times each word/item appears across all users/trials.
    last_complexity_estimate : numpy.ndarray
        3D array representing the previous iteration's complexity estimates for each word.

    Returns
    -------
    perf : numpy.ndarray
        Updated performance matrix after the current iteration.
    rt_mat : numpy.ndarray
        The original reaction times, returned for consistency.
    tau_max : float
        The maximum RT, returned for consistency.
    correct_mat : numpy.ndarray
        The original correctness matrix, returned for consistency.
    complexity_mat_new : numpy.ndarray
        The updated 2D complexity matrix (user x trials).
    word_indexer : numpy.ndarray
        The original 3D word indexer, returned for consistency.
    word_apprearance : numpy.ndarray
        The original array of word appearances, returned for consistency.
    word_complexity : numpy.ndarray
        New complexity estimates for each word, shaped as (word, 1, 1).
    avg_delta_complexity : float
        The average relative change in complexity across all words in this iteration.

    Notes
    -----
    - This function updates the `perf` matrix by computing performance as `(1 - rt_mat / tau_max)`.
    - Complexity values are refined by aggregating `(1 - perf)` for each word across users/trials,
      and normalizing by the word's total appearances.

    Examples
    --------
    Called within an iteration loop (e.g., in `word_complexities_from_rt`):
    >>> (perf, rt_mat, tau_max, correct_mat, complexity_mat,
    ...  word_indexer, word_apprearance, word_complexity, avg_delta_comp
    ... ) = update_complexity_estimate(
    ...         rt_mat, tau_max, correct_mat, complexity_mat,
    ...         word_indexer, word_apprearance, last_complexity_estimate
    ... )
    """
    
    perf = get_performance(rt_mat,tau_max,correct_mat,complexity_mat)
    complexity_mat_3D = np.where(word_indexer, 
                                 np.expand_dims((1-perf), 0),
                                 np.array([[[0]]]))
    
    # word_complexity = np.nansum(complexity_mat_3D, axis=(1,2)) / word_apprearance
    word_complexity = (nansum_on_first_second_axis(complexity_mat_3D) / 
                       word_apprearance)
    word_complexity = np.expand_dims(np.expand_dims(word_complexity, -1),-1)
    complexity_3D = np.where(word_indexer, word_complexity, 0)
    complexity_mat_new = nansum_on_zero_axis(complexity_3D)
    avg_delta_complexity = np.mean(
        np.abs(word_complexity - last_complexity_estimate) / 
        last_complexity_estimate
    )
    last_complexity_estimate = word_complexity
    return (perf, rt_mat, tau_max, correct_mat, complexity_mat_new, word_indexer,
            word_apprearance, word_complexity, avg_delta_complexity)
    


def get_performance(rt_mat,tau_max,correct_mat,complexity_mat):
    
    """
    Computes an initial performance metric as a function of reaction time, correctness,
    and complexity.

    The performance matrix is defined as:
    `perf = (1 - rt_mat / tau_max) * correct_mat * complexity_mat`

    Parameters
    ----------
    rt_mat : numpy.ndarray
        2D array of reaction times (user x trial).
    tau_max : float
        The maximum reaction time observed.
    correct_mat : numpy.ndarray
        2D boolean array indicating whether each response was correct.
    complexity_mat : numpy.ndarray
        2D array of current complexity estimates (user x trial).

    Returns
    -------
    perf : numpy.ndarray
        A 2D performance array (user x trial).

    Notes
    -----
    - If `rt_mat[i, j]` is large (close to `tau_max`), the `(1 - rt_mat/tau_max)`
      factor is small, reducing performance.
    - If `correct_mat[i, j]` is False, the performance is forced to 0 for that trial.

    Examples
    --------
    >>> perf_mat = get_performance(rt_mat, 5.0, correct_mat, complexity_mat)
    """
    
    return (1 - (rt_mat/tau_max)) * correct_mat * complexity_mat


def nansum_on_zero_axis(arr):
    
    """
    Aggregates (sums) values along axis 0 of a 3D array while ignoring NaNs.

    Parameters
    ----------
    arr : numpy.ndarray
        A 3D array. Summation is performed along the first dimension (index 0).

    Returns
    -------
    out_arr : numpy.ndarray
        A 2D array of shape (arr.shape[1], arr.shape[2]) containing the sum
        along axis 0, ignoring NaNs.
    """
    
    out_arr = np.zeros(arr.shape[1:])
    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            out_arr[i,j] = np.nansum(arr[:,i,j])
    return out_arr
    

def nansum_on_first_second_axis(arr):
    
    """
    Aggregates (sums) values along axes (1, 2) of a 3D array while ignoring NaNs.

    Parameters
    ----------
    arr : numpy.ndarray
        A 3D array. Summation is performed along the second and third dimensions
        (indices 1 and 2).

    Returns
    -------
    out_arr : numpy.ndarray
        A 1D array of shape (arr.shape[0],) containing the sum along axes (1, 2).
    """
    
    out_arr = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        out_arr[i] = np.nansum(arr[i,:,:])
    return out_arr

def get_abilities(perf):
    
    """
    Converts a performance matrix into a user-level ability score.

    For each user (row), the ability is computed as the sum of performance values
    across all trials (columns), divided by the number of valid (non-NaN) trials.

    Parameters
    ----------
    perf : numpy.ndarray
        2D array of shape (user x trial), representing performance scores.

    Returns
    -------
    ability : pandas.Series
        A series of user ability scores, with length equal to the number of users (rows).
    """
    
    non_na_values = ~np.isnan(perf)
    ability = np.nansum(perf, axis=1) / np.sum(non_na_values, axis=1)
    return ability


    
def ordinal_encode(str_mat):
    
    """
    Applies ordinal encoding to a DataFrame or array of strings.

    Converts categorical strings (e.g. trial definitions) into integer codes.
    Any missing values (NaN or None) are replaced with a dummy value (-1),
    which then maps back to NaN in the final encoding.

    Parameters
    ----------
    str_mat : pandas.DataFrame or numpy.ndarray
        The input matrix of strings (or numeric codes) to encode. Usually a trial
        definition matrix with shape (users x trials).

    Returns
    -------
    lookupTable : numpy.ndarray
        The unique encoded categories, with the dummy value replaced by np.nan
        if present.
    tmp_mat : numpy.ndarray
        The ordinal-encoded matrix with the same shape as `str_mat`, where
        the dummy value is replaced by np.nan.
    """
    
    if isinstance(str_mat, pd.DataFrame):
        str_mat = str_mat.values
    tmp = str_mat.copy()
    if tmp.dtype == "O":
        dummy_value = "-1"
    else:
        dummy_value = -1
    tmp[pd.isna(tmp)] = dummy_value
    lookupTable, int_word_mat = np.unique(tmp, return_inverse=True)
    nan_label = (lookupTable == dummy_value).nonzero()[0]
    #print(nan_label)
    tmp_mat = int_word_mat.reshape(str_mat.shape).astype(float)
    if len(lookupTable[lookupTable == dummy_value])>0:
        tmp_mat[tmp_mat == nan_label] = np.nan
        lookupTable[lookupTable == dummy_value] = np.nan
        
    return lookupTable, tmp_mat

if __name__ == "__main__":
    main_wrapper()
