import os
import pandas as pd
import numpy as np

def get_motor_information(directory, demographic_file = 'q_IC3_demographics_raw.csv'):
    
    """
    Merges and processes motor-related and handedness information from a raw demographic file.

    This function reads a raw demographics file (`q_IC3_demographics_raw.csv`) located
    in the specified directory. It then extracts user responses to specific questions
    regarding their handedness, current hand usage, motor ability, and whether they
    have any hand impairments. The final DataFrame includes:
    
    - `user_id`
    - Handedness, current hand usage, and a derived `dominant_hand` indicator
    - Motor difficulty scores (`motor_standing_issue`, `motor_sit_issue`, `motor_move_issue`)
    - `impaired_hand` indicating whether the user has an impairment in their current hand

    The function discards rows with missing data in either `impaired_hand` or `dominant_hand`.

    Parameters
    ----------
    directory : str
        Path to the directory containing the demographic file`.
    demographic_file : str
        The name of the demographic file, set as default as 'q_IC3_demographics_raw.csv'.
 
    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
        ['user_id', 'handedness', 'current_hand', 'dominant_hand', 'impaired_hand',
         'motor_standing_issue', 'motor_sit_issue', 'motor_move_issue'].
        Rows with missing values in `impaired_hand` or `dominant_hand` are removed.

    Raises
    ------
    FileNotFoundError
        If `q_IC3_demographics_raw.csv` is not found in the specified directory.
    KeyError
        If the expected columns (`question`, `response`, `user_id`) are not present
        in the raw demographics file.


    Examples
    --------
    >>> directory_path = "/path/to/demographics"
    >>> motor_info_df = get_motor_information(directory_path)
    >>> motor_info_df.head()
    
    """
    
    # Build the file path without changing the working directory
    file_path = os.path.join(directory, demographic_file)
    df_dem = pd.read_csv(file_path)
    
    # Define mappings for responses
    handedness_map_1 = {'1': 0, '2': 0, '4': 1, '5': 1, '3': 2, 'Female': np.nan, 'Male': np.nan}
    handedness_map_2 = {'1': 0, '2': 0, '4': 1, '5': 1, '3': 2}
    current_hand_map = {'Righthand': 0, 'Lefthand': 1, '1': np.nan}
    impaired_map = {'1': np.nan, 'No': 0, 'Yes': 1, 'Righthand': np.nan, 'Other:': np.nan, 'Lefthand': np.nan}
    motor_map = {
        'Notatalldifficult': 0,
        'SKIPPED': 0,
        'Alittledifficult': 1,
        'Somewhatdifficult': 2,
        'VeryDifficult': 3,
        'Couldnotdoitatall': 4
    }
    
    def extract_question_data(question, col_name, replacements=None, drop_dups=True):
        
        """
        Filters responses to a specific question and applies optional replacements.

        This helper function looks up rows in `df_dem` (a global/presumed-available
        DataFrame with demographics) where the column 'question' matches the provided
        `question` string. It selects only 'response' and 'user_id', applies a mapping
        of replacements (if provided), optionally removes duplicates, and renames the
        'response' column to `col_name`.

        Parameters
        ----------
        question : str
            The exact question text as it appears in `df_dem['question']`.
        col_name : str
            The name to assign to the 'response' column after extracting it. For example,
            'handedness' or 'impaired_hand'.
        replacements : dict, optional
            A dictionary mapping original response values to new values (e.g. 
            {'Yes': 1, 'No': 0}). If None, no replacements are applied.
        drop_dups : bool, optional
            If True (default), retains only the last occurrence of each `user_id`.
            If False, keeps all occurrences.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing two columns: ['user_id', `col_name`].
            Missing values may appear if the user did not answer or if replacements
            convert certain responses to NaN.

        Raises
        ------
        KeyError
            If 'question', 'response', or 'user_id' columns are missing in the
            global DataFrame `df_dem`.
        """
        
        df = df_dem.loc[df_dem['question'] == question, ['response', 'user_id']].copy()
        if replacements:
            df['response'] = df['response'].replace(replacements)
        if drop_dups:
            df = df.drop_duplicates(subset='user_id', keep='last')
        return df.rename(columns={'response': col_name})
    
    # Extract data for each question
    current_hand = extract_question_data(
        '<center>Whichhandareyouusingtomakeresponsesnow?</center>',
        'current_hand',
        replacements=current_hand_map
    )
    
    handedness_1 = extract_question_data(
        '<center>Areyounaturallyleft-handedorright-handed?Pleaserefertothehandyouhavebeenusingformostofyourlife.</center>',
        'handedness',
        replacements=handedness_map_1
    )
    
    handedness_2 = extract_question_data(
        '<center>Areyouleft-handedorright-handed?</center>',
        'handedness',
        replacements=handedness_map_2
    )
    
    # Combine the two handedness sources and drop any duplicate user entries
    handedness = pd.concat([handedness_1, handedness_2], ignore_index=True)
    handedness = handedness.drop_duplicates(subset='user_id', keep='last')
    
    impaired_hand = extract_question_data(
        '<center>Doyouhaveanyimpairmentsinthehandthatyouarecurrentlyusing?</center>',
        'impaired_hand',
        replacements=impaired_map
    )
    
    motor_standing = extract_question_data(
        '<center>Howdifficultisitforyoutostandwithoutlosingbalance?</center>',
        'motor_standing_issue',
        replacements=motor_map
    )
    
    motor_sit = extract_question_data(
        '<center>Howdifficultisitforyoutostaysittingwithoutlosingbalance?</center>',
        'motor_sit_issue',
        replacements=motor_map
    )
    
    motor_move = extract_question_data(
        '<center>Howdifficultisitforyoutomovefrombedtochair?</center>',
        'motor_move_issue',
        replacements=motor_map
    )
    
    # Compute dominant hand based on current_hand and handedness responses
    dominant = pd.merge(current_hand, handedness, on='user_id', how='left')
    dominant = dominant.dropna(subset=['current_hand', 'handedness']).copy()
    dominant['dominant_hand'] = np.where(
        ((dominant['current_hand'] == 0) & (dominant['handedness'] == 1)) |
        ((dominant['current_hand'] == 1) & (dominant['handedness'] == 0)),
        1, 0
    )
    dominant = dominant[['user_id', 'dominant_hand']]
    
    # Merge all the processed dataframes on user_id using outer joins
    data_frames = [handedness, current_hand, dominant, impaired_hand,
                   motor_standing, motor_sit, motor_move]
    
    df_merged = data_frames[0]
    for df in data_frames[1:]:
        df_merged = pd.merge(df_merged, df, on='user_id', how='outer')
    
    # Drop rows with any missing values (ensuring complete records)
    df_final = df_merged.dropna(subset=['impaired_hand','dominant_hand'])
    
    return df_final


