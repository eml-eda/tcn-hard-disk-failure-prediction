import os
import json
import logger
from datetime import datetime


def save_best_params_to_json(best_params, classifier_name, id_number):
    """
    Saves the best parameters to a JSON file.

    Args:
        best_params (dict): The best parameters.
        classifier_name (str): The name of the classifier.
        id_number (str): The ID number for the model.

    Returns:
        None
    """
    # Define the directory path
    param_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', id_number)
    # Create the directory if it doesn't exist
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    # Define the file path
    file_path = os.path.join(param_dir, f'{classifier_name.lower()}_{id_number}_best_params.json')

    # Save the best parameters to a JSON file
    with open(file_path, 'w') as f:
        json.dump(best_params, f)

    logger.info(f'Best parameters saved to: {file_path}')

def load_best_params_from_json(classifier_name, id_number):
    """
    Loads the best parameters from a JSON file.

    Args:
        classifier_name (str): The name of the classifier.
        id_number (str): The ID number for the model.

    Returns:
        dict: The best parameters.
    """
    # Define the directory path
    param_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', id_number)

    # Define the file path
    file_path = os.path.join(param_dir, f'{classifier_name.lower()}_{id_number}_best_params.json')

    # Load the best parameters from a JSON file
    with open(file_path, 'r') as f:
        best_params = json.load(f)

    logger.info(f'Best parameters loaded from: {file_path}')

    return best_params

def save_params_to_json(df, *args):
    """
    Save the parameters to a JSON file.

    Args:
        df (DataFrame): The input DataFrame.
        *args: Variable length argument list containing the parameter values.

    Returns:
        str: The file path where the parameters are saved.
    """
    # Define parameter names and create a dictionary of params
    param_names = [
        'model', 'id_number', 'years', 'test_type','windowing', 'min_days_hdd', 'days_considered_as_failure',
        'test_train_percentage', 'oversample_undersample', 'balancing_normal_failed',
        'history_signal', 'classifier', 'features_extraction_method', 'cuda_dev',
        'ranking', 'num_features', 'overlap', 'split_technique', 'interpolate_technique',
        'search_method', 'fillna_method', 'pca_components', 'smoothing_level'
    ]

    # Assign values directly from the dictionary
    params = dict(zip(param_names, args))

    # Get column names that start with 'smart_'
    smart_columns = [col for col in df.columns if col.startswith('smart_')]

    # Add smart_columns to params under the key 'smart_attributes'
    params['smart_attributes'] = smart_columns

    script_dir = os.path.dirname(os.path.abspath(__file__))
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_dir = os.path.join(script_dir, '..', 'model', params['id_number'])

    # Create the directory if it doesn't exist
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    logger.info(f'Saving parameters: {params}')

    # Define the file path
    file_path = os.path.join(param_dir, f"{params['classifier'].lower()}_{params['id_number']}_params_{now_str}.json")

    # Write the params dictionary to a JSON file
    with open(file_path, 'w') as f:
        json.dump(params, f)

    logger.info(f'Parameters saved to: {file_path}')

    return file_path
