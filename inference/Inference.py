import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import glob
import os
from Networks_inference import *
import json
from Dataset_processing import DatasetProcessing


def feature_extraction(X):
    """
    Extracts features from the input data.

    :param X (ndarray): Input data of shape (samples, features, dim_window).

    :return: ndarray: Extracted features of shape (samples, features, 4).
    """
    print('Extracting Features')
    samples, features, dim_window = X.shape
    X_feature = np.ndarray((X.shape[0],X.shape[1], 4))
    print('Sum')
    # sum of all the features
    X_feature[:,:,0] = np.sum((X), axis = 2)
    print('Min')
    # min of all the features
    X_feature[:,:,1] = np.min((X), axis = 2)
    print('Max')
    # max of all the features
    X_feature[:,:,2] = np.max((X), axis = 2)
    print('Similar slope')
    # Calculate the slope of the features
    X_feature[:,:,3] = (np.max((X), axis = 2) - np.min((X), axis = 2)) / dim_window
    '''
    print('Slope')
    for s in np.arange(samples):
        for f in np.arange(features):
            model = LinearRegression()
            model.fit(np.arange(X.shape[2]).reshape(-1,1),X[s,f,:])
            X_feature[s,f,3] = model.coef_
            X_feature[s,f,4] = model.intercept_
    '''
    return X_feature

def feature_selection(df, smart_attributes):
    """
    Selects columns from the dataframe that start with 'smart_' and match the smart attributes list,
    while retaining other columns that do not start with 'smart_'.
    
    Args:
        df (pandas.DataFrame): The input dataframe.
        smart_attributes (list): The list of smart attributes.

    Returns:
        pandas.DataFrame: The dataframe with selected features.
    """
    # Find columns that start with 'smart_' and match smart_attributes
    selected_smart_columns = [col for col in df.columns if col.startswith('smart_') and col in smart_attributes]

    # Define the list of other columns to retain
    essential_columns = ['serial_number', 'model', 'capacity_bytes', 'date']

    # Find columns that are in the list of other columns to retain
    other_columns = [col for col in df.columns if col in essential_columns]

    # Combine selected smart columns with other columns
    selected_columns = selected_smart_columns + other_columns

    # Return the dataframe with only the selected columns
    return df[selected_columns]

def read_params_from_json(classifier, id_number):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    param_dir = os.path.join(script_dir, '..', 'model', id_number)

    file_path_pattern = os.path.join(param_dir, f'{classifier.lower()}_{id_number}_params_*.json')
    file_path = max(glob.glob(file_path_pattern), key=os.path.getmtime)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None

    # Read the params dictionary from a JSON file
    with open(file_path, 'r') as f:
        params = json.load(f)

    print(f'Parameters read from: {file_path}')
    print('User parameters:', params)

    return params

def load_model(model, model_path):
    """
    Load the saved LSTM model from the specified path.

    Args:
        model (torch.nn.Module): The LSTM model class.
        model_path (str): The path to the saved model file.

    Returns:
        torch.nn.Module: The loaded LSTM model.
    """
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def infer(model, X, classifier):
    """
    Use the trained model to make predictions on new data.

    Args:
        model (torch.nn.Module): The trained LSTM model.
        X (np.ndarray): The input data for inference.
        classifier (str): The classifier type.

    Returns:
        np.ndarray: The predicted labels.
    """
    if classifier == 'LSTM':
        inference_loader = DataLoader(FPLSTMDataset(X), batch_size=1, shuffle=False, collate_fn=FPLSTM_collate)
    elif classifier in ['TCN', 'MLP_Torch']:
        inference_loader = DataLoader(TCNDataset(X), batch_size=1, shuffle=False, collate_fn=TCN_collate)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in inference_loader:
            sequences = batch.cuda()
            output = model(sequences)
            predicted_labels = output.argmax(dim=1).cpu().numpy()
            predictions.extend(predicted_labels)
    
    return np.array(predictions)

# Example usage:
def initialize_inference(*args):
    # Define parameter names and create a dictionary of params
    param_names = [
        'id_number', 'classifier', 'cuda_dev', 'csv_file'
    ]

    # Assign values directly from the dictionary
    (
        id_number, classifier, CUDA_DEV, csv_file
    ) = dict(zip(param_names, args)).values()

    params = read_params_from_json(classifier, id_number)
    
    # Unpack the params dictionary into individual variables
    serial_number = params['serial_number']
    overlap = params['overlap']
    windowing = params['windowing']
    history_signal = params['history_signal']
    features_extraction_method = params['features_extraction_method']
    interpolate_technique = params['interpolate_technique']
    smart_attributes = params['smart_attributes']
    days_considered_as_failure = params['days_considered_as_failure']

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', id_number)
    # Path to the saved model
    if classifier == 'LSTM':
        # Get a list of all files in model_dir that start with 'lstm_training_'
        files = glob.glob(os.path.join(model_dir, f'lstm_{id_number}_*.pth'))
    elif classifier == 'TCN':
        # Get a list of all files in model_dir that start with 'tcn_training_'
        files = glob.glob(os.path.join(model_dir, f'tcn_{id_number}_*.pth'))
    elif classifier == 'MLP_Torch':
        # Get a list of all files in model_dir that start with 'mlp_training_'
        files = glob.glob(os.path.join(model_dir, f'mlp_manual_{id_number}_*.pth'))

    # Check if any files were found
    if not files:
        raise ValueError("No files found for the specified classifier.")

    # Sort the files by modification time and get the most recent one
    latest_file = max(files, key=os.path.getmtime)

    # Define the model
    model = FPLSTM().cuda()

    # Now you can load the model from the latest file
    model.load_state_dict(torch.load(latest_file))

    # Load the model
    model = load_model(model, latest_file)

    # Read the CSV file
    df = pd.read_csv(csv_file.name)

    # Filter the DataFrame, then select the feature columns based on the smart attributes
    df = feature_selection(df.loc[(serial_number, slice(None))], smart_attributes)
    print('Used features')
    for column in list(df):
        print('{:.<27}'.format(column,))

    # Load or prepare your input data for inference
    X_inference = DatasetProcessing(
        df,
        overlap=overlap,
        windowing=windowing,
        window_dim=history_signal,
        days=days_considered_as_failure,
    )

    # Step x.1: Feature Extraction
    if features_extraction_method == True: 
        # Extract features for the train and test set
        Xtrain = feature_extraction(Xtrain)
        Xtest = feature_extraction(Xtest)

    if classifier in ['RandomForest', 'KNeighbors', 'DecisionTree', 'LogisticRegression', 'SVM', 'XGB', 'MLP', 'IsolationForest', 'ExtraTrees', 'GradientBoosting', 'NaiveBayes']:
        pass
    elif classifier in ['TCN', 'MLP_Torch', 'LSTM']:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEV
        # Make predictions
        predictions = infer(model, X_inference, classifier)

    # Print or use the predictions as needed
    print("Predictions:", predictions)