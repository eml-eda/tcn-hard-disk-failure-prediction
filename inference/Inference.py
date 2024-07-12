import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import glob
import os
from Networks_inference import *
import json
from Dataset_processing import DatasetProcessing
import logger
from tqdm import tqdm


# Define default global values
INFERENCE_PARAMS = {
    'dropout': 0.1,  # LSTM
    'lstm_hidden_s': 64,  # LSTM
    'fc1_hidden_s': 16,  # LSTM
    'hidden_dim': 128,  # MLP_Torch
    'hidden_size': 8,  # DenseNet
    'num_layers': 1,  # NNet
}

def feature_extraction_PCA(X, pca_components):
    """
    Perform feature extraction using Principal Component Analysis (PCA) on the input data.

    Parameters:
    - X: numpy array of shape (samples, features, dim_window)
        The input data to perform feature extraction on.
    - pca_components (int): The number of components to keep.

    Returns:
    - X_pca: numpy array of shape (samples, features, n_components)
        The transformed data after applying PCA.

    """
    samples, features, dim_window = X.shape
    logger.info(f'Extracting: samples: {samples}, features: {features}, dim_window: {dim_window}')
    
    # Ensure n_components is not more than the minimum of dim_window and pca_components
    n_components = min(pca_components, dim_window)
    X_pca = np.zeros((samples, features, n_components))

    for i in tqdm(range(samples), desc='Processing samples'):
        for j in range(features):
            current_data = X[i, j, :].reshape(-1, dim_window)
            current_n_samples, current_n_features = current_data.shape
            
            # Adjust n_components to be within the valid range
            valid_n_components = min(n_components, current_n_samples, current_n_features)
            pca = PCA(n_components=valid_n_components)
            X_pca[i, j, :valid_n_components] = pca.fit_transform(current_data)[:, :valid_n_components]

    return X_pca

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
        logger.info(f"File {file_path} does not exist.")
        return None

    # Read the params dictionary from a JSON file
    with open(file_path, 'r') as f:
        params = json.load(f)

    logger.info(f'Parameters read from: {file_path}')
    logger.info('User parameters:', params)

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

def set_inference_params(*args):
    param_names = ['dropout', 'lstm_hidden_s', 'fc1_hidden_s', 'hidden_dim', 'hidden_size', 'num_layers']
    # Use the global keyword when modifying global variables
    global INFERENCE_PARAMS
    INFERENCE_PARAMS = dict(zip(param_names, args))
    # Print out updated parameters to Gradio interface
    return f"Parameters successfully updated:\n" + "\n".join([f"{key}: {value}" for key, value in INFERENCE_PARAMS.items()])

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
    elif classifier in ['TCN', 'MLP_Torch', 'NNet', 'DenseNet']:
        inference_loader = DataLoader(TCNDataset(X), batch_size=1, shuffle=False)
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
        'id_number', 'serial_number', 'classifier', 'cuda_dev', 'pca_components', 'smoothing_level', 'augmentation_method', 'csv_file'
    ]

    # Assign values directly from the dictionary
    (
        id_number, serial_number, classifier, CUDA_DEV, pca_components, smoothing_level, augmentation_method, csv_file
    ) = dict(zip(param_names, args)).values()

    params = read_params_from_json(classifier, id_number)

    # Unpack the params dictionary into individual variables
    overlap = params['overlap']
    windowing = params['windowing']
    history_signal = params['history_signal']
    features_extraction_method = params['features_extraction_method']
    smart_attributes = params['smart_attributes']
    days_considered_as_failure = params['days_considered_as_failure']

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', id_number)

    # Read the CSV file
    df = pd.read_csv(csv_file.name)

    # Filter the DataFrame, then select the feature columns based on the smart attributes
    df = feature_selection(df.loc[(serial_number, slice(None))], smart_attributes)
    logger.info('Used features')
    for column in list(df):
        logger.info('{:.<27}'.format(column,))

    # Load or prepare your input data for inference
    X_inference = DatasetProcessing(
        df,
        overlap=overlap,
        windowing=windowing,
        window_dim=history_signal,
        days=days_considered_as_failure,
        smoothing_level=smoothing_level,
        augmentation_method=augmentation_method
    )

    # Step x.1: Feature Extraction
    if features_extraction_method == 'custom': 
        # Extract features for the train and test set
        X_inference = feature_extraction(X_inference)
    elif features_extraction_method == 'PCA':
        X_inference = feature_extraction_PCA(X_inference, pca_components)
    elif features_extraction_method == 'None':
        logger.info('Skipping features extraction for training data.')
    else:
        raise ValueError("Invalid features extraction method.")

    # Path to the saved model
    if classifier in ['LSTM', 'TCN', 'MLP_Torch', 'NNet', 'DenseNet']:
        # Get a list of all files in model_dir that start with 'lstm_training_'
        files = glob.glob(os.path.join(model_dir, f'{classifier.lower()}_{id_number}_*.pth'))

    # Check if any files were found
    if not files:
        raise ValueError("No files found for the specified classifier.")

    # Sort the files by modification time and get the most recent one
    latest_file = max(files, key=os.path.getmtime)

    if classifier == 'LSTM':
        num_inputs = X_inference.shape[1]
        lstm_hidden_s = INFERENCE_PARAMS['lstm_hidden_s']
        fc1_hidden_s = INFERENCE_PARAMS['fc1_hidden_s']
        dropout = INFERENCE_PARAMS['dropout']
        model = FPLSTM(lstm_hidden_s, fc1_hidden_s, num_inputs, 2, dropout)
    elif classifier == 'TCN':
        data_dim = X_inference.shape[2]
        num_inputs = X_inference.shape[1]
        model = TCN_Network(data_dim, num_inputs)
    elif classifier == 'MLP_Torch':
        input_dim = X_inference.shape[1] * X_inference.shape[2]
        hidden_dim = INFERENCE_PARAMS['hidden_dim']
        model = MLP(input_dim, hidden_dim)
    elif classifier == 'NNet':
        data_dim = X_inference.shape[2]
        hidden_dim = INFERENCE_PARAMS['hidden_dim']
        num_layers = INFERENCE_PARAMS['num_layers']
        dropout = INFERENCE_PARAMS['dropout']
        model = NNet(data_dim, hidden_dim, num_layers, dropout)
    elif classifier == 'DenseNet':
        num_inputs = X_inference.shape[1]
        hidden_size = INFERENCE_PARAMS['hidden_size']
        model = DenseNet(num_inputs, hidden_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Moving model to {device}')
    model.to(device)

    # Now you can load the model from the latest file
    model.load_state_dict(torch.load(latest_file))

    # Load the model
    model = load_model(model, latest_file)

    if classifier in ['RandomForest', 'KNeighbors', 'DecisionTree', 'LogisticRegression', 'SVM', 'XGB', 'MLP', 'IsolationForest', 'ExtraTrees', 'GradientBoosting', 'NaiveBayes']:
        pass
    elif classifier in ['TCN', 'MLP_Torch', 'LSTM', 'NNet', 'DenseNet']:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEV
        # Make predictions
        predictions = infer(model, X_inference, classifier)

    # Print or use the predictions as needed
    logger.info("Predictions:", predictions)