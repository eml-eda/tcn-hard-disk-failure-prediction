import pandas as pd
import numpy as np
import math
import pickle
import os
import matplotlib.pyplot as plt
import glob
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import scipy
import scipy.stats
import re
import dask.dataframe as dd
from collections import Counter
import logger
from tqdm import tqdm
from GeneticFeatureSelector import GeneticFeatureSelector
from hmmlearn import hmm
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def plot_feature(dataset):
    """
    Plots a scatter plot of a specific feature in the dataset.

    Parameters:
    - dataset (dict): A dictionary containing the dataset with keys 'X' and 'Y'.

    Returns:
    - None
    """
    X = dataset['X']
    Y = dataset['Y']
    feat = X[:,10]
    np.arange(len(feat))
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(feat[Y==0])),feat[Y==0], color ='C0', label='good HDD')
    ax.scatter(np.arange(len(feat[Y==0]),len(feat)),feat[Y==1],color ='C1',label='failed HDD')
    ax.set_ylabel('SMART feature [#]', fontsize=14, fontweight='bold', color = 'C0')
    ax.set_xlabel('time points / hdd', fontsize=14, fontweight='bold')
    legend_properties = {'weight':'bold'}
    plt.legend(fontsize=12, prop = legend_properties)
    plt.show()

def plot_hdd(X, fail, prediction):
    """
    Plots the SMART features of a hard disk drive (HDD) over time.

    Parameters:
    X (numpy.ndarray): The input array containing the SMART features of the HDD.
    fail (int): The failure status of the HDD (1 for failed, 0 for good).
    prediction (str): The predicted status of the HDD.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    features = {
        'total_features': [
            'date',
            'failure',
            'smart_1_normalized',
            'smart_5_normalized',
            'smart_5_raw',
            'smart_7_normalized',
            'smart_9_raw',
            'smart_12_raw',
            'smart_183_raw',
            'smart_184_normalized',
            'smart_184_raw',
            'smart_187_normalized',
            'smart_187_raw',
            'smart_189_normalized',
            'smart_193_normalized',
            'smart_193_raw',
            'smart_197_normalized',
            'smart_197_raw',
            'smart_198_normalized',
            'smart_198_raw',
            'smart_199_raw'
        ]
    }

    for k, i in enumerate([1, 2, 7, 8, 9, 10]):
        ax.plot(np.arange(X.shape[0]), X[:, i] + 0.01 * k, label=features['total_features'][i + 2])
    ax.set_ylabel('SMART features value [#]', fontsize=14, fontweight='bold', color='C0')
    ax.set_xlabel('time points', fontsize=14, fontweight='bold')
    legend_properties = {'weight': 'bold'}
    plt.legend(fontsize=12, prop=legend_properties)
    plt.title('The HDD is {} (failed=1/good=0) predicted as {}'.format(fail, prediction))
    plt.show()

def pandas_to_3dmatrix(read_dir, model, years, dataset_raw):
    """
    Convert a pandas DataFrame to a 3D matrix.

    Args:
        read_dir (str): The directory path where the matrix file will be read from or saved to.
        model: The model object.
        years (list): A list of years.
        dataset_raw (pandas.DataFrame): The raw dataset.

    Returns:
        dict: The converted 3D matrix dataset.

    Raises:
        FileNotFoundError: If the matrix file is not found.

    """
    join_years = '_' + '_'.join(years)
    name_file = f'Matrix_Dataset_{join_years}.pkl'

    try:
        with open(os.path.join(read_dir, name_file), 'rb') as handle:
            dataset = pickle.load(handle)
        print('Matrix 3d {} already present'.format(name_file))
    except FileNotFoundError:
        print(f'Creating matrix 3D {name_file}')
        valid_rows_mask = []

        # rows_dim = len(dataset_raw['failure'])
        # feat_dim = len(dataset_raw.columns) - 2

        # Remove HDDs with some features always NaN
        for _, row in dataset_raw.iterrows():
            valid = not any(len(col) == sum(math.isnan(x) for x in col) for col in row[2:])
            valid_rows_mask.append(valid)

        dataset_raw = dataset_raw[valid_rows_mask]
        # rows_dim = len(dataset_raw['failure'])

        # Compute max timestamps in an HDD
        print('Computing maximum number of timestamps of one HDD')
        max_len = max(len(row['date']) for _, row in dataset_raw.iterrows())

        matrix_3d = []
        for k, row in dataset_raw.iterrows():
            print(f'Analyzing HD number {k}', end="\r")
            hd = [row[feature] for feature in row.index[1:]]
            hd = np.asarray(hd)
            hd_padded = np.pad(hd, ((0, 0), (0, max_len - hd.shape[1])), mode='constant', constant_values=2)
            matrix_3d.append(hd_padded.T)

        matrix_3d = np.asarray(matrix_3d)

        # Debugging information
        good, failed = 0, 0
        for row in matrix_3d[:, :, 0]:
            if 1 in row:
                failed += 1
            else:
                good += 1

        print(f'There are {good} good disks and {failed} failed disks in the dataset')

        dataset = {'matrix': matrix_3d}
        with open(os.path.join(read_dir, name_file), 'wb') as handle:
            pickle.dump(dataset, handle)

    return dataset

def matrix3d_to_datasets(matrix, window=1, divide_hdd=1, training_percentage=0.7, resampler_balancing='auto', oversample_undersample=0):
    """
    Convert a 3D matrix to datasets for training and testing.

    Args:
        matrix (ndarray): The 3D matrix containing the data.
        window (int, optional): The size of the sliding window. Defaults to 1.
        divide_hdd (int, optional): Flag to divide the HDDs. Defaults to 1.
        training_percentage (float, optional): The percentage of data to use for training. Defaults to 0.7.
        resampler_balancing (float, optional): The resampler balancing factor. Defaults to 'auto'.
        oversample_undersample (int, optional): The type of resampling to use. Defaults to 0.

    Returns:
        dict: A dictionary containing the training and testing datasets.

    Raises:
        FileNotFoundError: If the dataset file is not found.
    """

    name_file = 'Final_Dataset.pkl'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    read_dir = os.path.join(script_dir, '..', 'output')

    try:
        with open(os.path.join(read_dir, name_file), 'rb') as handle:
            dataset = pickle.load(handle)
        print('Dataset {} already present'.format(name_file))
    except FileNotFoundError:
        X_matrix = matrix[:, :, :]
        Y_matrix = np.zeros(X_matrix.shape[0])

        for hdd in np.arange(X_matrix.shape[0]):
            if 1 in X_matrix[hdd, :, 0]:
                Y_matrix[hdd] = 1
        
        print('Failed hard disk = {}'.format(sum(Y_matrix)))

        if divide_hdd == 1:
            X_train_hd, X_test_hd, y_train_hd, y_test_hd = train_test_split(
                X_matrix, Y_matrix, stratify=Y_matrix, test_size=1 - training_percentage)

            def create_dataset(X_hd, y_hd, for_training=True):
                X, Y, HD_numbers = [], [], []
                for hdd_number in np.arange(y_hd.shape[0]):
                    print(f"Analyzing HD number {hdd_number}", end="\r")
                    if y_hd[hdd_number] == 1:
                        end_idx = np.where(X_hd[hdd_number, :, 0] == 1)[0][0]
                        temp = X_hd[hdd_number, :end_idx, 1:]
                        label = np.concatenate((np.zeros(temp.shape[0] - 7), np.ones(7)))
                    else:
                        end_idx = np.where(X_hd[hdd_number, :, 0] == 2)[0][0] - 8 if np.where(X_hd[hdd_number, :, 0] == 2)[0].size > 0 else -7
                        temp = X_hd[hdd_number, :end_idx, 1:]
                        label = np.zeros(temp.shape[0])

                    X.append(temp)
                    Y.append(label)
                    if not for_training:
                        HD_numbers.append(np.ones(temp.shape[0]) * hdd_number)

                X, Y = np.concatenate(X), np.concatenate(Y)
                if not for_training:
                    HD_numbers = np.concatenate(HD_numbers)
                    return X, Y, HD_numbers
                return X, Y

            X_train, Y_train = create_dataset(X_train_hd, y_train_hd)
            X_test, Y_test, HD_number_test = create_dataset(X_test_hd, y_test_hd, for_training=False)

            # Remove rows with NaN values
            non_nan_train = ~np.isnan(X_train).any(axis=1)
            X_train, Y_train = X_train[non_nan_train], Y_train[non_nan_train]

            non_nan_test = ~np.isnan(X_test).any(axis=1)
            X_test, Y_test, HD_number_test = X_test[non_nan_test], Y_test[non_nan_test], HD_number_test[non_nan_test]

            # Resampling
            resampler = RandomUnderSampler(sampling_strategy=resampler_balancing, random_state=42) if oversample_undersample == 0 else SMOTE(sampling_strategy=resampler_balancing, random_state=42)
            X_train, Y_train = resampler.fit_resample(X_train, Y_train)

            dataset = {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test, 'HDn_test': HD_number_test}

            with open(os.path.join(read_dir, name_file), 'wb') as handle:
                pickle.dump(dataset, handle)

    return dataset

def import_data(years, models, name, **args):
    """ Import hard drive data from csvs on disk.
    
    :param quarters: List of quarters to import (e.g., 1Q19, 4Q18, 3Q18, etc.)
    :param models: List of the hard drive model numbers to import.
    :param columns: List of the columns to import.
    :return: Dataframe with hard drive data.
    
    """

    # Read the correct .pkl file
    years_list = '_' + '_'.join(years)
    failed = False  # This should be set based on your specific criteria or kept as a placeholder
    suffix = 'failed' if failed else 'all'
    model_string = "_".join(models)

    # Fix the directory name as output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(script_dir, '..', 'output', f'HDD{years_list}_{suffix}_{model_string}_appended.pkl')

    if not os.path.exists(file):
        # Fix the directory name
        file = os.path.join(script_dir, '..', 'output', f'HDD{years_list}_{model_string}_all.pkl')

    try:
        df = pd.read_pickle(file)
        logger.info(f'Data loaded from {file}')
    except FileNotFoundError:
        logger.info('Creating new DataFrame from CSV files.')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        all_data = []
        unique_models = set()

        for y in tqdm(years, desc="Analyzing years"):
            # Fix the directory name
            for f in tqdm(glob.glob(os.path.join(script_dir, '..', 'HDD_dataset', y, '*.csv')), desc=f"Analyzing files in year {y}"):
                if 'features' in args and name in args['features']:
                    data = pd.read_csv(f, header=0, usecols=args['features'][name], parse_dates=['date'])
                else:
                    data = pd.read_csv(f, header=0, parse_dates=['date'])

                # Filter the data based on the model number or manufacturer prefix
                # If the 'manufacturer' key is not found in the args dictionary or its value is 'custom', we use the 'model' key to filter the data
                if args.get('manufacturer', 'custom') != 'custom':
                    model_data = data[data['model'].str.startswith(args['manufacturer'])].copy()
                    unique_models.update(model_data['model'].unique())
                    all_data.append(model_data)  # Append the filtered data to all_data
                else:
                    for model in models:
                        model_data = data[data.model == model].copy()
                        unique_models.add(model)
                        model_data.failure = model_data.failure.astype('int')
                        all_data.append(model_data)

        logger.info(f"Unique models: {unique_models}")
        df = pd.concat(all_data, ignore_index=True)
        df.set_index(['serial_number', 'date'], inplace=True)
        df.sort_index(inplace=True)
        df.to_pickle(file)
        logger.info(f'Data saved to {file}')

    return df


def filter_HDs_out(df, min_days, time_window, tolerance):
    """ Find HDs with an excessive amount of missing values.
    
    :param df: Input dataframe.
    :param min_days: Minimum number of days HD needs to have been powered on.
    :param time_window: Size of window to count missing values (e.g., 7 days as '7D', or five days as '5D')
    :param tolerance: Maximum number of days allowed to be missing within time window.
    :return: List of HDs with excessive amount of missing values
    
    """
    # Calculate the percentage of non-missing values
    pcts = df.notna().sum() / df.shape[0] * 100
    # Identify columns to remove True / False, select column names with 'True' value, generate a list of column names to remove
    cols = list(pcts[pcts < 45.0].index)
    df.drop(cols, axis=1, inplace=True)  # Drop columns with any missing values
    df = df.dropna()  # Drop rows with any missing values
    bad_power_hds = []
    bad_missing_hds = []

    # Loop over each group in the DataFrame, grouped by the first level of the index (serial number)
    for serial_num, inner_df in tqdm(df.groupby(level=0), desc="Analyzing Hard Drives", unit="drive", ncols=100):
        if len(inner_df) < min_days:  # identify HDs with too few power-on days
            bad_power_hds.append(serial_num)

        inner_df = inner_df.droplevel(level=0).asfreq('D')  # Convert inner_df to daily frequency

        # Find the moving average of missing values within a window of time_window days in any column of the DataFrame
        moving_avg_missing = inner_df.isna().rolling(time_window).mean()

        # Find the maximum number of missing values within any window of time_window days in any column of the DataFrame
        n_missing = inner_df.isna().rolling(time_window).sum()

        # Identify HDs with too many missing values, compared to the moving average
        bad_missing_hds = n_missing[n_missing > moving_avg_missing * tolerance].index.tolist()

    bad_hds = set(bad_missing_hds + bad_power_hds)
    logger.info(f'Filter result: bad_missing_hds: {len(bad_missing_hds)}, bad_power_hds: {len(bad_power_hds)}')
    hds_remove = len(bad_hds)
    hds_total = len(df.reset_index().serial_number.unique())
    logger.info(f'Total HDs: {hds_total}    HDs removed: {hds_remove} ({round(hds_remove / hds_total * 100, 2)}%)')

    bad_hds = pd.Series(list(bad_hds))
    bad_hds = bad_hds[bad_hds.isin(df.index)]
    df = df.drop(bad_hds, axis=0)

    num_fail = df.failure.sum()
    num_not_fail = len(df.reset_index().serial_number.unique()) - num_fail
    pct_fail = num_fail / (num_not_fail + num_fail) * 100 

    logger.info(f'{num_fail} failed')
    logger.info(f'{num_not_fail} did not fail')
    logger.info(f'{pct_fail:.5f}% failed')
    # Print on 
    return bad_missing_hds, bad_power_hds, df

def interpolate_ts(df, method='linear'):

    """ Interpolate hard drive Smart attribute time series.

    :param df: Input dataframe.
    :param method: String, interpolation method.
    :return: Dataframe with interpolated values.
    """

    interp_dfs = []
    total_interpolated = 0

    for serial_num, inner_df in tqdm(df.groupby(level=0), desc="Interpolating groups", leave=False, unit="drive", ncols=100):
        inner_df = inner_df.droplevel(level=0).asfreq('D')
        # Count the number of NaN values before interpolation
        before_interpolation = inner_df.isna().sum().sum()
        inner_df.interpolate(method=method, limit_direction='both', axis=0, inplace=True)

        # Specify the columns to round
        columns_to_round = ['predict_val', 'validate_val']

        # Add a small threshold and round all values below this threshold to 0
        threshold = 1e-10
        inner_df[columns_to_round] = inner_df[columns_to_round].applymap(lambda x: 0 if np.abs(x) < threshold else x)
        inner_df[columns_to_round] = inner_df[columns_to_round].round(0)

        # Count the number of NaN values after interpolation and rounding
        after_interpolation = inner_df.isna().sum().sum()

        # Update the total number of interpolated values
        total_interpolated += before_interpolation - after_interpolation
        inner_df.fillna(method='ffill', inplace=True)  # Forward fill
        inner_df.fillna(method='bfill', inplace=True)  # Backward fill
        inner_df['serial_number'] = serial_num
        inner_df = inner_df.reset_index()

        #print(f'Added {len(inner_df)} items for serial number: {serial_num}\r', end='\r')
        interp_dfs.append(inner_df)

    interp_df = pd.concat(interp_dfs, axis=0)
    df = interp_df.set_index(['serial_number', 'date']).sort_index()
    logger.info(f'Total interpolated values: {total_interpolated}, Percentage of interpolated values: {(total_interpolated / df.size) * 100}%')

    return df

def generate_failure_predictions(df, days, window):
    """
    Generate target arrays for binary classification based on the failure column in the input dataframe.

    :param df (pandas.DataFrame): Input dataframe containing the failure column.
    :param days (int): Number of days to consider for failure prediction.
    :param window (int): Number of days to consider for validation.

    :return: A tuple containing two numpy arrays - pred_list and valid_list.
            - pred_list: Array of predicted failure values, where 1 represents failure and 0 represents non-failure.
            - valid_list: Array of validation values, where 1 represents the validation period and 0 represents the non-validation period.
    """
    pred_list = np.asarray([])
    valid_list = np.asarray([])

    # Filter out groups with length less than or equal to days + window
    group_sizes = df.groupby(level=0).size()
    df = df[df.index.get_level_values(0).isin(group_sizes[group_sizes > days + window].index)]

    for i, (serial_num, inner_df) in enumerate(tqdm(df.groupby(level=0), desc="Analyzing Hard Drives", unit="drive", ncols=100), start=1):
        print('Analyzing HD {} number {} \r'.format(serial_num,i), end="\r")
        slicer_val = len(inner_df)  # save len(df) to use as slicer value on smooth_smart_9

        if inner_df.failure.max() == 1:
            # if the HD failed, we create a prediction array with 1s for the last 'days' days and 0s for the rest
            # np.ones(days) represents the period immediately before and including the failure
            # np.zeros(slicer_val - days) represents the period before the failure
            predictions = np.concatenate((np.zeros(slicer_val - days), np.ones(days)))
            # create a validation array with 1s for the last 'days' days and 0s for the rest
            valid = np.concatenate((np.zeros(slicer_val - days - window), np.ones(days + window)))
        else:
            # if the HD did not fail, we set the prediction array to all 0s
            predictions = np.zeros(slicer_val)
            valid = np.zeros(slicer_val)
        pred_list = np.concatenate((pred_list, predictions))
        valid_list = np.concatenate((valid_list, valid))
    logger.info(f'HDs analyzed: {i}')
    pred_list = np.asarray(pred_list)
    valid_list = np.asarray(valid_list)
    return df, pred_list, valid_list


def feature_extraction(X):
    """
    Extracts features from the input data with the following steps:
    1. Sum of all the features and store in the first column.
    2. Minimum of all the features and store in the second column.
    3. Maximum of all the features and store in the third column.
    4. Calculate the slope of the features and store in the fourth column.
    5. Store the y-intercept of the linear regression line (i.e., the expected mean value of Y when all X=0) in the fifth column.
    6. Use HMM to generate state sequences and store the most frequent state in the fifth column.
    7. Calculate the standard deviation and store in the seventh column.
    8. Calculate the autocorrelation for lag-1 and store in the eighth column.
    
    :param X (ndarray): Input data of shape (samples, features, dim_window).

    :return: ndarray: Extracted features of shape (samples, features, 8).
    """
    samples, features, dim_window = X.shape
    logger.info(f'Extracting: samples: {samples}, features: {features}, dim_window: {dim_window}')
    X_feature = np.zeros((samples, features, 8))
    # sum of all the features
    X_feature[:,:,0] = np.sum((X), axis=2)
    #print(f'Sum: {X_feature[:,:,0]}')
    # min of all the features
    X_feature[:,:,1] = np.min((X), axis=2)
    #print(f'Min: {X_feature[:,:,1]}')
    # max of all the features
    X_feature[:,:,2] = np.max((X), axis=2)
    #print(f'Max: {X_feature[:,:,2]}')
    # Calculate the slope of the features
    #X_feature[:,:,3] = (np.max((X), axis = 2) - np.min((X), axis = 2)) / dim_window
    #print(f'Similar slope: {X_feature[:,:,3]}')
    # Use Linear Regression to extract slope and intercept
    for s in tqdm(range(samples), desc='Processing samples with LinearRegression', unit='sample', ncols=100):
        for f in range(features):
            model = LinearRegression()
            model.fit(np.arange(dim_window).reshape(-1, 1), X[s, f, :])
            X_feature[s, f, 3] = model.coef_[0]  # Slope
            X_feature[s, f, 4] = model.intercept_  # Intercept
    #print(f'Coefficent: {X_feature[:,:,3]}')
    #print(f'Intercept: {X_feature[:,:,4]}')
    # Use HMM to generate state sequences
    for f in tqdm(range(features), desc='Processing features with GaussianHMM', leave=False, unit='feature', ncols=100):
        feature_series = X[:, f, :]
        feature_series_reshaped = feature_series.reshape(-1, dim_window)

        # Train HMM on the reshaped feature series
        n_states = 3  # Number of hidden states (this can be adjusted)
        hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
        hmm_model.fit(feature_series_reshaped)

        # Generate state sequences for each sample
        for s in tqdm(range(samples), desc='Generating state sequences for each sample', leave=False, unit='sample', ncols=100):
            state_seq = hmm_model.predict(feature_series[s].reshape(-1, 1))
            # Using the most frequent state as an additional feature
            most_frequent_state = np.bincount(state_seq).argmax()
            X_feature[s, f, 5] = most_frequent_state  # Store HMM result at index 5
    #print(f'HMM state sequence: {X_feature[:, :, 5]}')

    # Calculate the standard deviation
    X_feature[:, :, 6] = np.std(X, axis=2)
    #print(f'Standard Deviation: {X_feature[:, :, 6]}')

    # Calculate the autocorrelation for lag-1
    for s in tqdm(range(samples), desc="Processing samples with Auto Correlation"):
        for f in range(features):
            X_feature[s, f, 7] = np.corrcoef(X[s, f, :-1], X[s, f, 1:])[0, 1]
    #print(f'Autocorrelation: {X_feature[:, :, 7]}')

    return X_feature

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

class DatasetPartitioner:
    """
        https://github.com/Prognostika/tcn-hard-disk-failure-prediction/wiki/Code_Process#partition-dataset-subflowchart
    """
    def __init__(self, df, model, overlap=0, rank='None', num_features=10, test_type='t-test', technique='random',
                 test_train_perc=0.2, windowing=1, window_dim=5, resampler_balancing='auto', oversample_undersample='None', fillna_method='None', smoothing_level=0.5):
        """
        Initialize the DatasetPartitioner object.
        
        Parameters:
        - df (DataFrame): The input dataset.
        - model (str): The name of the model.
        - overlap (int): The overlap value for windowing (default: 0).
        - rank (str): The rank value (default: 'None').
        - num_features (int): The number of features (default: 10).
        - test_type (str): The test type (default: 't-test').
        - technique (str): The partitioning technique (default: 'random').
        - test_train_perc (float): The percentage of data to be used for testing (default: 0.2).
        - windowing (int): The windowing value (default: 1).
        - window_dim (int): The window dimension (default: 5).
        - resampler_balancing (float): The resampler balancing factor (default: auto).
        - oversample_undersample (str): The oversample/undersample value (default: None).
        - fillna_method (str): Method to fill the NA values (default: 'None').
        - smoothing_level (float): The smoothing level (default: 0.5).

        """
        self.df = df
        self.model = model
        self.overlap = overlap
        self.rank = rank
        self.num_features = num_features
        self.test_type = test_type
        self.technique = technique
        self.test_train_perc = test_train_perc
        self.windowing = windowing
        self.window_dim = window_dim
        self.resampler_balancing = resampler_balancing
        self.oversample_undersample = oversample_undersample
        self.fillna_method = fillna_method
        self.smoothing_level = smoothing_level
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = self.partition()

    def partition(self):
        """
        Partition the dataset into training and test sets.


        Parameters:
            None

        Returns:
        - Xtrain (ndarray): The training data.
        - Xtest (ndarray): The test data.
        - ytrain (Series): The training labels.
        - ytest (Series): The test labels.
        """
        self.df.reset_index(inplace=True) # Step 1.1: Reset index.
        # Step 1.2: Preprocess the dataset.
        mms = MinMaxScaler(feature_range=(0, 1)) # Normalize the dataset

        # Extract temporal data
        # Updated: temporal now also drops 'model' and 'capacity_bytes' columns, because they are object. We need float64.
        temporal = self.df[['serial_number', 'date', 'failure', 'predict_val', 'validate_val', 'model', 'capacity_bytes']]
        self.df.drop(columns=temporal.columns, inplace=True)
        self.df = pd.DataFrame(mms.fit_transform(self.df), columns=self.df.columns, index=self.df.index)
        # self.df is now normalized, but temporal is original string data, to avoid normalization of 'serial_number' and 'date' and other non float64 columns
        self.df = pd.concat([self.df, temporal], axis=1)
        # Perform the windowing action for time series data
        windowed_df = self.handle_windowing()
        # Preprocess the dataset for splitting, remove the redundant columns
        windowed_df = self.preprocess_dataset(windowed_df)
        # Add exponential smoothing method
        logger.info('Performing exponential smoothing...')
        for col in tqdm(windowed_df.columns, desc="Processing columns", leave=False, unit="column", ncols=100):
            if col.startswith('smart'):
                windowed_df[col] = ExponentialSmoothing(windowed_df[col], trend=None, seasonal=None, seasonal_periods=None).fit(smoothing_level=self.smoothing_level).fittedvalues
        logger.info('Creating training and test dataset...')
        return self.split_dataset(windowed_df)
    
    def factors(self, n):
        """
        Returns a list of factors of the given number.

        Parameters:
        - n (int): The number to find the factors of.

        Returns:
        list: A list of factors of the given number.
        """
        factors = []
        # Check for the smallest prime factor 2
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        # Check for odd factors from 3 upwards
        factor = 3
        while factor * factor <= n:
            while n % factor == 0:
                factors.append(factor)
                n //= factor
            factor += 2
        # If n became a prime number greater than 2
        if n > 1:
            factors.append(n)
        return factors
    
    def under_sample(self, df, down_factor):
        """
        Perform under-sampling on a DataFrame.
        As a result of the undersampling, any NA values introduced by shifting are filtered out, leaving no NA rows in the final windowed DataFrame.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            down_factor (int): The down-sampling factor.

        Returns:
            list: A list of indexes to be used for under-sampling.

        """
        indexes = (
            df.predict_val
            .rolling(down_factor) # Create a rolling window of size down_factor over the 'predict_val' column
            .max() # Find the maximum value in each window.
            [((len(df) - 1) % down_factor):-7:down_factor]
            .index
            .tolist()
        )
        return indexes

    def handle_windowing(self):
        """
        Handle the windowing process for the dataset.


        Parameters:
            None
        
        Returns:
        - DataFrame: The windowed dataset.
        """

        # Step 2: Check Windowing
        if self.windowing != 1:
            return self.df

        try:
            # Step 2.1.1: If Yes, attempt to load the pre-processed windowed dataset.
            windowed_df = pd.read_pickle(os.path.join(self.script_dir, '..', 'output', f'{self.model}_Dataset_windowed_{self.window_dim}_rank_{self.rank}_{self.num_features}_overlap_{self.overlap}.pkl'))
            logger.info('Loading the windowed dataset')
            return self.rename_columns(windowed_df)
        except FileNotFoundError:
            # Step 2.1.2: If No, perform windowing on the dataset.
            logger.info('Windowing the df')
            return self.perform_windowing()

    def rename_columns(self, df):
        """
        Rename the columns of the dataframe to avoid duplicates.
        --- Step 3: Prepare data for modeling.
        Parameters:
        - df (DataFrame): The input dataframe.

        Returns:
        - DataFrame: The dataframe with renamed columns.
        """

        cols = []
        count = {}
        for column in df.columns:
            if column not in count:
                count[column] = 0
            count[column] += 1
            new_column = f"{column}_{count[column]}" if count[column] > 1 else column
            cols.append(new_column)
        df.columns = cols
        df.sort_index(axis=1, inplace=True)
        return df

    def perform_windowing(self):
        """
        Perform the windowing operation on the dataset.
        We have the serial_number and date columns in the dataset here.

        Parameters:
            None

        Returns:
        - DataFrame: The windowed dataframe.
        """
        # Convert the initial DataFrame to a Dask DataFrame for heavy operations
        chunk_columns = 100000
        windowed_df = dd.from_pandas(self.df.copy(), npartitions=int(len(self.df)/chunk_columns) + 1)
        if self.overlap == 1:  # If the overlap option is chosed as complete overlap
            # The following code will generate self.window_dim - 1 columns for each column in the dataset
            for i in np.arange(self.window_dim - 1):
                print(f'Concatenating time - {i} \r', end="\r")
                # Shift the dataframe and concatenate along the columns
                windowed_df = dd.concat([self.df.shift(i + 1), windowed_df], axis=1)
        elif self.overlap == 2:  # If the overlap option is chosed as dynamic overlap based on the factors of window_dim
            # Get the factors of window_dim
            window_dim_divisors = self.factors(self.window_dim)
            total_shifts = 0
            previous_down_factor = 1
            serials = self.df['serial_number']
            for down_factor in window_dim_divisors:
                # Shift the dataframe by the factor and concatenate
                for i in np.arange(down_factor - 1):
                    total_shifts += previous_down_factor
                    print(f'Concatenating time - {total_shifts} \r', end="\r")
                    windowed_df = dd.concat([self.df.shift(i + 1), windowed_df], axis=1)
                previous_down_factor *= down_factor

                # Compute intermediate result to apply sampling
                windowed_df = windowed_df.compute()  # Convert back to pandas for sampling
                # Under sample the dataframe based on the serial numbers and the factor
                indexes = windowed_df.groupby(serials).apply(self.under_sample, down_factor)
                # Update windowed_df based on the indexes, undersamples the DataFrame based on the serial numbers and the factor down_factor, reducing the number of rows in the DataFrame.
                windowed_df = windowed_df.loc[np.concatenate(indexes.values.tolist(), axis=0), :]
                # Convert back to Dask DataFrame
                windowed_df = dd.from_pandas(windowed_df, npartitions=int(len(windowed_df)/chunk_columns) + 1)
        else:  # If the overlap is other value, then we only completely overlap the dataset for the failed HDDs, and dynamically overlap the dataset for the good HDDs
            # Get the factors of window_dim
            window_dim_divisors = self.factors(self.window_dim)
            total_shifts = 0
            previous_down_factor = 1
            serials = self.df['serial_number']
            df_failed = self.df[self.df['validate_val']==1]
            windowed_df_failed = df_failed
            for i in np.arange(self.window_dim - 1):
                print(f'Concatenating time - {i} \r', end="\r")
                # Shift the dataframe and concatenate along the columns
                windowed_df_failed = dd.concat([self.df.shift(i + 1), windowed_df_failed], axis=1)
            for down_factor in window_dim_divisors:
                # Shift the dataframe by the factor and concatenate
                for i in np.arange(down_factor - 1):
                    total_shifts += previous_down_factor
                    print(f'Concatenating time - {total_shifts} \r', end="\r")
                    windowed_df = dd.concat([self.df.shift(i + 1), windowed_df], axis=1)
                previous_down_factor *= down_factor

                # Compute intermediate result to apply sampling
                windowed_df = windowed_df.compute()  # Convert back to pandas for sampling
                # Under sample the dataframe based on the serial numbers and the factor
                indexes = windowed_df.groupby(serials).apply(self.under_sample, down_factor)
                # Update windowed_df based on the indexes
                windowed_df = windowed_df.loc[np.concatenate(indexes.values.tolist(), axis=0), :]
                # Convert back to Dask DataFrame
                windowed_df = dd.from_pandas(windowed_df, npartitions=int(len(windowed_df)/chunk_columns) + 1)

            windowed_df = dd.concat([windowed_df, windowed_df_failed])
            windowed_df.reset_index(inplace=True, drop=True)

        # Compute the final Dask DataFrame to pandas DataFrame
        final_df = windowed_df.compute()
        # Generate the final DataFrame
        final_df.to_pickle(os.path.join(self.script_dir, '..', 'output', f'{self.model}_Dataset_windowed_{self.window_dim}_rank_{self.rank}_{self.num_features}_overlap_{self.overlap}.pkl'))
        return self.rename_columns(final_df)

    def split_dataset(self, df):
        """
        Split the dataset into training and test sets based on the specified technique.

        --- Step 4: Technique selection.

        Parameters:
        - df (DataFrame): The input dataframe.

        Returns:
        - Xtrain (ndarray): The training data.
        - Xtest (ndarray): The test data.
        - ytrain (Series): The training labels.
        - ytest (Series): The test labels.
        """
        if self.technique == 'random':
            y = df['predict_val']   # y represents the prediction value (Series)
            df.drop(columns=['serial_number', 'date', 'failure', 'predict_val', 'validate_val'], inplace=True)
            X = df.values   # X represents the smart features (ndarray)

            if self.windowing == 1:
                # If we use the down sampling, then the data_dim will be the sum of the factors of the window_dim
                data_dim = sum(number - 1 for number in self.factors(self.window_dim)) + 1 if self.overlap != 1 else self.window_dim
                X = self.arrays_to_matrix(X, data_dim)
            else:
                X = np.expand_dims(X, axis=1)  # Add an extra dimension
            logger.info(f'Augmented data of predict_val is: {Counter(y)}')
            # Print the shapes of the train and test sets
            # Xtrain: ndarray, Xtest: ndarray, ytrain: Series, ytest: Series
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, stratify=y, test_size=self.test_train_perc, random_state=42)

            return self.balance_data(Xtrain, ytrain, Xtest, ytest) 
        
        elif self.technique == 'hdd':
            # Step 4.2.2: Apply Sampling Techniques for HDD-based partitioning.
            np.random.seed(0)

            failed, not_failed = self.get_failed_not_failed_drives(df)
            test_failed, test_not_failed = self.get_test_drives(failed, not_failed)
            test = test_failed + test_not_failed
            train = self.get_train_drives(failed, not_failed, test)

            df_train = df.loc[train, :].sort_index()
            df_test = df.loc[test, :].sort_index()

            ytrain = df_train.predict_val
            ytest = df_test.predict_val

            df_train.drop(columns=['serial_number', 'date', 'failure', 'predict_val', 'validate_val'], inplace=True)
            df_test.drop(columns=['serial_number', 'date', 'failure', 'predict_val', 'validate_val'], inplace=True)

            Xtrain = df_train.values
            Xtest = df_test.values

            if self.windowing == 1:
                # Currently we can only choose overlap == 1 since other options will cause the windows dimension to change, inconsistent with the dim of the network
                data_dim = sum(number - 1 for number in self.factors(self.window_dim)) + 1 if self.overlap != 1 else self.window_dim
                Xtrain = self.arrays_to_matrix(Xtrain, data_dim)
                Xtest = self.arrays_to_matrix(Xtest, data_dim)

            return self.balance_data(Xtrain, ytrain, Xtest, ytest)
        
        else:
            # FIXME: Step 4.2.3: Apply Sampling Techniques.
            #df.set_index('date', inplace=True)
            #df.sort_index(inplace=True)

            y = df.predict_val
            df.drop(columns=['serial_number', 'failure', 'predict_val'], inplace=True)
            X = df.values

            split_idx = int(X.shape[0] * (1 - self.test_train_perc))
            if self.windowing == 1:
                Xtrain = X[:split_idx, :, :]
                Xtest = X[split_idx:, :, :]
            else:
                Xtrain = X[:split_idx, :]
                Xtest = X[split_idx:, :]

            ytrain = y[:split_idx]
            ytest = y[split_idx:]
            return self.balance_data(Xtrain, ytrain, Xtest, ytest)

    def arrays_to_matrix(self, X, wind_dim):
        """
        Reshapes the input array X into a matrix with a specified window dimension.
        Moved into DatasetPartitioner class.

        Parameters:
        X (ndarray): The input array to be reshaped.
        wind_dim (int): The window dimension for reshaping the array.

        Returns:
        ndarray: The reshaped matrix.

        """
        X_new = X.reshape(X.shape[0], int(X.shape[1] / wind_dim), wind_dim)
        return X_new

    def preprocess_dataset(self, df):
        """
        Preprocess the dataset for splitting.

        --- Step 4.1.1: Apply sampling technique.

        Parameters:
        - df (DataFrame): The input dataframe.

        Returns:
        - DataFrame: The preprocessed dataframe.
        """
        if self.windowing != 1:
            df.drop(columns=['model', 'capacity_bytes'], inplace=True)
            return df
         
        # Replace the 'predict_val' column with a new column 'predict_val' that contains the maximum value of the 'predict_val' columns
        # Fixed RegEx problem
        predict_val_cols = [col for col in df.columns if re.match(r'^predict_val_\d+$', col)]
        df['predict_val'] = df[predict_val_cols].max(axis=1)

        # Fix the pattern to match only the exact column names with a number suffix
        base_names = ['serial_number', 'date', 'failure', 'predict_val', 'validate_val', 'capacity_bytes', 'model']
        pattern = '|'.join([f'^{base_name}_\\d+$' for base_name in base_names])  # Adjusted pattern
        columns_to_drop = [col for col in df.columns if re.match(pattern, col)]
        df.drop(columns=columns_to_drop, inplace=True)

        if self.fillna_method == 'None':
            # Record the number of rows before dropping
            # rows_before = df.shape[0]

            # # Drop rows where 'model' column has NA values
            # df.dropna(subset=['model'], inplace=True)

            # # Calculate the number of rows dropped
            # rows_dropped = rows_before - df.shape[0]

            # print(f"Dropped {rows_dropped} rows")
            rows_dropped = self.window_dim - 1 if self.overlap == 1 else 2 ** (len(self.factors(self.window_dim)) - 1) - 1
            df = df.iloc[rows_dropped:]
            logger.info(f"Dropped {rows_dropped} rows")
        else:
            # Handle NA values with padding
            df.fillna(method=self.fillna_method, inplace=True)

        # Drop missing value columns - dropped the rows based on missing values
        df.dropna(axis='columns', inplace=True)

        # Drop model, capacity_bytes columns to match exact shape when creating matrix
        df.drop(columns=['model', 'capacity_bytes'], inplace=True)
        df.set_index(['serial_number', 'date'], inplace=True)
        df.sort_index(inplace=True)

        logger.info('Dropping invalid windows')   
        df.reset_index(inplace=True)
        
        return df

    def balance_data(self, Xtrain, ytrain, Xtest, ytest):
        """
        Balance the training data using undersampling or oversampling.

        --- Step 5: Final Dataset Creation

        Parameters:
        - Xtrain (ndarray): The training data.
        - ytrain (Series): The training labels.
        - Xtest (ndarray): The test data.
        - ytest (Series): The test labels.

        Returns:
        - Xtrain (ndarray): The balanced training data.
        - Xtest (ndarray): The test data.
        - ytrain (ndarray): The balanced training labels.
        - ytest (ndarray): The test labels.
        """

        if self.oversample_undersample != 'None':
            # Define pipeline
            over = SMOTE(sampling_strategy=self.resampler_balancing, random_state=42)
            under = RandomUnderSampler(sampling_strategy=self.resampler_balancing, random_state=42)
            steps = [('o', over), ('u', under)]
            pipeline = Pipeline(steps=steps)
            Xtrain, ytrain = self.resample_windowed_data(Xtrain, ytrain, pipeline)
        else:
            ytrain = ytrain.astype(int)
            ytest = ytest.astype(int)
        ytrain = ytrain.values
        ytest = ytest.values

        return Xtrain, Xtest, ytrain, ytest

    def resample_windowed_data(self, Xtrain, ytrain, rus):
        """
        Resample the windowed training data.

        Parameters:
        - Xtrain (ndarray): The training data.
        - ytrain (Series): The training labels.
        - rus: The resampling strategy.

        Returns:
        - Xtrain (ndarray): The resampled training data.
        - ytrain (Series): The resampled training labels.
        """
        dim1 = Xtrain.shape[1]
        dim2 = Xtrain.shape[2]
        Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
        ytrain = ytrain.astype(int)
        Xtrain, ytrain = rus.fit_resample(Xtrain, ytrain)
        Xtrain = Xtrain.reshape(Xtrain.shape[0], dim1, dim2)
        return Xtrain, ytrain

    def get_failed_not_failed_drives(self, df):
        """
        Get the lists of failed and not failed drives.

        Parameters:
        - df (DataFrame): The input dataframe.

        Returns:
        - list: The list of failed drives.
        - list: The list of not failed drives.
        """

        if self.windowing == 1:
            failed = df[df.predict_val == 1].index.tolist()
        else:
            failed = df[df.failure == 1].index.tolist()

        failed_set = set(failed)
        #not_failed = [h for h in df.index if h not in failed_set]
        not_failed = []
        for i, h in enumerate(df.index):
            if h not in failed_set:
                not_failed.append(h)
            print(f"\rProcessing index: {i+1}/{len(df.index)}", end="")
        
        return failed, not_failed

    def get_test_drives(self, failed, not_failed):
        """
        Get the test drives from the lists of failed and not failed drives.

        Parameters:
        - failed (list): The list of failed drives.
        - not_failed (list): The list of not failed drives.

        Returns:
        - list: The list of failed test drives.
        - list: The list of not failed test drives.
        """
        test_failed_size = int(len(failed) * self.test_train_perc)
        test_not_failed_size = int(len(not_failed) * self.test_train_perc)

        test_failed = np.random.choice(failed, size=test_failed_size, replace=False).tolist()
        test_not_failed = np.random.choice(not_failed, size=test_not_failed_size, replace=False).tolist()
        return test_failed, test_not_failed

    def get_train_drives(self, failed, not_failed, test):
        """
        Get the training drives from the lists of failed and not failed drives.

        Parameters:
        - failed (list): The list of failed drives.
        - not_failed (list): The list of not failed drives.
        - test (list): The list of test drives.

        Returns:
        - list: The list of training drives.
        """
        # Subtract test drives from total drives to get training drives
        train_set = set(failed + not_failed) - set(test)
        # Convert the resulting set back to a list
        return list(train_set)

    def __iter__(self):
        """
        Return the training and test datasets.

        --- Step 6: Return the training and test datasets.

        Parameters:
            None
        """
        return iter((self.Xtrain, self.Xtest, self.ytrain, self.ytest))


def calculate_pearson_correlation_matrix(data):
    """
    Calculate the Pearson correlation matrix for the given data.
    """
    def safe_pearsonr(u, v):
        if np.std(u) == 0 or np.std(v) == 0:
            return np.nan
        return pearsonr(u, v)[0]
    # Fill NaN values with the mean of the column
    data_filled = data.fillna(data.mean())
    pairwise_corr = pairwise_distances(data_filled.T, metric=lambda u, v: safe_pearsonr(u, v))
    return pd.DataFrame(pairwise_corr, index=data.columns, columns=data.columns)

def find_relevant_models(df):
    """
    Calculate the correlation between different models based on smart attributes over time,
    and determine the most relevant and irrelevant models based on the correlation.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'date', 'model', 'serial_number' column, and 'smart*' attribute columns.

    Returns:
    relevant_models (list): List of relevant models.
    irrelevant_models (list): List of irrelevant models.
    """
    # Create a copy of the DataFrame and reset index
    df_copy = df.copy()
    df_copy.reset_index(inplace=True)
    
    # Select columns starting with 'smart*'
    smart_cols = [col for col in df_copy.columns if col.startswith('smart')]

    # Ensure 'date' is in datetime format
    df_copy['date'] = pd.to_datetime(df_copy['date'])

    # Sort the DataFrame by 'date'
    df_copy = df_copy.sort_values(by='date')

    # Get the unique models
    unique_models = df_copy['model'].unique()

    # Initialize a dictionary to store weighted mean correlations
    weighted_mean_correlations = {}

    # Count the occurrences of each model
    model_counts = df_copy['model'].value_counts()

    for model in tqdm(unique_models, desc="Processing models", unit='model', ncols=100):
        model_data = df_copy[df_copy['model'] == model][smart_cols].fillna(0)

        correlations = []
        for other_model in tqdm(unique_models, desc=f"Processing correlations for model {model}", leave=False, unit='model', ncols=100):
            if model != other_model:
                other_model_data = df_copy[df_copy['model'] == other_model][smart_cols].fillna(0)
                combined_data = pd.concat([model_data.add_prefix(model + '_'), other_model_data.add_prefix(other_model + '_')], axis=1)
                correlation_matrix = calculate_pearson_correlation_matrix(combined_data)
                
                for smart_col in smart_cols:
                    col1 = model + '_' + smart_col
                    col2 = other_model + '_' + smart_col
                    corr_value = correlation_matrix.loc[col1, col2]
                    correlations.append(corr_value)

        if correlations:
            # Weight the mean correlation by the count of the model
            weighted_mean_correlations[model] = np.nanmean(correlations) * model_counts[model]
        else:
            weighted_mean_correlations[model] = 0

    # Convert the weighted mean correlations to a DataFrame
    mean_corr_df = pd.DataFrame.from_dict(weighted_mean_correlations, orient='index', columns=['weighted_mean_correlation'])

    # Print the intermediate results for debugging
    print("Weighted Mean Correlations:")
    print(mean_corr_df)

    # Define the threshold for relevance
    threshold = mean_corr_df['weighted_mean_correlation'].mean()

    # Identify the most relevant and irrelevant models based on the weighted mean correlation
    relevant_models = mean_corr_df[mean_corr_df['weighted_mean_correlation'] > threshold].index.tolist()
    irrelevant_models = mean_corr_df[mean_corr_df['weighted_mean_correlation'] <= threshold].index.tolist()

    return relevant_models, irrelevant_models

def feature_selection(df, num_features, test_type):
    """
    Selects the top 'num_features' features from the given dataframe based on statistical tests.
    Step 1.4: Feature selection from Classification.py
    Args:
        df (pandas.DataFrame): The input dataframe.
        num_features (int): The number of features to select.

    Returns:
        pandas.DataFrame: The dataframe with the selected features.
    """
    # n_pop = 10
    # n_gen = 2
    # y = df['predict_val']
    # X = df.drop(columns=['predict_val'])

    # selector = GeneticFeatureSelector(X, y, n_population=n_pop, n_generation=n_gen)

    # logger.info("Running Genetic Algorithm for feature selection")
    # hof = selector.run_genetic_algorithm()

    # accuracy, individual, header = selector.best_individual()
    # logger.info(f'Best Accuracy: {accuracy}')
    # logger.info(f'Number of Features in Subset: {individual.count(1)}')
    # logger.info(f'Feature Subset: {header}')

    # df = df[header + ['predict_val']]

    # # Print the column name of the df
    # print("TEST DF Column:", list(df.columns))

    # Step 1.4.1: Define empty lists and dictionary
    features = []
    dict1 = {}

    logger.info(f'Number of feature selected for classification: {num_features}')

    # Step 1.4.2: For each feature in df.columns
    for feature in df.columns:
        # Step 1.4.2.1: if 'raw' in feature Perform T-test
        if 'raw' in feature:
            logger.info(f'Feature: {feature}')

            if feature.replace('raw', 'normalized') in df.columns:
                # (Not used) Pearson correlation to measure the linear relationship between two variables
                correlation, _ = scipy.stats.pearsonr(df[feature], df[feature.replace('raw', 'normalized')])
                logger.info(f'Pearson correlation: {correlation:.3f}')

            # Select the statistical test based on test_type
            if test_type == 't-test':
                # T-test to compare the means of two groups of features
                _, p_val = scipy.stats.ttest_ind(df[df['predict_val'] == 0][feature], df[df['predict_val'] == 1][feature], axis=0, nan_policy='omit')
                logger.info(f'T-test p-value: {p_val:.6f}')
            elif test_type == 'mannwhitneyu':
                # Mann-Whitney U test to compare distributions between two groups
                _, p_val = scipy.stats.mannwhitneyu(df[df['predict_val'] == 0][feature], df[df['predict_val'] == 1][feature], alternative='two-sided')
                logger.info(f'Mann-Whitney U test p-value: {p_val:.6f}')
            else:
                raise ValueError(f'Invalid test type: {test_type}')

            dict1[feature] = p_val

    logger.info('Sorting features')

    # Step 1.4.2.2: Sort the features based on the p-values (item[1] is used to sort the dictionary by its value, and item[0] is used to sort by key)
    features = {k: v for k, v in sorted(dict1.items(), key=lambda item: item[1])}
    # Step 1.4.2.3: Convert dictionary to DataFrame and drop NaNs
    features = pd.DataFrame(features.items(), index=features.keys()).dropna()
    # Step 1.4.2.4: Select top 'num_features' features
    features = features[:num_features][0].values
    for feature in df.columns:
        if 'smart' not in feature:
            features = np.concatenate((features, np.asarray(feature).reshape(1,)))
    # Step 1.4.2.5: Update df to only include selected features
    df = df[features]
    return df

if __name__ == '__main__':
    features = {
        'total_features': [
            'date',
            'serial_number',
            'model',
            'failure',
            'smart_1_normalized',
            'smart_5_normalized',
            'smart_5_raw',
            'smart_7_normalized',
            'smart_9_raw',
            'smart_12_raw',
            'smart_183_raw',
            'smart_184_normalized',
            'smart_184_raw',
            'smart_187_normalized',
            'smart_187_raw',
            'smart_189_normalized',
            'smart_193_normalized',
            'smart_193_raw',
            'smart_197_normalized',
            'smart_197_raw',
            'smart_198_normalized',
            'smart_198_raw',
            'smart_199_raw'
        ]
    }

    model = 'ST3000DM001'
    years = ['2013', '2014', '2015', '2016', '2017']
    df = import_data(years, model, features)
    logger.info('Data imported successfully, processing smart attributes...')
    for column in list(df):
        missing = round(df[column].notna().sum() / df.shape[0] * 100, 2)
        logger.info(f"{column:.<27}{missing}%")
    # drop bad HDs
    
    bad_missing_hds, bad_power_hds, df = filter_HDs_out(df, min_days = 30, time_window='30D', tolerance=2)
    df['predict_val'] = generate_failure_predictions(df, days=7) # define RUL piecewise
    ## -------- ##
    # random: stratified without keeping time
    # hdd --> separate different hdd
    # temporal --> separate by time
    Xtrain, Xtest, ytrain, ytest = DatasetPartitioner(df, technique = 'random')
    #method = 'linear'
    #df = interpolate_ts(df, method=method)
