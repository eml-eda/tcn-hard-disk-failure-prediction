import pandas as pd
# import datetime
import numpy as np
from numpy import *
import math
import pickle
# from scipy.stats.stats import pearsonr
# import sys
# from sklearn.utils import shuffle
import os
import matplotlib.pyplot as plt
import glob
# from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import scipy
#import pdb;pdb.set_trace()
import scipy.stats
#
import matplotlib.pyplot as plt
#
import sys
import re
## here there are many functions used inside Classification.py


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
    fig,ax = plt.subplots()
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
        'Xiao_et_al': [
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
    k = 0
    for i in [1, 2, 7, 8, 9, 10]:
        ax.plot(np.arange(X.shape[0]), X[:, i] + 0.01 * k, label=features['Xiao_et_al'][i + 2])
        k += 1
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
    join_years = '_'.join(years) + '_'
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

def matrix3d_to_datasets(matrix, window=1, divide_hdd=1, training_percentage=0.7, resampler_balancing=5, oversample_undersample=0):
    """
    Convert a 3D matrix to datasets for training and testing.

    Args:
        matrix (ndarray): The 3D matrix containing the data.
        window (int, optional): The size of the sliding window. Defaults to 1.
        divide_hdd (int, optional): Flag to divide the HDDs. Defaults to 1.
        training_percentage (float, optional): The percentage of data to use for training. Defaults to 0.7.
        resampler_balancing (int, optional): The resampler balancing factor. Defaults to 5.
        oversample_undersample (int, optional): The type of resampling to use. Defaults to 0.

    Returns:
        dict: A dictionary containing the training and testing datasets.

    Raises:
        FileNotFoundError: If the dataset file is not found.
    """

    name_file = 'Final_Dataset.pkl'
    read_dir = os.path.join('..', 'data_input')

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
            resampler = RandomUnderSampler(1 / resampler_balancing, random_state=42) if oversample_undersample == 0 else SMOTE(1 / resampler_balancing, random_state=42)
            X_train, Y_train = resampler.fit_resample(X_train, Y_train)

            dataset = {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test, 'HDn_test': HD_number_test}

            with open(os.path.join(read_dir, name_file), 'wb') as handle:
                pickle.dump(dataset, handle)

    return dataset

def import_data(years, model, name, **args):
    """ Import hard drive data from csvs on disk.
    
    :param quarters: List of quarters to import (e.g., 1Q19, 4Q18, 3Q18, etc.)
    :param model: String of the hard drive model number to import.
    :param columns: List of the columns to import.
    :return: Dataframe with hard drive data.
    
    """

    # Read the correct .pkl file
    years_list = '_' + '_'.join(years)
    failed = False  # This should be set based on your specific criteria or kept as a placeholder
    suffix = 'failed' if failed else 'all'

    # Fix the directory name as output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(script_dir, '..', 'output', f'HDD{years_list}_{suffix}_{model}_appended.pkl')

    if not os.path.exists(file):
        # Fix the directory name
        file = os.path.join(script_dir, '..', 'output', f'HDD{years_list}_{model}_all.pkl')

    try:
        df = pd.read_pickle(file)
        print(f'Data loaded from {file}')
    except FileNotFoundError:
        print('Creating new DataFrame from CSV files.')
        cwd = os.getcwd()
        all_data = []

        for y in years:
            print(f'Analyzing year {y}', end="\r")
            # Fix the directory name
            for f in glob.glob(os.path.join(cwd, '..', 'HDD_dataset', y, '*.csv')):
                try:
                    data = pd.read_csv(f, header=0, usecols=args['features'][name], parse_dates=['date'])
                except ValueError:
                    data = pd.read_csv(f, header=0, parse_dates=['date'])
                
                data = data[data.model == model].copy()
                data.drop(columns=['model'], inplace=True)
                data.failure = data.failure.astype('int')
                all_data.append(data)

        df = pd.concat(all_data, ignore_index=True)
        df.set_index(['serial_number', 'date'], inplace=True)
        df.sort_index(inplace=True)
        df.to_pickle(file)
        print(f'Data saved to {file}')

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
    for serial_num, inner_df in df.groupby(level=0):
        if len(inner_df) < min_days:  # identify HDs with too few power-on days
            bad_power_hds.append(serial_num)

        inner_df = inner_df.droplevel(level=0)
        inner_df = inner_df.asfreq('D')  # Convert inner_df to daily frequency

        # Find the maximum number of missing values within any window of time_window days in any column of the DataFrame
        # Commented out to avoid to many missing values
        # n_missing = max(inner_df.isna().rolling(time_window).sum().max()) # original code DO NOT CHANGE

        # if n_missing >= tolerance:  # identify HDs with too many missing values #original code DO NOT CHANGE
        #     bad_missing_hds.append(serial_num) # original code DO NOT CHANGE



    bad_hds = set(bad_missing_hds + bad_power_hds)
    print(f"Filter result: bad_missing_hds: {len(bad_missing_hds)}, bad_power_hds: {len(bad_power_hds)}")
    hds_remove = len(bad_hds)
    hds_total = len(df.reset_index().serial_number.unique())
    print('Total HDs: {}    HDs removed: {} ({}%)'.format(hds_total, hds_remove, round(hds_remove / hds_total * 100, 2)))

    df = df.drop(bad_hds, axis=0)

    num_fail = df.failure.sum()
    num_not_fail = len(df.reset_index().serial_number.unique()) - num_fail
    pct_fail = num_fail / (num_not_fail + num_fail) * 100 

    print('{} failed'.format(num_fail))
    print('{} did not fail'.format(num_not_fail))
    print('{:5f}% failed'.format(pct_fail))
    return bad_missing_hds, bad_power_hds, df

def interpolate_ts(df, method='linear'):
    
    """ Interpolate hard drive Smart attribute time series.
    
    :param df: Input dataframe.
    :param method: String, interpolation method.
    :return: Dataframe with interpolated values.
    """
    
    interp_df = pd.DataFrame()

    for serial_num, inner_df in df.groupby(level=0):
        inner_df = inner_df.droplevel(level=0)
        inner_df = inner_df.asfreq('D') 
        inner_df.interpolate(method=method, axis=0, inplace=True)
        inner_df['serial_number'] = serial_num
        inner_df = inner_df.reset_index()

        interp_df = pd.concat([interp_df, inner_df], axis=0)

    df = interp_df.set_index(['serial_number', 'date']).sort_index()
    
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
    i = 0
    for serial_num, inner_df in df.groupby(level=0):
        print('Analyzing HD {} number {} \r'.format(serial_num,i), end="\r")
        slicer_val = len(inner_df)  # save len(df) to use as slicer value on smooth_smart_9 
        i += 1
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
    print('HDs analyzed: {}'.format(i))
    pred_list = np.asarray(pred_list)
    valid_list = np.asarray(valid_list)
    return pred_list, valid_list

def arrays_to_matrix(X, wind_dim):
    """
    Reshapes the input array X into a matrix with a specified window dimension.

    Parameters:
    X (ndarray): The input array to be reshaped.
    wind_dim (int): The window dimension for reshaping the array.

    Returns:
    ndarray: The reshaped matrix.

    """
    X_new = X.reshape(X.shape[0], int(X.shape[1] / wind_dim), wind_dim)
    return X_new

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


class DatasetPartitioner:
    """
        https://github.com/Prognostika/tcn-hard-disk-failure-prediction/wiki/Code_Process#partition-dataset-subflowchart
    """
    def __init__(self, df, model, overlap=0, rank='None', num_features=10, technique='random',
                 test_train_perc=0.2, windowing=1, window_dim=5, resampler_balancing=5, oversample_undersample=0):
        """
        Initialize the DatasetPartitioner object.
        
        Parameters:
        - df (DataFrame): The input dataset.
        - model (str): The name of the model.
        - overlap (int): The overlap value for windowing (default: 0).
        - rank (str): The rank value (default: 'None').
        - num_features (int): The number of features (default: 10).
        - technique (str): The partitioning technique (default: 'random').
        - test_train_perc (float): The percentage of data to be used for testing (default: 0.2).
        - windowing (int): The windowing value (default: 1).
        - window_dim (int): The window dimension (default: 5).
        - resampler_balancing (int): The resampler balancing value (default: 5).
        - oversample_undersample (int): The oversample/undersample value (default: 0).

        """
        self.df = df
        self.model = model
        self.overlap = overlap
        self.rank = rank
        self.num_features = num_features
        self.technique = technique
        self.test_train_perc = test_train_perc
        self.windowing = windowing
        self.window_dim = window_dim
        self.resampler_balancing = resampler_balancing
        self.oversample_undersample = oversample_undersample
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = self.partition()

    def partition(self):
        """
        Partition the dataset into training and test sets.


        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.

        Returns:
        - Xtrain (ndarray): The training data.
        - Xtest (ndarray): The test data.
        - ytrain (Series): The training labels.
        - ytest (Series): The test labels.
        """
        self.df.reset_index(inplace=True) # Step 1.1: Reset index.
        print("DF index name:", self.df.index.names)

        # Step 1.2: Preprocess the dataset.
        mms = MinMaxScaler(feature_range=(0, 1)) # Normalize the dataset

        # Extract temporal data
        # Updated: temporal now also drops 'model' and 'capacity_bytes' columns, because they are object. We need float64.
        temporal = self.df[['serial_number', 'date', 'failure', 'predict_val', 'validate_val', 'model', 'capacity_bytes']]
        self.df.drop(columns=temporal.columns, inplace=True)
        self.df = pd.DataFrame(mms.fit_transform(self.df),
        columns=self.df.columns, index=self.df.index)  # FIXME: 
        self.df = pd.concat([self.df, temporal], axis=1)

        windowed_df = self.handle_windowing()

        print('Creating training and test dataset')
        return self.split_dataset(windowed_df)
    
    def factors(self, n):
        """
        Returns a list of factors of the given number.

        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.
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
        - self (DatasetPartitioner): The DatasetPartitioner object.

        Returns:
        - DataFrame: The windowed dataset.
        """

        # Step 2: Check Windowing
        if self.windowing != 1:
            return self.df

        try:
            # Step 2.1.1: If Yes, attempt to load the pre-processed windowed dataset.
            windowed_df = pd.read_pickle(os.path.join(self.script_dir, '..', 'output', f'{self.model}_Dataset_windowed_{self.window_dim}_rank_{self.rank}_{self.num_features}_overlap_{self.overlap}.pkl'))
            print('Loading the windowed dataset')
            return self.rename_columns(windowed_df)
        except FileNotFoundError:
            # Step 2.1.2: If No, perform windowing on the dataset.
            print('Windowing the df')  # FIXME: Currently all columns are indexed.
            return self.perform_windowing()

    def rename_columns(self, df):
        """
        Rename the columns of the dataframe to avoid duplicates.
        --- Step 3: Prepare data for modeling.
        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.
        - df (DataFrame): The input dataframe.

        Returns:
        - DataFrame: The dataframe with renamed columns.
        """

        cols = []
        count = {}
        print('\nTEST', df.columns)
        for column in df.columns:
            if column not in count:
                count[column] = 0
            count[column] += 1
            new_column = f"{column}_{count[column]}" if count[column] > 1 else column
            cols.append(new_column)
        df.columns = cols
        df.sort_index(axis=1, inplace=True)

        print('\nTest: ',count['predict_val']) 
        return df

    # def process_in_chunks(self, down_factor=0, previous_down_factor=0, total_shifts=0):
    #     chunk_size = 10000  # adjust this value to a size that fits comfortably in your memory
    #     num_chunks = len(self.df) // chunk_size + 1

    #     for chunk_id in range(num_chunks):
    #         start = chunk_id * chunk_size
    #         end = start + chunk_size
    #         df_chunk = self.df.iloc[start:end].copy()

    #         windowed_df = df_chunk.copy()
    #         if down_factor != 0:
    #             # for i in np.arange(down_factor - 1):
    #             #     total_shifts += previous_down_factor
    #             #     print(f'Concatenating time - {total_shifts}', end="\n")
    #             #     windowed_df = pd.concat([self.df.shift(i + 1), windowed_df], axis=1)
    #             total_shifts = 0
    #             for i in np.arange(down_factor - 1):
    #                 total_shifts += previous_down_factor
    #                 print(f'Concatenating time - {total_shifts}', end="\n")
    #                 try:
    #                     shifted_df = df_chunk.shift(i + 1)
    #                     windowed_df = pd.concat([shifted_df, windowed_df], axis=1)
    #                     del shifted_df  # delete the temporary dataframe to free up memory
    #                 except Exception as e:
    #                     print(f"An error occurred: {e}")
    #                 print(f'Finished iteration {i}')

    #         else: 
    #             for i in np.arange(self.window_dim - 1):
    #                 print(f'Concatenating time - {i}', end="\n")
    #                 try:
    #                     shifted_df = df_chunk.shift(i + 1)
    #                     windowed_df = pd.concat([shifted_df, windowed_df], axis=1)
    #                     del shifted_df  # delete the temporary dataframe to free up memory
    #                 except Exception as e:
    #                     print(f"An error occurred: {e}")
    #                 print(f'Finished iteration {i}')

    #         # Write the windowed dataframe to a binary file
    #         filename = f'windowed_df_{chunk_id}.pkl'
    #         with open(filename, 'wb') as f:
    #             pickle.dump(windowed_df, f)
    #         print(f'Wrote chunk {chunk_id} to {filename}')

    #     # Combine all the binary files into one dataframe
    #     pkl_files = [f for f in os.listdir() if f.startswith('windowed_df_')]
    #     df_combined = pd.concat((pickle.load(open(f, 'rb')) for f in pkl_files))

    #     # Clean up the binary files
    #     for f in pkl_files:
    #         os.remove(f)

    #     if down_factor != 0:
    #         return df_combined
    #     else:
    #         return df_combined, down_factor
    def process_in_chunks(self, down_factor=0):
        """
        Process the dataframe in chunks to avoid memory issues.

        Parameters:
        - self: The DatasetPartitioner object.
        - down_factor: The downsample factor.

        Returns:
        - DataFrame: The processed dataframe.
        - int: The downsample factor.
        """
        # Define the size of each chunk
        chunk_size = 10000  # adjust this value to a size that fits comfortably in your memory

        # Calculate the number of chunks
        num_chunks = len(self.df) // chunk_size + 1

        # Initialize an empty dataframe to store the combined result
        df_combined = pd.DataFrame()

        # Process each chunk
        for chunk_id in range(num_chunks):
            # Calculate the start and end indices for this chunk
            start = chunk_id * chunk_size
            end = start + chunk_size

            # Extract the chunk from the dataframe
            df_chunk = self.df.iloc[start:end].copy()

            # Initialize a copy of the chunk to store the windowed result
            windowed_df = df_chunk.copy()

            # Determine the range of shifts based on the downsample factor
            if down_factor != 0:
                shift_range = np.arange(down_factor - 1)
            else:
                shift_range = np.arange(self.window_dim - 1)

            # Perform the shifts and concatenation
            for i in shift_range:
                print(f'Concatenating time - {i}', end="\n")
                try:
                    # Shift the dataframe
                    shifted_df = df_chunk.shift(i + 1)

                    # Concatenate the shifted dataframe with the windowed dataframe
                    windowed_df = pd.concat([shifted_df, windowed_df], axis=1)

                    # Delete the temporary dataframe to free up memory
                    del shifted_df
                except Exception as e:
                    # Print an error message if an exception occurs
                    print(f"An error occurred: {type(e).__name__}, {e}")

                print(f'Finished iteration {i}')

            # Concatenate the windowed dataframe with the combined dataframe
            df_combined = pd.concat([df_combined, windowed_df])

            print(f'Processed chunk {chunk_id}')

        # Return the combined dataframe and the downsample factor
        return df_combined, down_factor if down_factor != 0 else df_combined

    def perform_windowing(self):
        """
        Perform the windowing operation on the dataset.
        --- Step 2.1.2: Perform windowing on the dataset.
        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.

        Returns:
        - DataFrame: The windowed dataframe.
        """
        windowed_df = self.df.copy()
        if self.overlap == 1:
            # for i in np.arange(self.window_dim - 1):
                # print(f'Concatenating time - {i}', end="\n") # To keep the last iteration info, change '\r' to '\n'
                # Shift the dataframe and concatenate along the columns
                # windowed_df = pd.concat([self.df.shift(i + 1), windowed_df], axis=1) # We enter the "if" block, we will not have duplicate columns because we did not downsample
            windowed_df = self.process_in_chunks()
        else:
            # Get the factors of window_dim
            window_dim_divisors = self.factors(self.window_dim)
            total_shifts = 0
            previous_down_factor = 1
            serials = self.df.serial_number
            # print("TEST window_dim_divisors", window_dim_divisors)
            for down_factor in window_dim_divisors: # window_dim_divisors = [2,2,2,2,2]
                # Shift the dataframe by the factor and concatenate
                # for i in np.arange(down_factor - 1):
                #     total_shifts += previous_down_factor
                #     print(f'Concatenating time - {total_shifts}', end="\n")
                #     windowed_df = pd.concat([self.df.shift(i + 1), windowed_df], axis=1)
                windowed_df, down_factor = self.process_in_chunks(down_factor)
                previous_down_factor *= down_factor
                # Under sample the dataframe based on the serial numbers and the factor
                indexes = windowed_df.groupby(serials).apply(self.under_sample, down_factor)
                # Update windowed_df based on the indexes
                windowed_df = windowed_df.loc[np.concatenate(indexes.values.tolist(), axis=0), :]
                # Update the original dataframe
                self.df = windowed_df
        # TODO: We need to test the generated file here
        # pd.read_pickle(os.path.join(self.script_dir, '..', 'output', f'{self.model}_Dataset_windowed_{self.window_dim}_rank_{self.rank}_{self.num_features}_overlap_{self.overlap}.pkl'))
        print("Concatenating finished")
        return self.rename_columns(windowed_df)

    def split_dataset(self, df):
        """
        Split the dataset into training and test sets based on the specified technique.

        --- Step 4: Technique selection.

        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.
        - df (DataFrame): The input dataframe.

        Returns:
        - Xtrain (ndarray): The training data.
        - Xtest (ndarray): The test data.
        - ytrain (Series): The training labels.
        - ytest (Series): The test labels.
        """
        if self.technique == 'random':
            return self.random_split(df) # Step 4.1: Random partitioning
        elif self.technique == 'hdd':
            return self.hdd_split(df)  # Step 4.2: HDD partitioning
        else:
            return self.date_split(df)  # Step 4.3: Other techniques

    def random_split(self, df):
        """
        Randomly split the dataset into training and test sets.

        --- Step 4.1: Random partitioning.

        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.
        - df (DataFrame): The input dataframe.

        Returns:
        - Xtrain (ndarray): The training data.
        - Xtest (ndarray): The test data.
        - ytrain (Series): The training labels.
        - ytest (Series): The test labels.
        """
        df = self.preprocess_random(df)
        y = df['predict_val']
        df.drop(columns=['serial_number', 'date', 'failure', 'predict_val', 'validate_val'], inplace=True)
        X = df.values

        if self.windowing == 1:
            X = self.arrays_to_matrix(X)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, stratify=y, test_size=self.test_train_perc, random_state=42)
        return self.balance_data(Xtrain, ytrain, Xtest, ytest)

    def preprocess_random(self, df):
        """
        Preprocess the dataset for random splitting.

        --- Step 4.1.1: Apply sampling technique.

        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.
        - df (DataFrame): The input dataframe.

        Returns:
        - DataFrame: The preprocessed dataframe.
        """
        if self.windowing != 1:
            return df
        print(df.columns)
        # df['predict_val'] = df[f'predict_val_{self.window_dim - 1}']    #FIXME: If we did not downsample, there will be no duplicate columns. 
        # Replace the 'predict_val' column with a new column 'predict_val' that contains the maximum value of the 'predict_val' columns
        predict_val_cols = [col for col in df.columns if re.match(r'predict_val_\d+', col)]
        df['predict_val'] = df[predict_val_cols].max(axis=1)

        # FIXME: Drop the columns serial_number, date, failure, predict_val_0, predict_val_1, predict_val_2, predict_val_3, predict_val_4, validate_val_0, validate_val_1, validate_val_2, validate_val_3, validate_val_4
        base_names = ['serial_number_', 'date_', 'failure_', 'predict_val_', 'validate_val_']
        columns_to_drop = [col for col in df.columns if any(re.match(f"{base_name}\d+", col) for base_name in base_names)]
        df.drop(columns=columns_to_drop, inplace=True)

        df.dropna(inplace=True)
        df.set_index(['serial_number', 'date'], inplace=True)
        df.sort_index(inplace=True)
        indexes = self.get_invalid_indexes(df)
        print('Dropping invalid windows ')
        df.drop(indexes, inplace=True)
        df.reset_index(inplace=True)
        return df

    def get_invalid_indexes(self, df):
        """
        Get the indexes of invalid windows in the dataframe.

        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.
        - df (DataFrame): The input dataframe.

        Returns:
        - list: The indexes of invalid windows.
        """
        indexes = []
        for serial_num, inner_df in df.groupby(level=0):
            print(f'Computing index of invalid windows of HD number {serial_num} \r', end="\r")
            index_to_append = inner_df[:self.window_dim].index if self.overlap == 1 else inner_df[:1].index
            indexes.append(index_to_append)
        return indexes

    def balance_data(self, Xtrain, ytrain, Xtest, ytest):
        """
        Balance the training data using undersampling or oversampling.

        --- Step 5: Final Dataset Creation

        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.
        - Xtrain (ndarray): The training data.
        - ytrain (Series): The training labels.
        - Xtest (ndarray): The test data.
        - ytest (Series): The test labels.

        Returns:
        - Xtrain (ndarray): The balanced training data.
        - Xtest (ndarray): The test data.
        - ytrain (Series): The balanced training labels.
        - ytest (Series): The test labels.
        """
        resampler = RandomUnderSampler(1 / self.resampler_balancing, random_state=42) if self.oversample_undersample == 0 else SMOTE(1 / self.resampler_balancing, random_state=42)

        if self.oversample_undersample != 2:
            Xtrain, ytrain = self.resample_windowed_data(Xtrain, ytrain, resampler) if self.windowing else resampler.fit_resample(Xtrain, ytrain)
        else:
            ytrain = ytrain.astype(int)
            ytest = ytest.astype(int)
            ytrain = ytrain.values

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
        ytest = ytest.astype(int)
        return Xtrain, ytrain

    def hdd_split(self, df):
        """
        Split the dataset based on HDD failure.

        --- Step 4.2: HDD partitioning.

        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.
        - df (DataFrame): The input dataframe.

        Returns:
        - Xtrain (ndarray): The training data.
        - Xtest (ndarray): The test data.
        - ytrain (Series): The training labels.
        - ytest (Series): The test labels.
        """

        # Step 4.2.2: Apply Sampling Techniques.
        np.random.seed(0)
        df.set_index(['serial_number', 'date'], inplace=True)
        df.sort_index(inplace=True)

        failed, not_failed = self.get_failed_not_failed_drives(df)
        test_failed, test_not_failed = self.get_test_drives(failed, not_failed)
        test = test_failed + test_not_failed
        train = self.get_train_drives(failed, not_failed, test)

        df_train = df.loc[train, :].sort_index()
        df_test = df.loc[test, :].sort_index()
        df_train.reset_index(inplace=True)
        df_test.reset_index(inplace=True)

        ytrain = df_train.predict_val
        ytest = df_test.predict_val

        df_train.drop(columns=['serial_number', 'date', 'failure', 'predict_val'], inplace=True)
        df_test.drop(columns=['serial_number', 'date', 'failure', 'predict_val'], inplace=True)

        Xtrain = df_train.values
        Xtest = df_test.values

        if self.windowing == 1:
            Xtrain = self.arrays_to_matrix(Xtrain)
            Xtest = self.arrays_to_matrix(Xtest)

        return self.balance_data(Xtrain, ytrain, Xtest, ytest)

    def get_failed_not_failed_drives(self, df):
        """
        Get the lists of failed and not failed drives.

        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.
        - df (DataFrame): The input dataframe.

        Returns:
        - list: The list of failed drives.
        - list: The list of not failed drives.
        """
        if self.windowing == 1:
            failed = [h[0] for h in list(df[df.predict_val == 1].index)]
        else:
            failed = [h[0] for h in list(df[df.failure == 1].index)]
        failed = list(set(failed))

        not_failed = [h[0] for h in list(df.index) if h[0] not in failed]
        not_failed = list(set(not_failed))
        return failed, not_failed

    def get_test_drives(self, failed, not_failed):
        """
        Get the test drives from the lists of failed and not failed drives.

        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.
        - failed (list): The list of failed drives.
        - not_failed (list): The list of not failed drives.

        Returns:
        - list: The list of failed test drives.
        - list: The list of not failed test drives.
        """
        test_failed = list(np.random.choice(failed, size=int(len(failed) * self.test_train_perc)))
        test_not_failed = list(np.random.choice(not_failed, size=int(len(not_failed) * self.test_train_perc)))
        return test_failed, test_not_failed

    def get_train_drives(self, failed, not_failed, test):
        """
        Get the training drives from the lists of failed and not failed drives.

        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.
        - failed (list): The list of failed drives.
        - not_failed (list): The list of not failed drives.
        - test (list): The list of test drives.

        Returns:
        - list: The list of training drives.
        """
        train = failed + not_failed
        train = list(filter(lambda x: x not in test, train))
        return train

    def date_split(self, df):
        """
        Split the dataset based on date.

        --- Step 4.3: Other techniques.

        Parameters:
        - df (DataFrame): The input dataframe.

        Returns:
        - Xtrain (ndarray): The training data.
        - Xtest (ndarray): The test data.
        - ytrain (Series): The training labels.
        - ytest (Series): The test labels.
        """

        # Step 4.2.3: Apply Sampling Techniques.
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

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

    def __iter__(self):
        """
        Return the training and test datasets.

        --- Step 6: Return the training and test datasets.

        Parameters:
        - self (DatasetPartitioner): The DatasetPartitioner object.
        """
        return iter((self.Xtrain, self.Xtest, self.ytrain, self.ytest))

def feature_selection(df, num_features):
    """
    Selects the top 'num_features' features from the given dataframe based on statistical tests.
    Step 1.4: Feature selection from Classification.py
    Args:
        df (pandas.DataFrame): The input dataframe.
        num_features (int): The number of features to select.

    Returns:
        pandas.DataFrame: The dataframe with the selected features.
    """
    # Step 1.4.1: Define empty lists and dictionary
    features = []
    p = []
    dict1 = {}

    print('Feature selection')

    # Step 1.4.2: For each feature in df.columns
    for feature in df.columns:
        # Step 1.4.2.1: if 'raw' in feature Perform T-test
        if 'raw' in feature:
            print('Feature: {}'.format(feature))
        
            # (Not used) Pearson correlation to measure the linear relationship between two variables
            correlation, _ = scipy.stats.pearsonr(df[feature], df[feature.replace('raw', 'normalized')])
            print('Pearson correlation: %.3f' % correlation)
            
            # T-test to compare the means of two groups of features
            _, p_val = scipy.stats.ttest_ind(df[df['predict_val'] == 0][feature], df[df['predict_val'] == 1][feature], axis=0, nan_policy='omit')
            print('T-test p-value: %.3f' % p_val)

            dict1[feature] = p_val

    print('Sorting features')

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
        'Xiao_et_al': [
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
    #dataset = dataset[features['Xiao_et_al']]
    model = 'ST3000DM001'
    years = ['2013', '2014', '2015', '2016', '2017']
    df = import_data(years, model, features)
    print(df.head())
    for column in list(df):
        missing = round(df[column].notna().sum() / df.shape[0] * 100, 2)
        print('{:.<27}{}%'.format(column, missing))
    # drop bad HDs
    
    bad_missing_hds, bad_power_hds, df = filter_HDs_out(df, min_days = 30, time_window='30D', tolerance=30)
    df['predict_val'] = generate_failure_predictions(df, days=7) # define RUL piecewise
    ## -------- ##
    # random: stratified without keeping time
    # hdd --> separate different hdd
    # temporal --> separate by time
    Xtrain, Xtest, ytrain, ytest = DatasetPartitioner(df, technique = 'random')
    #method = 'linear'
    #df = interpolate_ts(df, method=method)
