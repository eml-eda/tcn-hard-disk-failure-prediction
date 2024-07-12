import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


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
