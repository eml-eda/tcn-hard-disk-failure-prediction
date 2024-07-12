import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import re
import dask.dataframe as dd
from tqdm import tqdm
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class DatasetProcessing:
    """
        https://github.com/Prognostika/tcn-hard-disk-failure-prediction/wiki/Code_Process#partition-dataset-subflowchart
    """
    def __init__(self, df, overlap=0, windowing=1, window_dim=5, days=7, smoothing_level=0.5, augmentation_method='duplicate'):
        """
        Initialize the DatasetProcessing object.
        
        Parameters:
        - df (DataFrame): The input dataset.
        - overlap (int): The overlap value for windowing (default: 0).
        - windowing (int): The windowing value (default: 1).
        - window_dim (int): The window dimension (default: 5).
        - days (int): The number of days (default: 7).
        - smoothing_level (float): The smoothing level (default: 0.5).
        - augmentation_method (str): The augmentation method (default: 'duplicate').

        """
        self.df = df
        self.overlap = overlap
        self.windowing = windowing
        self.window_dim = window_dim
        self.days = days
        self.smoothing_level = smoothing_level
        self.augmentation_method = augmentation_method
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.X = self.partition()

    def partition(self):
        """
        Partition the dataset into training and test sets.

        Parameters:
            None

        Returns:
        - X (ndarray): The partitioned dataset.
        """
        self.df.reset_index(inplace=True) # Step 1.1: Reset index.
        # Step 1.2: Preprocess the dataset.
        mms = MinMaxScaler(feature_range=(0, 1)) # Normalize the dataset

        # Extract temporal data
        # Updated: temporal now also drops 'model' and 'capacity_bytes' columns, because they are object. We need float64.
        temporal = self.df[['serial_number', 'date', 'failure', 'model', 'capacity_bytes']]
        self.df.drop(columns=temporal.columns, inplace=True)
        self.df = pd.DataFrame(mms.fit_transform(self.df), columns=self.df.columns, index=self.df.index)
        # self.df is now normalized, but temporal is original string data, to avoid normalization of 'serial_number' and 'date' and other non float64 columns
        self.df = pd.concat([self.df, temporal], axis=1)

        # Repeat the data until it reaches the required length
        if self.windowing == 1:
            group_sizes = self.days + self.window_dim
            # If the length of the df is less than group_sizes
            if len(self.df) < group_sizes:
                if self.augmentation_method == 'duplicate':
                    while len(self.df) < group_sizes:
                        self.df = pd.concat([self.df, self.df], ignore_index=True)
                elif self.augmentation_method == 'interpolate':
                    # Calculate the interpolation factor
                    interp_factor = group_sizes / len(self.df)

                    # Create a new index for interpolation
                    new_index = np.arange(0, len(self.df), 1/interp_factor)

                    # Interpolate the data
                    self.df = self.df.reindex(new_index).interpolate(method='index')
                else:
                    raise ValueError("Invalid method. Choose either 'duplicate' or 'interpolate'.")

                # Trim the df to the required length
                self.df = self.df.iloc[:group_sizes]

        print ('Windowing the df')
        windowed_df = self.perform_windowing()

        print('Preprocessing test dataset')
        return self.preprocess_dataset(windowed_df)

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
        else:  # FIXME: If the overlap option is chosed as dynamic overlap based on the factors of window_dim
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
                # It is suitable for inference when you need to preprocess new data in a way consistent with the training data preprocessing but without labels.
                indexes = windowed_df.groupby(serials).apply(lambda x: x.iloc[::down_factor].index)
                # Update windowed_df based on the indexes
                windowed_df = windowed_df.loc[np.concatenate(indexes.values.tolist(), axis=0), :]
                # Convert back to Dask DataFrame
                windowed_df = dd.from_pandas(windowed_df, npartitions=int(len(windowed_df)/chunk_columns) + 1)

        # Compute the final Dask DataFrame to pandas DataFrame
        final_df = windowed_df.compute()
        # Handle NA values with padding
        final_df = final_df.fillna(method='ffill')
        
        #print('perform_windowing:', self.df.columns)

        return self.rename_columns(final_df)

    def arrays_to_matrix(self, X, wind_dim):
        """
        Reshapes the input array X into a matrix with a specified window dimension.
        Moved into DatasetProcessing class.

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
        - X (ndarray): The preprocessed dataset.
        """
        if self.windowing == 1:
            # Fix the pattern to match only the exact column names with a number suffix
            base_names = ['serial_number', 'date', 'capacity_bytes', 'model']
            pattern = '|'.join([f'^{base_name}_\\d+$' for base_name in base_names])  # Adjusted pattern
            columns_to_drop = [col for col in df.columns if re.match(pattern, col)]
            df.drop(columns=columns_to_drop, inplace=True)

            # If we found some data about the 'date', 'serial_number', 'capacity_bytes' are missing, we should drop them.
            essential_columns = ['date', 'serial_number', 'capacity_bytes']

            # Identify any missing columns in the DataFrame
            missing_columns = [col for col in essential_columns if col not in df.columns]

            # If no essential columns are missing, drop rows with missing data in these columns
            if not missing_columns:
                # Record the number of rows before dropping
                rows_before = df.shape[0]
                
                # Drop rows where any of the essential columns have missing data
                df.dropna(subset=essential_columns, inplace=True)
                
                # Record the number of rows after dropping
                rows_after = df.shape[0]

                # Calculate the number of rows dropped
                rows_dropped = rows_before - rows_after

                print(f"Dropped {rows_dropped} rows")
            else:
                # Print an error message if essential columns are missing
                print(f"Columns {missing_columns} do not exist in the DataFrame.")

            # Drop missing value columns - dropped the rows based on missing values
            df.dropna(axis='columns', inplace=True)

            # Drop model, capacity_bytes columns to match exact shape when creating matrix
            # df.drop(columns=['model', 'capacity_bytes'], inplace=True)
            df.set_index(['serial_number', 'date'], inplace=True)
            df.sort_index(inplace=True)

            print('Dropping invalid windows')   
            # print(df.columns)
            df.reset_index(inplace=True)

        #########################
        df.drop(columns=['serial_number', 'date', 'model', 'capacity_bytes'], inplace=True)
        
        # TODO:
        for col in tqdm(df.columns, desc="Processing columns", leave=False, unit="column", ncols=100):
            if col.startswith('smart'):
                df[col] = ExponentialSmoothing(df[col], trend=None, seasonal=None, seasonal_periods=None).fit(smoothing_level=self.smoothing_level).fittedvalues

        X = df.values   # X represents the smart features (ndarray)

        if self.windowing == 1:
            # If we use the down sampling, then the data_dim will be the sum of the factors of the window_dim
            data_dim = sum(number - 1 for number in self.factors(self.window_dim)) + 1 if self.overlap != 1 else self.window_dim
            X = self.arrays_to_matrix(X, data_dim)
        else:
            X = np.expand_dims(X, axis=1)  # Add an extra dimension

        return X

    def __iter__(self):
        """
        Return the training and test datasets.

        --- Step 6: Return the training and test datasets.

        Parameters:
            None
        """
        return iter(self.X)