import numpy as np
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors

class tSMOTE:
    def __init__(self, k_neighbors=5, random_state=None, n_slices=10):
        """
        Initialize the tSMOTE object.

        Parameters:
        - k_neighbors (int): The number of nearest neighbors to consider when generating synthetic samples.
        - random_state (int, RandomState instance or None): Determines random number generation for shuffling the data.
        - n_slices (int): The number of slices to divide the dataset into for parallel processing.

        Returns:
        None
        """
        self.k_neighbors = k_neighbors
        self.random_state = check_random_state(random_state)
        self.n_slices = n_slices

    def _create_time_slices(self, times):
        """
        Create time slices based on quantiles to ensure roughly equal distribution.

        Parameters:
        times (array-like): An array-like object containing the timestamps.

        Returns:
        array-like: An array-like object containing the time slices.
        """
        time_slices = np.quantile(times, np.linspace(0, 1, self.n_slices + 1))
        return time_slices

    def _slice_data(self, X, times):
        """
        Slice the data based on the given time points.

        Parameters:
        - X: The input data.
        - times: The time points used for slicing.

        Returns:
        - slice_dict: A dictionary where the keys represent the slice indices and the values are the indices of the data points belonging to each slice.
        """
        time_slices = self._create_time_slices(times)
        indices = np.digitize(times, time_slices) - 1
        slice_dict = {i: np.where(indices == i)[0] for i in range(self.n_slices)}
        return slice_dict

    def _make_samples(self, X, nn_data, nn_num, n_samples, step_size=1.0):
        """
        Generate synthetic samples using tSMOTE algorithm.

        Parameters:
        - X (numpy.ndarray): The original samples.
        - nn_data (numpy.ndarray): The nearest neighbors data.
        - nn_num (numpy.ndarray): The nearest neighbors indices.
        - n_samples (int): The number of synthetic samples to generate.
        - step_size (float): The step size for generating synthetic samples.

        Returns:
        - X_new (numpy.ndarray): The generated synthetic samples.
        """
        samples_indices = self.random_state.randint(0, len(nn_num), n_samples)
        steps = step_size * self.random_state.uniform(size=n_samples)
        rows = samples_indices // nn_num.shape[1]
        cols = samples_indices % nn_num.shape[1]

        X_new = np.zeros((n_samples, X.shape[1]))
        for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
            X_new[i] = X[row] + step * (nn_data[nn_num[row, col]] - X[row])
        return X_new

    def fit_resample(self, X, y, times):
        """
        Fit the tSMOTE algorithm to the input data and resample the dataset.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target labels.
            times (int): The number of times to apply tSMOTE.

        Returns:
            X_resampled (array-like): The resampled features.
            y_resampled (array-like): The resampled target labels.
        """
        slice_dict = self._slice_data(X, times)
        X_resampled, y_resampled = X.copy(), y.copy()
        
        for slice_indices in slice_dict.values():
            if len(slice_indices) < 2:
                continue  # Skip slices with less than two samples for SMOTE
            
            X_slice, y_slice = X[slice_indices], y[slice_indices]
            nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(X_slice)
            nns = nn.kneighbors(X_slice, return_distance=False)[:, 1:]
            
            n_samples = len(slice_indices)
            X_new = self._make_samples(X_slice, X_slice, nns, n_samples, step_size=0.5)
            X_resampled = np.vstack([X_resampled, X_new])
            y_resampled = np.concatenate([y_resampled, y_slice])
        
        return X_resampled, y_resampled

# Usage example:
# Assuming 'times' is an array of time stamps corresponding to each sample in X
# smote = tSMOTE(random_state=42, n_slices=5)
# X_resampled, y_resampled = smote.fit_resample(X, y, times)
