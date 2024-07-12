import numpy as np
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class tSMOTE:
    def __init__(self, k_neighbors=5, random_state=None, n_slices=10, sampling_strategy='auto'):
        """
        Initialize the tSMOTE object.

        Parameters:
        - k_neighbors (int): The number of nearest neighbors to consider when generating synthetic samples.
        - random_state (int, RandomState instance or None): Determines random number generation for shuffling the data.
        - n_slices (int): The number of slices to divide the dataset into for parallel processing.
        - sampling_strategy (str, dict, or float): Determines the sampling strategy to use for resampling. Options are:
            'minority': Resample only the minority class;
            'not minority': Resample all classes but the minority class;
            'not majority': Resample all classes but the majority class;
            'all': Resample all classes;
            When dict, the keys correspond to the targeted classes and the values to the desired number of samples for each class.
            When float, it corresponds to the ratio of the number of samples in the minority class over the number of samples in the majority class after resampling.

        Returns:
        None
        """
        self.k_neighbors = k_neighbors
        self.random_state = check_random_state(random_state)
        self.n_slices = n_slices
        self.sampling_strategy = sampling_strategy

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

    def check_neighbors_object(self, param_name, value, additional_neighbor=0):
        """
        Checks if the provided value is a NearestNeighbors instance or creates one if not.

        Parameters:
        - param_name (str): The name of the parameter being checked.
        - value: The actual value provided for the parameter.
        - additional_neighbor (int): Additional number of neighbors to configure in the NearestNeighbors instance.

        Returns:
        - NearestNeighbors: A configured NearestNeighbors instance.
        """
        if isinstance(value, NearestNeighbors):
            nn = value
        elif isinstance(value, int):
            nn = NearestNeighbors(n_neighbors=value + additional_neighbor)
        else:
            raise ValueError(f"Invalid type for {param_name}. Expected NearestNeighbors instance or int, got {type(value)} instead.")
        
        return nn

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
            if len(slice_indices) < self.k_neighbors + 1:
                continue  # Not enough samples in the slice to apply SMOTE

            X_slice, y_slice = X[slice_indices], y[slice_indices]
            nn = self.check_neighbors_object('n_neighbors', self.k_neighbors, additional_neighbor=1)
            nn.fit(X_slice)
            nns = nn.kneighbors(X_slice, return_distance=False)[:, 1:]

            for class_sample in np.unique(y_slice):
                class_mask = y_slice == class_sample
                class_indices = slice_indices[class_mask]
                if len(class_indices) < self.k_neighbors + 1:
                    continue

                # Correct indexing within the slice
                local_indices = np.arange(len(y_slice))[class_mask]
                nns_class = nns[local_indices]

                # Define the number of samples to generate
                n_samples = (max(np.bincount(y)) - np.bincount(y)[class_sample]) // 2  # Example calculation
                X_new = self._make_samples(X_slice[local_indices], X_slice, nns_class, n_samples, step_size=0.5)
                X_resampled = np.vstack([X_resampled, X_new])
                y_resampled = np.concatenate([y_resampled, [class_sample] * len(X_new)])

        return X_resampled, y_resampled

    def _define_sampling_strategy(self, class_counts):
        """
        Define the sampling strategy based on the given class counts.

        Parameters:
            class_counts (dict): A dictionary containing the counts of each class.

        Returns:
            dict: A dictionary representing the sampling strategy.

        Raises:
            None

        """
        if isinstance(self.sampling_strategy, dict):
            return self.sampling_strategy
        elif isinstance(self.sampling_strategy, float):
            # Assuming minority class is the one with the fewest samples
            min_class = min(class_counts, key=class_counts.get)
            max_count = max(class_counts.values())
            target_num = int(self.sampling_strategy * max_count)
            return {min_class: target_num - class_counts[min_class]}
        elif self.sampling_strategy == 'minority':
            min_class = min(class_counts, key=class_counts.get)
            max_count = max(class_counts.values())
            return {min_class: max_count - class_counts[min_class]}
        elif self.sampling_strategy == 'not minority':
            min_class = min(class_counts, key=class_counts.get)
            return {cls: max(class_counts.values()) - count for cls, count in class_counts.items() if cls != min_class}
        elif self.sampling_strategy == 'not majority':
            max_class = max(class_counts, key=class_counts.get)
            return {cls: class_counts[max_class] - count for cls, count in class_counts.items() if cls != max_class}
        elif self.sampling_strategy == 'all':
            max_count = max(class_counts.values())
            return {cls: max_count - count for cls, count in class_counts.items()}
        else:
            return {}

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, n_clusters_per_class=1, n_classes=2, random_state=42)

# Simulate time data
times = np.linspace(0, 1, num=1000) + np.random.normal(0, 0.1, 1000)  # Randomly perturbed linear times

# Initialize tSMOTE
t_smote = tSMOTE(random_state=42, n_slices=10, sampling_strategy=0.1)

# Resample the dataset
X_resampled, y_resampled = t_smote.fit_resample(X, y, times)

# You can then split this data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Here you could fit a model and then print classification report or other metrics
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# print(classification_report(y_test, predictions))

print("Original dataset size:", X.shape[0])
print("Resampled dataset size:", X_resampled.shape[0])
