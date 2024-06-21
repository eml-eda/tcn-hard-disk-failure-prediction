import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, log_loss
from sklearn.utils import shuffle
import math
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR
from torch.autograd import grad
from torch.utils.data import DataLoader
from tqdm import tqdm


def report_metrics(Y_test_real, prediction, metric, writer, iteration):
    """
    Calculate and print various evaluation metrics based on the predicted and actual values.
    
    Parameters:
    - Y_test_real (array-like): The actual values of the target variable.
    - prediction (array-like): The predicted values of the target variable.
    - metric (list): A list of metrics to calculate and print.
    - writer (SummaryWriter): The TensorBoard writer.
    - iteration (int): The current iteration.

    Returns:
    - float: The F1 score based on the predicted and actual values.
    """
    Y_test_real = np.asarray(Y_test_real)
    prediction = np.asarray(prediction)
    tp = np.sum((prediction == 1) & (Y_test_real == 1))
    fp = np.sum((prediction == 1) & (Y_test_real == 0))
    tn = np.sum((prediction == 0) & (Y_test_real == 0))
    fn = np.sum((prediction == 0) & (Y_test_real == 1))
    
    metrics = {
        'RMSE': lambda: np.sqrt(mean_squared_error(Y_test_real, prediction)),
        'MAE': lambda: mean_absolute_error(Y_test_real, prediction),
        'FDR': lambda: (fp / (fp + tp)) if (fp + tp) > 0 else 0,  # False Discovery Rate
        'FAR': lambda: (fp / (tn + fp)) if (tn + fp) > 0 else 0,  # False Alarm Rate
        'F1': lambda: f1_score(Y_test_real, prediction), # F1 Score
        'recall': lambda: recall_score(Y_test_real, prediction), # Recall (sensitivity)
        'precision': lambda: precision_score(Y_test_real, prediction), # Precision (positive predictive value)
        'ROC AUC': lambda: roc_auc_score(Y_test_real, prediction) # ROC AUC
    }
    for m in metric:
        if m in metrics:
            score = metrics[m]()
            logger.info(f'SCORE {m}: {score:.3f}')
            writer.add_scalar(f'SCORE {m}', score, iteration)
    return f1_score(Y_test_real, prediction)

# these 2 functions are used to rightly convert the dataset for LSTM prediction
class FPLSTMDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset class for the FPLSTM model.

    Args:
        x (numpy.ndarray): Input data of shape (num_samples, num_timesteps, num_features).
        y (numpy.ndarray): Target labels of shape (num_samples,).

    Attributes:
        x_tensors (dict): Dictionary containing input data tensors, with keys as indices and values as torch.Tensor objects.
        y_tensors (dict): Dictionary containing target label tensors, with keys as indices and values as torch.Tensor objects.
    """

    def __init__(self, x, y):
        # swap axes to have timesteps before features
        self.x_tensors = {i : torch.as_tensor(np.swapaxes(x[i,:,:], 0, 1),
            dtype=torch.float32) for i in range(x.shape[0])}
        self.y_tensors = {i : torch.as_tensor(y[i], dtype=torch.int64)
                for i in range(y.shape[0])}

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.x_tensors.keys())

    def __getitem__(self, idx):
        """
        Returns the data and label at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the input data tensor and the target label tensor.
        """
        return (self.x_tensors[idx], self.y_tensors[idx])

# fault prediction LSTM: this network is used as a reference in the paper
class FPLSTM(nn.Module):

    def __init__(self, lstm_size, fc1_size, input_size, n_classes, dropout_prob):
        """
        Initialize the FPLSTM class.

        Args:
            lstm_size (int): The size of the LSTM layer.
            fc1_size (int): The size of the first fully connected layer.
            input_size (int): The size of the input.
            n_classes (int): The number of output classes.
            dropout_prob (float): The probability of dropout.

        Returns:
            None
        """
        super(FPLSTM, self).__init__()
        # The model layers include:
        # - LSTM: Processes the input sequence with a specified number of features in the hidden state.
        # - Dropout1: Applies dropout after the LSTM to reduce overfitting.
        # - FC1: A fully connected layer that maps the LSTM output to a higher or lower dimensional space.
        # - Dropout2: Applies dropout after the first fully connected layer.
        # - FC2: The final fully connected layer that outputs the predictions for the given number of classes.
        self.lstm_size = lstm_size
        self.lstm = nn.LSTM(input_size, lstm_size)
        self.do1 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(lstm_size, fc1_size)
        self.do2 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(fc1_size, n_classes)

    def forward(self, x_batch):
        """
        Forward pass of the network.

        Args:
            x_batch (torch.Tensor): Input batch of data.

        Returns:
            torch.Tensor: Output of the network.
        """
        _, last_lstm_out = self.lstm(x_batch)
        (h_last, c_last) = last_lstm_out
        # reshape to (batch_size, hidden_size)
        h_last = h_last[-1]
        do1_out = self.do1(h_last)
        fc1_out = F.relu(self.fc1(do1_out))
        do2_out = self.do2(fc1_out)
        # fc2_out = F.log_softmax(self.fc2(do2_out), dim=1)
        fc2_out = self.fc2(do2_out)
        return fc2_out

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the MLP class.

        Args:
            input_dim (int): The input dimension of the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
        """
        super(MLP, self).__init__()
        # The model layers include:
        # - Linear1: A fully connected layer that maps the input to the hidden dimension.
        # - ReLU: Applies the ReLU activation function to the output of the first fully connected layer.
        # - Linear2: A fully connected layer that maps the hidden dimension to the output dimension.
        # - ReLU: Applies the ReLU activation function to the output of the second fully connected layer.
        # - Linear3: A fully connected layer that maps the output dimension to the number of classes.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Define the layers
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, 2)

    def forward(self, input):
        """
        Performs the forward pass of the network.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        flattened_input = input.view(input.shape[0], -1).float()  # Ensure data is flattened
        hidden_layer1_output = F.relu(self.lin1(flattened_input))
        hidden_layer2_output = F.relu(self.lin2(hidden_layer1_output))
        final_output = self.lin3(hidden_layer2_output)
        return final_output

class NNet(nn.Module):
    def __init__(self, input_size, hidden_dim=4, num_layers=1, dropout=0.1):
        """
        Initializes the Networks_pytorch class.

        Args:
            input_size (int): The size of the input.
            hidden_dim (int, optional): The number of features in the hidden state. Defaults to 4.
            num_layers (int, optional): Number of recurrent layers. Defaults to 1.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout,
                           batch_first=True)
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, input):
        """
        Performs the forward pass of the network.

        Args:
            input: The input tensor.

        Returns:
            out: The output tensor.
        """
        _, (h_n, _) = self.rnn(input)
        repr_ = h_n[-1]
        out = self.linear(repr_)
        return out

class DenseNet(nn.Module):
    def __init__(self, input_size, hidden_size=8):
        """
        Initializes a Networks_pytorch object.

        Args:
            input_size (int): The size of the input layer.
            hidden_size (int or tuple): The size of the hidden layer(s). If a tuple is provided, it should contain two integers representing the sizes of the two hidden layers. Defaults to 8.

        Returns:
            None
        """
        hs1, hs2 = hidden_size
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hs1), nn.Tanh(),
            nn.Linear(hs1, hs2), nn.Tanh(),
            nn.Linear(hs2, 2)
        )

    def forward(self, input):
        """
        Performs a forward pass through the network.

        Args:
            input: The input tensor.

        Returns:
            The output tensor after passing through the network.
        """
        out = self.layers(input)
        return out

## this is the network used in the paper. It is a 1D conv with dilation
class TCN_Network(nn.Module):
    
    def __init__(self, history_signal, num_inputs):
        """
        Initializes the TCN_Network class.

        Args:
            history_signal (int): The length of the input signal history.
            num_inputs (int): The number of input features.

        """
        super(TCN_Network, self).__init__()

        # Dilated Convolution Block 0: 
        # - 1D Convolutional layer (Conv1d) with num_inputs input channels, 32 output channels, kernel size of 3, dilation of 2, and padding of 2.
        # - Batch normalization (BatchNorm1d) for 32 features.
        # - ReLU activation function.
        # - Second 1D Convolutional layer (Conv1d) with 32 input channels, 64 output channels, kernel size of 3, dilation of 2, and padding of 2.
        # - 1D Average pooling layer (AvgPool1d) with kernel size of 3, stride of 2, and padding of 1.
        # - Batch normalization (BatchNorm1d) for 64 features.
        # - ReLU activation function.
        self.b0_tcn0 = nn.Conv1d(num_inputs, 32, 3, dilation=2, padding=2)
        self.b0_tcn0_BN = nn.BatchNorm1d(32)
        self.b0_tcn0_ReLU = nn.ReLU()
        self.b0_tcn1 = nn.Conv1d(32, 64, 3, dilation=2, padding=2)
        self.b0_conv_pool = torch.nn.AvgPool1d(3, stride=2, padding=1)
        self.b0_tcn1_BN = nn.BatchNorm1d(64)
        self.b0_tcn1_ReLU = nn.ReLU()

        # Dilated Convolution Block 1:
        # - 1D Convolutional layer (Conv1d) with 64 input channels, 64 output channels, kernel size of 3, dilation of 2, and padding of 2.
        # - Batch normalization (BatchNorm1d) for 64 features.
        # - ReLU activation function.
        # - Second 1D Convolutional layer (Conv1d) with 64 input channels, 128 output channels, kernel size of 3, dilation of 2, and padding of 2.
        # - 1D Average pooling layer (AvgPool1d) with kernel size of 3, stride of 2, and padding of 1.
        # - Batch normalization (BatchNorm1d) for 128 features.
        # - ReLU activation function.
        self.b1_tcn0 = nn.Conv1d(64, 64, 3, dilation=2, padding=2)
        self.b1_tcn0_BN = nn.BatchNorm1d(64)
        self.b1_tcn0_ReLU = nn.ReLU()
        self.b1_tcn1 = nn.Conv1d(64, 128, 3, dilation=2, padding=2)
        self.b1_conv_pool = torch.nn.AvgPool1d(3, stride=2, padding=1)
        self.b1_tcn1_BN = nn.BatchNorm1d(128)
        self.b1_tcn1_ReLU = nn.ReLU()

        # Dilated Convolution Block 2:
        # - 1D Convolutional layer (Conv1d) with 128 input channels, 128 output channels, kernel size of 3, dilation of 4, and padding of 4.
        # - Batch normalization (BatchNorm1d) for 128 features.
        # - ReLU activation function.
        # - Repeat 1D Convolutional layer with the same specifications as the first.
        # - 1D Average pooling layer (AvgPool1d) with kernel size of 3, stride of 2, and padding of 1.
        # - Batch normalization (BatchNorm1d) for 128 features from the second convolutional layer.
        # - ReLU activation function.
        self.b2_tcn0 = nn.Conv1d(128, 128, 3, dilation=4, padding=4)
        self.b2_tcn0_BN = nn.BatchNorm1d(128)
        self.b2_tcn0_ReLU = nn.ReLU()
        self.b2_tcn1 = nn.Conv1d(128, 128, 3, dilation=4, padding=4)
        self.b2_conv_pool = torch.nn.AvgPool1d(3, stride=2, padding=1)
        self.b2_tcn1_BN = nn.BatchNorm1d(128)
        self.b2_tcn1_ReLU = nn.ReLU()

        # Fully Connected Layer 0:
        # - FC0: Linear transformation from dynamically calculated dimension (based on signal history and pooling) to 256 units. Calculated as the ceiling of three halvings of history_signal multiplied by 128.
        # - Batch normalization (BatchNorm1d) for 256 features.
        # - ReLU activation function.
        # - Dropout applied at 50% rate to reduce overfitting.

        dim_fc = int(math.ceil(math.ceil(math.ceil(history_signal/2)/2)/2)*128)
        self.FC0 = nn.Linear(dim_fc, 256) # 592 in the Excel, 768 ours with pooling
        self.FC0_BN = nn.BatchNorm1d(256)
        self.FC0_ReLU = nn.ReLU()
        self.FC0_dropout = nn.Dropout(0.5)

        # Fully Connected Layer 1:
        # - FC1: Linear transformation from 256 to 64 units.
        # - Batch normalization (BatchNorm1d) for 64 features.
        # - ReLU activation function.
        # - Dropout applied at 50% rate.
        self.FC1 = nn.Linear(256, 64)
        self.FC1_BN = nn.BatchNorm1d(64)
        self.FC1_ReLU = nn.ReLU()
        self.FC1_dropout = nn.Dropout(0.5)

        # Final Linear transformation from 64 units to 2 output units for binary classification.
        self.GwayFC = nn.Linear(64, 2)

    def forward(self, x): # computation --> Pool --> BN --> activ --> dropout
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.b0_tcn0_ReLU(self.b0_tcn0_BN(self.b0_tcn0(x)))
        x = self.b0_tcn1_ReLU(self.b0_tcn1_BN(self.b0_conv_pool(self.b0_tcn1(x))))

        x = self.b1_tcn0_ReLU(self.b1_tcn0_BN(self.b1_tcn0(x)))
        x = self.b1_tcn1_ReLU(self.b1_tcn1_BN(self.b1_conv_pool(self.b1_tcn1(x))))

        x = self.b2_tcn0_ReLU(self.b2_tcn0_BN(self.b2_tcn0(x)))
        x = self.b2_tcn1_ReLU(self.b2_tcn1_BN(self.b2_conv_pool(self.b2_tcn1(x))))

        x = x.flatten(1)

        x = self.FC0_dropout(self.FC0_ReLU(self.FC0_BN(self.FC0(x))))
        x = self.FC1_dropout(self.FC1_ReLU(self.FC1_BN(self.FC1(x))))
        x = self.GwayFC(x)

        return x

class TCNDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset class for the TCN model.

    Args:
        x (numpy.ndarray): Input data of shape (num_samples, num_timesteps, num_features).
        y (numpy.ndarray): Target labels of shape (num_samples,).

    Attributes:
        x_tensors (dict): Dictionary containing input data tensors, with keys as indices and values as torch.Tensor objects.
        y_tensors (dict): Dictionary containing target label tensors, with keys as indices and values as torch.Tensor objects.
    """

    def __init__(self, x, y):
        # No need to swap axes for TCN, keep the shape as (num_samples, num_timesteps, num_features)
        self.x_tensors = torch.as_tensor(x, dtype=torch.float32)
        self.y_tensors = torch.as_tensor(y, dtype=torch.int64)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.x_tensors)

    def __getitem__(self, idx):
        """
        Returns the data and label at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the input data tensor and the target label tensor.
        """
        return (self.x_tensors[idx], self.y_tensors[idx])

class UnifiedTrainer:
    def __init__(
        self,
        model,
        optimizer,
        epochs=100,
        batch_size=128,
        lr=0.001,
        reg=1,
        id_number=1,
        model_type='TCN',
        num_workers=4,
        scheduler_type='ReduceLROnPlateau',
        scheduler_factor=0.1,
        scheduler_patience=10,
        scheduler_step_size=30,
        scheduler_gamma=0.9,
        loss_function='CrossEntropy'
    ):
        """
        Initialize the UnifiedTrainer with all necessary components.

        Args:
            model (torch.nn.Module): The model to be trained and tested (LSTM, TCN, or MLP).
            optimizer (torch.optim.Optimizer): Optimizer used for training the model.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            lr (float): Learning rate for the optimizer.
            reg (float): Regularization factor.
            id_number (int): The ID number of the model.
            model_type (str): The type of model ('LSTM', 'TCN', 'MLP').
            num_workers (int): Number of workers for the DataLoader.
            scheduler_type (str): The type of scheduler ('ReduceLROnPlateau', 'ExponentialLR').
            scheduler_factor (float): The factor for the scheduler.
            scheduler_patience (int): The patience for the scheduler.
            scheduler_step_size (int): The step size for the scheduler.
            scheduler_gamma (float): The gamma for the scheduler.
            loss_function (str): The loss function to use for training.
        """
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.reg = reg
        self.id_number = id_number
        self.model_type = model_type
        self.num_workers = num_workers
        self.train_writer = SummaryWriter(f'runs/{model_type}_Training_Graph')
        if scheduler_type == 'ReduceLROnPlateau':
            # factor is the factor by which the learning rate will be reduced. new_lr = lr * factor
            # patience is the number of epochs with no improvement after which learning rate will be reduced
            self.scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)  # factor=0.1, patience=10
        elif scheduler_type == 'ExponentialLR':
            self.scheduler = ExponentialLR(optimizer, gamma=scheduler_gamma)  # gamma=0.9
        elif scheduler_type == 'StepLR':
            # step_size is the number of epochs after which the learning rate is multiplied by gamma.
            # gamma is the factor by which the learning rate is multiplied after each step_size epochs.
            self.scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=1 - scheduler_gamma)  # step_size=30, gamma=0.1
        else:
            raise ValueError(f"Invalid scheduler_type: {scheduler_type}")
        self.loss_function = loss_function
        self.test_writer = SummaryWriter(f'runs/{model_type}_Test_Graph')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Get the device
        self.test_accuracy = 0  # Initialize test_accuracy

    def FPLSTM_collate(self, batch):
        """
        Collates a batch of data for FPLSTM model.

        Args:
            batch (list): A list of tuples containing input and target tensors.

        Returns:
            tuple: A tuple containing the collated input tensor and target tensor.
        """
        xx, yy = zip(*batch)
        x_batch = torch.stack(xx).permute(1, 0, 2)
        y_batch = torch.stack(yy)
        return x_batch, y_batch

    def calculate_total_loss(self, error, reg):
        """
        Calculates the total loss for the model.

        Parameters:
        - error: The error term representing the model's prediction error.
        - reg: The regularization parameter.

        Returns:
        - total_loss: The total loss, which is a combination of the prediction error and regularization penalty.
        """
        grads = grad(error, self.model.parameters(), create_graph=True)
        penalty = sum(g.pow(2).mean() for g in grads)
        total_loss = reg * error + (1 - reg) * penalty
        return total_loss

    def train(self, train_loader, train_loader_tqdm, epoch):
        """
        Trains the LSTM model using the given training data.

        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.
            train_loader_tqdm (tqdm): The tqdm wrapper for the training data loader.
            epoch (int): The current epoch number.

        Returns:
            dict: A dictionary containing the F1 scores for different metrics.
        """
        # Set the model to training mode
        self.model.train()
        # Define class weights for the loss function, with the first class being the majority class and the second class being the minority class
        weights = [1.7, 0.3]
        # Convert class weights to a CUDA tensor
        class_weights = torch.FloatTensor(weights).to(self.device)
        # We use the CrossEntropyLoss as loss function to guide the model towards making accurate predictions on the training data.
        if self.loss_function == 'CrossEntropy':
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif self.loss_function == 'BCEWithLogits':
            criterion = torch.nn.BCEWithLogitsLoss(weight=class_weights)
        else:
            raise ValueError(f"Invalid loss function: {self.loss_function}")
        predictions = np.zeros((len(train_loader.dataset), 2))  # Store the model's predictions
        true_labels = np.zeros(len(train_loader.dataset))  # Store the true labels

        for batch_idx, data in enumerate(train_loader_tqdm):
            # Input sequences and their corresponding labels
            sequences, labels = data
            # Move sequences and labels to GPU
            sequences, labels = sequences.to(self.device), labels.to(self.device)
            # Reset gradients from previous iteration
            self.optimizer.zero_grad()
            # Disable CuDNN for the forward pass to avoid double backward issues
            with torch.backends.cudnn.flags(enabled=False):
                # Forward pass through the model
                output = self.model(sequences)
                # Apply softmax to the output
                output_softmax = F.softmax(output, dim=1)
                # Calculate loss between model output and true labels
                loss = criterion(output, labels)
                # Calculate the total loss (error + penalty)
                total_loss = self.calculate_total_loss(loss, self.reg)

            # Backward pass and parameter update
            total_loss.backward()
            self.optimizer.step()
            # Store the predicted labels for this batch in the predictions array
            predictions[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size), :] = output_softmax.cpu().detach().numpy()
            # Store the true labels for this batch in the true_labels array
            true_labels[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size)] = labels.cpu().numpy()

            if batch_idx > 0 and batch_idx % 10 == 0:
                # Calculate average loss for the last 10 batches
                avg_loss = log_loss(true_labels[:((batch_idx + 1) * self.batch_size)], predictions[:((batch_idx + 1) * self.batch_size)], labels=[0, 1])
                avg_accuracy = accuracy_score(true_labels[:((batch_idx + 1) * self.batch_size)], predictions[:((batch_idx + 1) * self.batch_size)].argmax(axis=1))
                self.lr = self.optimizer.param_groups[0]['lr']
                # Log to TensorBoard
                self.train_writer.add_scalar('Training Loss', avg_loss, epoch * len(train_loader) + batch_idx)
                self.train_writer.add_scalar('Training Accuracy', avg_accuracy, epoch * len(train_loader) + batch_idx)
                self.train_writer.add_scalar('Learning Rate', self.lr, epoch * len(train_loader) + batch_idx)

                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.4f} LR: {:.6f}'.format(
                    epoch, batch_idx * self.batch_size, len(train_loader.dataset), 
                    (100. * (batch_idx * self.batch_size) / len(train_loader.dataset)),
                    avg_loss, avg_accuracy, self.lr), end="\r")

        avg_train_loss = log_loss(true_labels[:((batch_idx + 1) * self.batch_size)], predictions[:((batch_idx + 1) * self.batch_size)], labels=[0, 1])
        avg_train_acc = accuracy_score(true_labels[:((batch_idx + 1) * self.batch_size)], predictions[:((batch_idx + 1) * self.batch_size)].argmax(axis=1))

        # Log to TensorBoard
        self.train_writer.add_scalar('Average Loss', avg_train_loss, epoch)
        self.train_writer.add_scalar('Average Accuracy', avg_train_acc, epoch)

        logger.info(
            f'Train Epoch: {epoch} '
            f'Avg Loss: {avg_train_loss:.6f} '
            f'Avg Accuracy: {int(avg_train_acc * len(train_loader.dataset))}/{len(train_loader.dataset)} '
            f'({100. * avg_train_acc:.0f}%)'
        )
        print('\n')
        return report_metrics(true_labels, predictions.argmax(axis=1), ['FDR', 'FAR', 'F1', 'recall', 'precision', 'ROC AUC'], self.train_writer, epoch)

    def test(self, test_loader, test_loader_tqdm, epoch):
        """
        Test the LSTM model on the test dataset.

        Args:
            test_loader (torch.utils.data.DataLoader): The test data loader.
            test_loader_tqdm (tqdm): The tqdm wrapper for the test data loader.
            epoch (int): The current epoch number.

        Returns:
            None
        """
        self.model.eval()
        if self.loss_function == 'CrossEntropy':
            criterion = torch.nn.CrossEntropyLoss()
        elif self.loss_function == 'BCEWithLogits':
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Invalid loss function: {self.loss_function}")
        predictions = np.zeros((len(test_loader.dataset), 2))
        true_labels = np.zeros(len(test_loader.dataset))
    
        with torch.no_grad():
            for batch_idx, test_data in enumerate(test_loader_tqdm):
                sequences, labels = test_data
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                # Forward pass through the model
                output = self.model(sequences)
                # Apply softmax to the output
                output_softmax = F.softmax(output, dim=1)
                loss = criterion(output, labels)

                # Store the predictions and true labels for this batch
                predictions[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size), :] = output_softmax.cpu().numpy()
                true_labels[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size)] = labels.cpu().numpy()

        # Calculate the average loss and accuracy over all of the batches
        avg_test_loss = log_loss(true_labels, predictions, labels=[0, 1])
        avg_test_acc = accuracy_score(true_labels, predictions.argmax(axis=1))

        # Log to TensorBoard
        self.test_writer.add_scalar('Average Loss', avg_test_loss, epoch)
        self.test_writer.add_scalar('Average Accuracy', avg_test_acc, epoch)
        logger.info(
            f'Test set: '
            f'Average loss: {avg_test_loss:.4f}, '
            f'Accuracy: {avg_test_acc:.4f}'
        )
        print('\n')
        report_metrics(true_labels, predictions.argmax(axis=1), ['FDR', 'FAR', 'F1', 'recall', 'precision', 'ROC AUC'], self.test_writer, epoch)
        self.test_accuracy = avg_test_acc  # Store test accuracy
        #return predictions.argmax(axis=1)

    def run(self, Xtrain, ytrain, Xtest, ytest):
        """
        Run the training and testing process for the model.

        Args:
            Xtrain (np.ndarray): The training input data.
            ytrain (np.ndarray): The training target data.
            Xtest (np.ndarray): The testing input data.
            ytest (np.ndarray): The testing target data.
        """
        if self.model_type == 'LSTM':
            train_loader = DataLoader(FPLSTMDataset(Xtrain, ytrain), batch_size=self.batch_size, shuffle=True, collate_fn=self.FPLSTM_collate, pin_memory=True, num_workers=self.num_workers, prefetch_factor=10, persistent_workers=True)
            test_loader = DataLoader(FPLSTMDataset(Xtest, ytest), batch_size=self.batch_size, shuffle=True, collate_fn=self.FPLSTM_collate, pin_memory=True, num_workers=self.num_workers, prefetch_factor=10, persistent_workers=True)
        else:
            train_loader = DataLoader(TCNDataset(Xtrain, ytrain), batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers, prefetch_factor=10, persistent_workers=True)
            test_loader = DataLoader(TCNDataset(Xtest, ytest), batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers, prefetch_factor=10, persistent_workers=True)

        # Wrap the DataLoader with tqdm
        train_loader_tqdm = tqdm(train_loader)
        test_loader_tqdm = tqdm(test_loader)
        F1_list = deque(maxlen=5)
        for epoch in range(1, self.epochs):
            F1 = self.train(train_loader, train_loader_tqdm, epoch)
            self.test(test_loader, test_loader_tqdm, epoch)
            F1_list.append(F1)

            if len(F1_list) == 5 and len(set(F1_list)) == 1:
                logger.info("Exited because last 5 epochs has constant F1")
                break

            # if epoch % 20 == 0:
            #     self.lr /= 10
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = self.lr
        self.train_writer.close()
        self.test_writer.close()
        logger.info('Training completed, saving the model...')

        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', self.id_number)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f'{self.model_type.lower()}_{self.id_number}_epochs_{self.epochs}_batchsize_{self.batch_size}_lr_{self.lr}_{now_str}.pth')
        torch.save(self.model.state_dict(), model_path)
        logger.info(f'Model saved as: {model_path}')
