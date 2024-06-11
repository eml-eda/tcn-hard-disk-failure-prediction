from torch import nn
import torch
import torch.nn.functional as F
import math
import numpy as np

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

# Function to collate batch for inference
def FPLSTM_collate(batch):
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
    return (x_batch, y_batch)
