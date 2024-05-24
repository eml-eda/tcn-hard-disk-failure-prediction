import torch.nn.functional as F
from torch import nn
import torch.nn as nn
# from torch.nn.utils import weight_norm
from torch.autograd import Variable
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, precision_score
from sklearn.utils import shuffle
import math
from collections import deque

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

## called inside Classification
def init_net(lr, history_signal, num_inputs):
    """
    Initializes the neural network model and optimizer.

    Args:
        lr (float): The learning rate for the optimizer.
        history_signal (int): The number of historical signals used as input to the model.
        num_inputs (int): The number of input features.

    Returns:
        tuple: A tuple containing the initialized neural network model and optimizer.
    """
    net = TCN_Network(history_signal, num_inputs)
    # Return a new optimizer object for the given model parameters
    optimizer = getattr(optim, 'Adam')(net.parameters(), lr=lr)
    if torch.cuda.is_available():
        print('Moving model to cuda')
        net.cuda()
    else:
        print('Model to cpu')
    return net, optimizer

# reported metrics for test dataset
def report_metrics(Y_test_real, prediction, metric):
    """
    Calculate and print various evaluation metrics based on the predicted and actual values.

    Parameters:
    - Y_test_real (array-like): The actual values of the target variable.
    - prediction (array-like): The predicted values of the target variable.
    - metric (list): A list of metrics to calculate and print.

    Returns:
    - float: The F1 score based on the predicted and actual values.
    """
    Y_test_real = np.asarray(Y_test_real)
    prediction = np.asarray(prediction)
    prediction_1_true = prediction[Y_test_real==1]
    prediction_0_true = prediction[Y_test_real==0]
    total_1 = len(prediction_1_true)
    total_0 = len(prediction_0_true)
    predicted_correct_1 = sum(prediction_1_true)
    metrics = {
        'RMSE': lambda: np.sqrt(mean_squared_error(Y_test_real, prediction)),  # Root Mean Squared Error
        'MAE': lambda: mean_absolute_error(Y_test_real, prediction),  # Mean Absolute Error
        'FDR': lambda: (sum(prediction_1_true) / total_1 * 100),  # False Discovery Rate
        'FAR': lambda: (sum(prediction_0_true) / total_0 * 100),  # False Alarm Rate
        'F1': lambda: f1_score(Y_test_real, prediction),  # F1 Score
        'recall': lambda: recall_score(Y_test_real, prediction),  # Recall (sensitivity)
        'precision': lambda: precision_score(Y_test_real, prediction),  # Precision (positive predictive value)
    }

    for m in metric:
        if m in metrics:
            print(f'SCORE {m}: %.3f' % metrics[m]())
    return f1_score(Y_test_real, prediction)

def train(ep, Xtrain, ytrain, batchsize, optimizer, model, Xtest, ytest):
    """
    Trains the model using the given training data and parameters.
    Args:
        ep (int): The current epoch number.
        Xtrain (numpy.ndarray): The input training data.
        ytrain (numpy.ndarray): The target training data.
        batchsize (int): The batch size for training.
        optimizer: The optimizer used for training.
        model: The model to be trained.
        Xtest (numpy.ndarray): The input test data.
        ytest (numpy.ndarray): The target test data.

    Returns:
        numpy.ndarray: The F1 scores calculated using the training data.

    """
    train_loss = 0
    # Randomize the order of the elements in the training set to ensure that the training process is not influenced by the order of the data
    Xtrain, ytrain = shuffle(Xtrain, ytrain)
    model.train()
    samples, features, dim_window = Xtrain.shape
    nbatches = Xtrain.shape[0] // batchsize
    correct = 0
    # we weights the different classes. We both use an unbalance management and the weighting of the classes
    weights = [1.7, 0.3]
    # we use the GPU to train
    class_weights = torch.FloatTensor(weights).cuda()
    # we use the CrossEntropyLoss as loss function
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    predictions = np.ndarray(Xtrain.shape[0])
    #criterion = torch.nn.CrossEntropyLoss()
    
    for batch_idx in np.arange(nbatches + 1):
        data = Xtrain[(batch_idx * batchsize):((batch_idx + 1) * batchsize), :, :]
        target = ytrain[(batch_idx * batchsize):((batch_idx + 1) * batchsize)]
        
        if torch.cuda.is_available():
            # Convert the data and target to tensors and move them to the GPU
            data, target = torch.Tensor(data).cuda(), torch.Tensor(target).cuda()
        else:
            # Convert the data and target to tensors
            data, target = torch.Tensor(data), torch.Tensor(target)

        # Wrap the data and target in a Variable to allow automatic differentiation  
        data, target = Variable(data), Variable(target)
        # Zero the gradients since PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        # Get the output predictions from the model
        output = model(data)
        # Calculate the loss between the predictions and the target
        loss = criterion(output, target.long())
        # Perform backpropagation to calculate the gradients of the loss with respect to the model parameters
        loss.backward()
        # Update the model parameters using the gradients and the optimizer
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        predictions[(batch_idx * batchsize):((batch_idx + 1) * batchsize)] = pred.cpu().numpy()[:, 0]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        train_loss += loss
        
        if batch_idx > 0 and batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {} \r'.format(
                ep, batch_idx * batchsize, samples, (100. * batch_idx * batchsize) / samples,
                train_loss.item() / (10 * batchsize), correct / ((batch_idx + 1) * batchsize)), end="\r")
            train_loss = 0
    
    print('Training completed')
    F1 = report_metrics(ytrain, predictions, ['FDR', 'FAR', 'F1', 'recall', 'precision'])
    # at each epoch, we test the network to print the accuracy
    test(Xtest, ytest, model)
    #test(Xtrain,ytrain, model)
    return F1
    

def test(Xtest, ytest, model):
    """
    Evaluate the model on the test dataset and calculate various metrics.

    Args:
        Xtest (numpy.ndarray): Input test data.
        ytest (numpy.ndarray): Target test data.
        model: The trained model to be evaluated.

    Returns:
        numpy.ndarray: Predictions made by the model.

    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0  # Initialize the total test loss to 0
    correct = 0  # Initialize the number of correct predictions to 0
    batchsize = 30000  # Define the number of samples in each batch
    nbatches = Xtest.shape[0] // batchsize  # Calculate the number of batches
    predictions = np.ndarray(Xtest.shape[0])  # Initialize an array to store the model's predictions
    criterion = torch.nn.CrossEntropyLoss()  # Define the loss function

    # Disable gradient calculations (since we are in test mode)
    with torch.no_grad():
        for batch_idx in np.arange(nbatches + 1):
            # Extract the data and target for this batch
            data, target = Variable(torch.Tensor(Xtest[(batch_idx * batchsize):((batch_idx + 1) * batchsize), :, :]),
                                   volatile=True), Variable(torch.Tensor(ytest[(batch_idx * batchsize):((batch_idx + 1) * batchsize)]))
            # If CUDA is available, move the data and target to the GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # Calculate the batch loss
            test_loss = criterion(output, target.long()).item()
            # Get the predicted class from the maximum class score
            pred = output.data.max(1, keepdim=True)[1]
            # Compare predictions to true label
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # Store the predictions for this batch
            predictions[(batch_idx * batchsize):((batch_idx + 1) * batchsize)] = pred.cpu().numpy()[:, 0]

    # Calculate the average loss over all of the batches
    test_loss /= Xtest.shape[0]
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, Xtest.shape[0], 100. * correct / Xtest.shape[0]))
    report_metrics(ytest, predictions, ['FDR', 'FAR', 'F1', 'recall', 'precision'])
    return predictions

def net_train_validate_TCN(net, optimizer, Xtrain, ytrain, Xtest, ytest, epochs, batch_size, lr):
    """
    Train and validate a neural network model using TCN architecture.
    --- Step 1.7.2: Perform Classification using TCN. Subflowchart: TCN Subflowchart.
    Args:
        net (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        Xtrain (numpy.ndarray): The training input data.
        ytrain (numpy.ndarray): The training target data.
        Xtest (numpy.ndarray): The validation input data.
        ytest (numpy.ndarray): The validation target data.
        epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size used for training.
        lr (float): The learning rate for the optimizer.

    Returns:
        None
    """
    ytest = ytest.values
    # Use a deque to store the last 5 F1 scores to check for convergence
    F1_list = deque(maxlen=5)

    for epoch in range(1, epochs):
        # the train include also the test inside
        F1 = train(epoch, Xtrain, ytrain, batch_size, optimizer, net, Xtest, ytest)
        F1_list.append(F1)

        if len(F1_list) == 5 and len(set(F1_list)) == 1:
            print("Exited because last 5 epochs has constant F1")
            break

        if epoch % 20 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    print('Training completed')


def train_LSTM(ep, train_loader, optimizer, model, Xtrain_examples):
    """
    Trains the LSTM model using the given training data.

    Args:
        ep (int): The current epoch number.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        model (torch.nn.Module): The LSTM model.
        Xtrain_examples (int): The total number of training examples.

    Returns:
        dict: A dictionary containing the F1 scores for different metrics.
    """
    train_loss = 0  # Initialize the total training loss to 0
    model.train()  # Set the model to training mode
    correct = 0  # Initialize the number of correct predictions to 0
    weights = [1.9, 0.1]  # Define class weights for the loss function
    class_weights = torch.FloatTensor(weights).cuda()  # Convert class weights to a CUDA tensor
    predictions = np.ndarray(Xtrain_examples)  # Store the model's predictions
    ytrain = np.ndarray(Xtrain_examples)  # Store the true labels
    #criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion = torch.nn.CrossEntropyLoss()

    for i, data in enumerate(train_loader):
        sequences, labels = data  # Input sequences and their corresponding labels
        batchsize = sequences.shape[1]  # Number of sequences in each batch
        sequences = sequences.cuda()  # Move sequences to GPU
        labels = labels.cuda()  # Move labels to GPU
        optimizer.zero_grad()  # Reset gradients from previous iteration
        output = model(sequences)  # Forward pass through the model
        l = criterion(output, labels)  # Calculate loss between model output and true labels
        l.backward()  # Backward pass to calculate gradients
        optimizer.step()  # Update model parameters
        pred = output.data.max(1, keepdim=True)[1]  # Get the predicted labels
        # Store the predicted labels for this batch in the predictions array
        predictions[(i * batchsize):((i + 1) * batchsize)] = pred.cpu().numpy()[:, 0]
        # Store the true labels for this batch in the ytrain array
        ytrain[(i * batchsize):((i + 1) * batchsize)] = labels.cpu().numpy()
        # Calculate the number of correct predictions
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        train_loss += l.item()  # Add the loss for this batch to the total training loss
        if i > 0 and i % 10 == 0:  # Every 10 iterations, print the average loss and accuracy for the last 10 batches
            print(f'Train Epoch: {ep} [{i * batchsize}/{Xtrain_examples} ({(100. * i * batchsize) / Xtrain_examples:.0f}%)] Loss: {train_loss / (10 * batchsize):.6f} Accuracy: {float(correct) / ((i + 1) * batchsize):.4f}', end="\r")
            # train_loss = 0 # FIXME: We do not need to set the loss to 0 here

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = correct / len(train_loader.dataset)
    print('Train Epoch: {} Avg Loss: {:.6f} Avg Accuracy: {:.6f}'.format(ep, avg_train_loss, avg_train_acc))

    ytrain = ytrain[:((i + 1) * batchsize)]
    predictions = predictions[:((i + 1) * batchsize)]
    F1 = report_metrics(ytrain, predictions, ['FDR', 'FAR', 'F1', 'recall', 'precision'])
    return F1

def test_LSTM(model, test_loader, Xtest_examples):
    """
    Test the LSTM model on the test dataset.

    Args:
        model (torch.nn.Module): The LSTM model to be tested.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        Xtest_examples (int): The number of examples in the test dataset.

    Returns:
        None

    """
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        test_preds = torch.as_tensor([])
        test_labels = torch.as_tensor([], dtype=torch.long)
        for i, test_data in enumerate(test_loader):
            sequences, labels = test_data
            sequences = sequences.cuda()
            labels = labels.cuda()
            preds = model(sequences)
            pred = preds.data.max(1, keepdim=True)[1]
            l = criterion(preds, labels)
            test_loss += l.item()
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
            test_preds = torch.cat((test_preds, preds.cpu()))
            test_labels = torch.cat((test_labels, labels.cpu()))
            _, pred_labels = torch.max(test_preds, 1)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, Xtest_examples,
                100. * correct / Xtest_examples))
    report_metrics(test_labels, pred_labels, ['FDR', 'FAR', 'F1', 'recall', 'precision'])

def net_train_validate_LSTM(net, optimizer, train_loader, test_loader, epochs, Xtest_examples, Xtrain_examples, lr):
    """
    Train and validate the LSTM network.

    Args:
        net (torch.nn.Module): The LSTM network model.
        optimizer (torch.optim.Optimizer): The optimizer used for training the network.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        test_loader (torch.utils.data.DataLoader): The data loader for the testing dataset.
        epochs (int): The number of epochs to train the network.
        Xtest_examples (list): List of examples from the testing dataset.
        Xtrain_examples (list): List of examples from the training dataset.
        lr (float): The learning rate for the optimizer.

    Returns:
        None
    """
    # Training Loop
    F1_list = np.ndarray(5)
    i = 0
    # identical to net_train_validate but train and test are separated and train does not include test
    for epoch in range(1, epochs):
        F1 = train_LSTM(epoch, train_loader, optimizer, net, Xtrain_examples)
        test_LSTM(net, test_loader, Xtest_examples)
        F1_list[i] = F1
        i += 1
        if i == 5:
            i = 0
        if F1_list[0] != 0 and (max(F1_list) - min(F1_list)) == 0:
            print("Exited because last 5 epochs has constant F1")
        if epoch % 20 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    print('T')