from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer, silhouette_score
from sklearn.utils import shuffle
import torch.optim as optim
import logger
from joblib import dump
from json_param import save_best_params_to_json, load_best_params_from_json
from ray import tune
from Networks_pytorch import *
from tqdm import tqdm
from datetime import datetime


def train_and_evaluate_model(model, param_grid, classifier_name, X_train, Y_train, X_test, Y_test, id_number, metric, search_method='randomized', n_iterations=100):
    """
    Trains and evaluates a machine learning model.

    Args:
        model (object): The machine learning model to be trained and evaluated.
        param_grid (dict): The parameter grid to search over.
        classifier_name (str): The name of the classifier.
        X_train (array-like): The training data features.
        Y_train (array-like): The training data labels.
        X_test (array-like): The test data features.
        Y_test (array-like): The test data labels.
        id_number (str): The ID number for the model.
        metric (str): The metric to be used for evaluation.
        search_method (str, optional): The search method to use. Defaults to 'randomized'.
        n_iterations (int, optional): The number of iterations for training. Defaults to 100.

    Returns:
        str: The path to the saved model file.
    """
    writer = SummaryWriter(f'runs/{classifier_name}_Training_Graph')
    X_train, Y_train = shuffle(X_train, Y_train)

    # Define supervised classifiers
    supervised_classifiers = ['RandomForest', 'KNeighbors', 'DecisionTree', 'LogisticRegression', 'SVM', 'XGB', 'MLP', 'ExtraTrees', 'GradientBoosting', 'NaiveBayes', 'RGF']

    # Define scoring metrics based on the type of classifier
    if classifier_name in supervised_classifiers:
        scoring = {'accuracy': make_scorer(accuracy_score), 'f1': make_scorer(f1_score)}
    else:
        scoring = {'silhouette': make_scorer(silhouette_score)}

    if param_grid:
        if search_method == 'grid':
            # Initialize GridSearchCV
            search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring=scoring, refit='f1')
        elif search_method == 'randomized':
            # Initialize RandomizedSearchCV
            search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3, n_jobs=-1, verbose=2, scoring=scoring, refit='f1', n_iter=n_iterations)
        else:
            raise ValueError(f'Invalid search method: {search_method}')

        # Fit the search method
        search.fit(X_train, Y_train)
        best_params = search.best_params_
        best_model = search.best_estimator_
    else:
        best_model = model
        best_params = model.get_params()

    if classifier_name in supervised_classifiers:
        # Calculate cross validation score, more splits reduce bias but increase variance
        cv_scores = cross_val_score(best_model, X_train, Y_train, cv=StratifiedKFold(n_splits=5))

        logger.info(f"Cross validation scores: {cv_scores}")
        logger.info(f"Mean cross validation score: {cv_scores.mean()}")
    else:
        # For unsupervised classifiers, calculate silhouette score
        labels = best_model.fit_predict(X_train)
        silhouette_avg = silhouette_score(X_train, labels)
        logger.info(f"Silhouette score: {silhouette_avg}")

    logger.info(f"Best parameters: {best_params}")

    # Save the best parameters to a JSON file
    save_best_params_to_json(best_params, classifier_name, id_number)

    # Split the training data into multiple batches
    batch_size = len(X_train) // n_iterations
    pbar = tqdm(total=n_iterations)  # Initialize tqdm with the total number of batches

    for i in range(0, len(X_train), batch_size):
        best_model.fit(X_train[i:i+batch_size, :], Y_train[i:i+batch_size])
        pbar.update(1)  # Update the progress bar

    pbar.close()
    prediction = best_model.predict(X_test)
    Y_test_real = Y_test
    accuracy = accuracy_score(Y_test_real, prediction)
    auc = roc_auc_score(Y_test_real, prediction)
    logger.info(f'{classifier_name} Prediction Accuracy: {accuracy * 100:.4f}%, AUC: {auc:.2f}')
    report_metrics(Y_test_real, prediction, metric, writer, n_iterations)
    writer.close()

    # Save the trained model to a file
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', id_number)
    # Create the directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Format as string
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save the model
    model_path = os.path.join(model_dir, f'{classifier_name}_{id_number}_iterations_{n_iterations}_{now_str}.joblib')
    dump(best_model, model_path)
    logger.info(f'Model saved as: {model_path}')

    return model_path

def train_lstm(config, data, enable_tuning=True, incremental_learning=False, transfer_learning=False, classifier='FPLSTM', id_number=1):
    Xtrain, ytrain, Xtest, ytest = data

    # Set training parameters
    lr = config['lr']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    epochs = config['epochs']
    dropout = config['dropout']
    lstm_hidden_s = config['lstm_hidden_s']
    fc1_hidden_s = config['fc1_hidden_s']
    optimizer_type = config['optimizer_type']
    reg = config['reg']
    num_workers = config['num_workers']
    num_inputs = Xtrain.shape[1]

    net = FPLSTM(lstm_hidden_s, fc1_hidden_s, num_inputs, 2, dropout)
    
    if incremental_learning:
        net.load_state_dict(torch.load(f'{classifier.lower()}_{id_number}_epochs_{epochs}_batchsize_{batch_size}_lr_{lr}_*.pth'))
        if transfer_learning:
            for name, param in net.named_parameters():
                if 'lstm' in name or 'do1' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Moving model to {device}')
    net.to(device)

    if optimizer_type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer type. Please choose either "Adam" or "SGD".')

    trainer = UnifiedTrainer(
        model=net,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        reg=reg,
        id_number=id_number,
        model_type='LSTM',
        num_workers=num_workers
    )

    trainer.run(Xtrain, ytrain, Xtest, ytest)

    # Report the test accuracy to Ray Tune if tuning is enabled
    if enable_tuning:
        tune.report(accuracy=trainer.test_accuracy)

def train_nnet(config, data, enable_tuning=True, incremental_learning=False, transfer_learning=False, classifier='NNet', id_number=1):
    """
    Trains a neural network classifier.

    Args:
        config (dict): A dictionary containing the training parameters.
        data (tuple): A tuple containing the training and testing data.
        enable_tuning (bool, optional): Whether to report the test accuracy to Ray Tune. Defaults to True.
        incremental_learning (bool, optional): Whether to perform incremental learning. Defaults to False.
        transfer_learning (bool, optional): Whether to perform transfer learning. Defaults to False.
        classifier (str, optional): The type of classifier. Defaults to 'NNet'.
        id_number (int, optional): The ID number of the classifier. Defaults to 1.

    Returns:
        None
    """
    Xtrain, ytrain, Xtest, ytest = data

    # Set training parameters
    batch_size = config['batch_size']
    lr = config['lr']
    weight_decay = config['weight_decay']
    epochs = config['epochs']
    dropout = config['dropout']
    hidden_dim = config['hidden_dim']
    num_layers = config['num_layers']
    optimizer_type = config['optimizer_type']
    reg = config['reg']
    num_workers = config['num_workers']
    data_dim = Xtrain.shape[2]  # Number of features in the input

    logger.info(f'data dimension: {data_dim}, hidden_dim: {hidden_dim}')
    
    net = NNet(input_size=data_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    
    if incremental_learning:
        # Load the pre-trained model
        net.load_state_dict(torch.load(f'{classifier.lower()}_{id_number}_epochs_{epochs}_batchsize_{batch_size}_lr_{lr}_*.pth'))
        if transfer_learning:
            # Freeze the LSTM layers (rnn), fine-tune the rest
            for name, param in net.named_parameters():
                if 'rnn' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Moving model to {device}')
        net.to(device)

        if optimizer_type == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer type. Please choose either "Adam" or "SGD".')
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Moving model to {device}')
        net.to(device)

        if optimizer_type == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer type. Please choose either "Adam" or "SGD".')

    # Initialize the trainer
    nnet_trainer = UnifiedTrainer(
        model=net,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        reg=reg,
        id_number=id_number,
        model_type='NNet',
        num_workers=num_workers
    )
    
    # Run training and testing
    nnet_trainer.run(Xtrain, ytrain, Xtest, ytest)
    
    # Report the test accuracy to Ray Tune if tuning is enabled
    if enable_tuning:
        tune.report(accuracy=nnet_trainer.test_accuracy)

def train_tcn(config, data, enable_tuning=True, incremental_learning=False, transfer_learning=False, classifier='TCN', id_number=1):
    """
    Trains a neural network classifier.

    Args:
        config (dict): A dictionary containing the training parameters.
        data (tuple): A tuple containing the training and testing data.
        enable_tuning (bool, optional): Whether to report the test accuracy to Ray Tune. Defaults to True.
        incremental_learning (bool, optional): Whether to perform incremental learning. Defaults to False.
        transfer_learning (bool, optional): Whether to perform transfer learning. Defaults to False.
        classifier (str, optional): The type of classifier. Defaults to 'NNet'.
        id_number (int, optional): The ID number of the classifier. Defaults to 1.

    Returns:
        None
    """
    Xtrain, ytrain, Xtest, ytest = data

    # Set training parameters
    batch_size = config['batch_size']
    lr = config['lr']
    weight_decay = config['weight_decay']
    epochs = config['epochs']
    optimizer_type = config['optimizer_type']
    reg = config['reg']
    num_workers = config['num_workers']
    data_dim = Xtrain.shape[2]
    num_inputs = Xtrain.shape[1]

    logger.info(f'number of inputs: {num_inputs}, data_dim: {data_dim}')
    
    net = TCN_Network(data_dim, num_inputs)
    
    if incremental_learning:
        net.load_state_dict(torch.load(f'{classifier.lower()}_{id_number}_epochs_{epochs}_batchsize_{batch_size}_lr_{lr}_*.pth'))
        if transfer_learning:
            for name, param in net.named_parameters():
                if 'b0_' in name or 'b1_' in name or 'b2_' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Moving model to {device}')
        net.to(device)

        if optimizer_type == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer type. Please choose either "Adam" or "SGD".')
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Moving model to {device}')
        net.to(device)

        if optimizer_type == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer type. Please choose either "Adam" or "SGD".')

    # Initialize the trainer
    tcn_trainer = UnifiedTrainer(
        model=net,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        reg=reg,
        id_number=id_number,
        model_type='TCN',
        num_workers=num_workers
    )
    
    # Run training and testing
    tcn_trainer.run(Xtrain, ytrain, Xtest, ytest)
    
    # Report the test accuracy to Ray Tune if tuning is enabled
    if enable_tuning:
        tune.report(accuracy=tcn_trainer.test_accuracy)

def train_densenet(config, data, enable_tuning=True, incremental_learning=False, transfer_learning=False, classifier='DenseNet', id_number=1):
    """
    Trains a neural network classifier.

    Args:
        config (dict): A dictionary containing the training parameters.
        data (tuple): A tuple containing the training and testing data.
        enable_tuning (bool, optional): Whether to report the test accuracy to Ray Tune. Defaults to True.
        incremental_learning (bool, optional): Whether to perform incremental learning. Defaults to False.
        transfer_learning (bool, optional): Whether to perform transfer learning. Defaults to False.
        classifier (str, optional): The type of classifier. Defaults to 'NNet'.
        id_number (int, optional): The ID number of the classifier. Defaults to 1.

    Returns:
        None
    """
    Xtrain, ytrain, Xtest, ytest = data

    # Set training parameters
    batch_size = config['batch_size']
    lr = config['lr']
    weight_decay = config['weight_decay']
    epochs = config['epochs']
    hidden_size = config['hidden_size']
    optimizer_type = config['optimizer_type']
    reg = config['reg']
    num_workers = config['num_workers']
    num_inputs = Xtrain.shape[1]

    logger.info(f'number of inputs: {num_inputs}, hidden_size: {hidden_size} x {hidden_size}')
    
    net = DenseNet(input_size=num_inputs, hidden_size=hidden_size)
    
    if incremental_learning:
        # Load the pre-trained model
        net.load_state_dict(torch.load(f'{classifier.lower()}_{id_number}_epochs_{epochs}_batchsize_{batch_size}_lr_{lr}_*.pth'))
        if transfer_learning:
            for name, param in net.named_parameters():
                if 'layers.0' in name or 'layers.2' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Moving model to {device}')
        net.to(device)

        if optimizer_type == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer type. Please choose either "Adam" or "SGD".')
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Moving model to {device}')
        net.to(device)

        if optimizer_type == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer type. Please choose either "Adam" or "SGD".')

    # Initialize the trainer
    densenet_trainer = UnifiedTrainer(
        model=net,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        reg=reg,
        id_number=id_number,
        model_type='DenseNet',
        num_workers=num_workers
    )
    
    # Run training and testing
    densenet_trainer.run(Xtrain, ytrain, Xtest, ytest)
    
    # Report the test accuracy to Ray Tune if tuning is enabled
    if enable_tuning:
        tune.report(accuracy=densenet_trainer.test_accuracy)

def train_mlp(config, data, enable_tuning=True, incremental_learning=False, transfer_learning=False, classifier='MLP', id_number=1):
    """
    Trains a neural network classifier.

    Args:
        config (dict): A dictionary containing the training parameters.
        data (tuple): A tuple containing the training and testing data.
        enable_tuning (bool, optional): Whether to report the test accuracy to Ray Tune. Defaults to True.
        incremental_learning (bool, optional): Whether to perform incremental learning. Defaults to False.
        transfer_learning (bool, optional): Whether to perform transfer learning. Defaults to False.
        classifier (str, optional): The type of classifier. Defaults to 'NNet'.
        id_number (int, optional): The ID number of the classifier. Defaults to 1.

    Returns:
        None
    """
    Xtrain, ytrain, Xtest, ytest = data

    # Set training parameters
    batch_size = config['batch_size']
    lr = config['lr']
    weight_decay = config['weight_decay']
    epochs = config['epochs']
    input_dim = Xtrain.shape[1] * Xtrain.shape[2]  # Number of features in the input
    hidden_dim = config['hidden_dim']
    optimizer_type = config['optimizer_type']
    reg = config['reg']
    num_workers = config['num_workers']

    logger.info(f'number of inputs: {input_dim}, hidden_dim: {hidden_dim}')

    net = MLP(input_dim=input_dim, hidden_dim=hidden_dim)

    if incremental_learning:
        # Load the pre-trained model
        net.load_state_dict(torch.load(f'{classifier.lower()}_{id_number}_epochs_{epochs}_batchsize_{batch_size}_lr_{lr}_*.pth'))
        if transfer_learning:
            for name, param in net.named_parameters():
                if 'lin1' in name or 'lin2' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Moving model to {device}')
        net.to(device)

        if optimizer_type == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer type. Please choose either "Adam" or "SGD".')
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Moving model to {device}')
        net.to(device)

        if optimizer_type == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer type. Please choose either "Adam" or "SGD".')

    # Initialize the trainer
    mlp_trainer = UnifiedTrainer(
        model=net,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        reg=reg,
        id_number=id_number,
        model_type='MLP',
        num_workers=num_workers
    )

    # Run training and testing
    mlp_trainer.run(Xtrain, ytrain, Xtest, ytest)

    # Report the test accuracy to Ray Tune if tuning is enabled
    if enable_tuning:
        tune.report(accuracy=mlp_trainer.test_accuracy)
