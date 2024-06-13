import os
# import pandas as pd
import modin.pandas as pd
import sys
from sklearn.cluster import DBSCAN
from Dataset_manipulation import *
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
from sklearn.ensemble import RandomForestClassifier, IsolationForest, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from Networks_pytorch import *
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer, silhouette_score
import torch.optim as optim
from sklearn import svm
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from datetime import datetime
from joblib import dump
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from rgf.sklearn import RGFClassifier
import json
import logger
import ray
from ray import tune


# Define default global values
TRAINING_PARAMS = {
    'reg': 0.1,
    'batch_size': 256,
    'lr': 0.001,
    'weight_decay': 0.01,
    'epochs': 200,
    'dropout': 0.1,  # LSTM
    'lstm_hidden_s': 64,  # LSTM
    'fc1_hidden_s': 16,  # LSTM
    'hidden_dim': 128,  # MLP_Torch
    'hidden_size': 8,  # DenseNet
    'num_layers': 1,  # NNet
    'optimizer_type': 'Adam',
    'num_workers': 8
}

def save_best_params_to_json(best_params, classifier_name, id_number):
    """
    Saves the best parameters to a JSON file.

    Args:
        best_params (dict): The best parameters.
        classifier_name (str): The name of the classifier.
        id_number (str): The ID number for the model.

    Returns:
        None
    """
    # Define the directory path
    param_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', id_number)
    # Create the directory if it doesn't exist
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    # Define the file path
    file_path = os.path.join(param_dir, f'{classifier_name.lower()}_{id_number}_best_params.json')

    # Save the best parameters to a JSON file
    with open(file_path, 'w') as f:
        json.dump(best_params, f)

    logger.info(f'Best parameters saved to: {file_path}')

def load_best_params_from_json(classifier_name, id_number):
    """
    Loads the best parameters from a JSON file.

    Args:
        classifier_name (str): The name of the classifier.
        id_number (str): The ID number for the model.

    Returns:
        dict: The best parameters.
    """
    # Define the directory path
    param_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', id_number)

    # Define the file path
    file_path = os.path.join(param_dir, f'{classifier_name.lower()}_{id_number}_best_params.json')

    # Load the best parameters from a JSON file
    with open(file_path, 'r') as f:
        best_params = json.load(f)

    logger.info(f'Best parameters loaded from: {file_path}')

    return best_params

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

    if search_method == 'grid':
        # Initialize GridSearchCV
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring=scoring, refit='f1')
    elif search_method == 'randomized':
        # Initialize RandomizedSearchCV
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3, n_jobs=-1, verbose=2, scoring=scoring, refit='f1', n_iter=100)
    else:
        raise ValueError(f'Invalid search method: {search_method}')

    # Fit the search method
    search.fit(X_train, Y_train)

    if classifier_name in supervised_classifiers:
        # Calculate cross validation score, more splits reduce bias but increase variance
        cv_scores = cross_val_score(search.best_estimator_, X_train, Y_train, cv=StratifiedKFold(n_splits=5))

        logger.info(f"Cross validation scores: {cv_scores}")
        logger.info(f"Mean cross validation score: {cv_scores.mean()}")
    else:
        # For unsupervised classifiers, calculate silhouette score
        labels = search.best_estimator_.fit_predict(X_train)
        silhouette_avg = silhouette_score(X_train, labels)
        logger.info(f"Silhouette score: {silhouette_avg}")

    # Get the best parameters
    best_params = search.best_params_
    logger.info(f"Best parameters: {best_params}")

    # Save the best parameters to a JSON file
    save_best_params_to_json(best_params, classifier_name, id_number)

    # Get the best estimator
    best_model = search.best_estimator_

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
    report_metrics(Y_test_real, prediction, metric, writer, n_iterations)  # TODO:
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

def classification(X_train, Y_train, X_test, Y_test, classifier, metric, **args):
    """
    Perform classification using the specified classifier.
    --- Step 1.7: Perform Classification
    Parameters:
    - X_train (array-like): Training data features.
    - Y_train (array-like): Training data labels.
    - X_test (array-like): Test data features.
    - Y_test (array-like): Test data labels.
    - classifier (str): The classifier to use. Options: 'RandomForest', 'TCN', 'LSTM'.
    - metric (str): The metric to evaluate the classification performance.
    - **args: Additional arguments specific to each classifier.

    Returns:
    - str: The path to the saved model file.
    """
    logger.info(f'Classification using {classifier} is starting')

    n_iterations = 100
    if classifier == 'RandomForest':
        # Step 1.7.1: Perform Classification using RandomForest.
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = RandomForestClassifier(random_state=3, warm_start=True)

        # Define the parameter grid
        param_grid = {
            'n_estimators': [1000, 2000, 3000],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [10, 20, 30, 40, None],
            'criterion': ['gini', 'entropy'],
            'bootstrap': [True, False]
        }

        # If the best parameters exist, use them
        if best_params:
            model.set_params(**best_params)
            param_grid = {}

        return train_and_evaluate_model(model, param_grid, 'RandomForest', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)
    elif classifier == 'KNeighbors':
        # Step 1.7.2: Perform Classification using KNeighbors.
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = KNeighborsClassifier()

        # Define the parameter grid
        param_grid = {
            'n_neighbors': list(range(1, 31)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        # If the best parameters exist, use them
        if best_params:
            model.set_params(**best_params)
            param_grid = {}

        return train_and_evaluate_model(model, param_grid, 'KNeighbors', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)
    elif classifier == 'DecisionTree':
        # Step 1.7.3: Perform Classification using DecisionTree.
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = DecisionTreeClassifier()

        # Define the parameter grid
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': list(range(1, 31)),
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2', None]
        }

        # If the best parameters exist, use them
        if best_params:
            model.set_params(**best_params)
            param_grid = {}

        return train_and_evaluate_model(model, param_grid, 'DecisionTree', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)
    elif classifier == 'LogisticRegression':
        # Step 1.7.4: Perform Classification using LogisticRegression.
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = LogisticRegression()

        # Define the parameter grid
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': np.logspace(-4, 4, 20),
            'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 1000, 2500, 5000]
        }

        # If the best parameters exist, use them
        if best_params:
            model.set_params(**best_params)
            param_grid = {}

        return train_and_evaluate_model(model, param_grid, 'LogisticRegression', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)
    elif classifier == 'SVM':
        # Step 1.7.5: Perform Classification using SVM.
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = svm.SVC()

        # Define the parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],  
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }

        # If the best parameters exist, use them
        if best_params:
            model.set_params(**best_params)
            param_grid = {}

        return train_and_evaluate_model(model, param_grid, 'SVM', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)
    elif classifier == 'XGB':
        # Step 1.7.7: Perform Classification using XGBoost.
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = XGBClassifier()

        # Define the parameter grid
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'n_estimators': [100, 500, 1000, 1500],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0.1, 0.2, 0.3, 0.4],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'objective': ['binary:logistic']
        }

        # If the best parameters exist, use them
        if best_params:
            model.set_params(**best_params)
            param_grid = {}

        return train_and_evaluate_model(model, param_grid, 'XGB', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)
    elif classifier == 'IsolationForest':
        # Step 1.7.9: Perform Classification using IsolationForest.
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = IsolationForest()

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_samples': ['auto', 100, 200, 300, 400, 500],
            'contamination': ['auto', 0.1, 0.2, 0.3, 0.4, 0.5],
            'max_features': [1, 2, 3, 4, 5],
            'bootstrap': [True, False]
        }

        # If the best parameters exist, use them
        if best_params:
            model.set_params(**best_params)
            param_grid = {}

        return train_and_evaluate_model(model, param_grid, 'IsolationForest', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)
    elif classifier == 'ExtraTrees':
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = ExtraTreesClassifier()

        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }

        if best_params:
            model.set_params(**best_params)
            param_grid = {}

        return train_and_evaluate_model(model, param_grid, 'ExtraTreesClassifier', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)
    elif classifier == 'GradientBoosting':
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = GradientBoostingClassifier()

        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.1, 0.05, 0.01],
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10]
        }

        if best_params:
            model.set_params(**best_params)
            param_grid = {}

        return train_and_evaluate_model(model, param_grid, 'GradientBoostingClassifier', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)
    elif classifier == 'NaiveBayes':
        # Step 1.7.6: Perform Classification using Naive Bayes.
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = GaussianNB()

        # Define the parameter grid
        # Naive Bayes does not have any hyperparameters that need to be tuned
        param_grid = {}

        # If the best parameters exist, use them
        if best_params:
            model.set_params(**best_params)

        return train_and_evaluate_model(model, param_grid, 'NaiveBayes', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)
    elif classifier == 'RGF':
        # Step 1.7.7: Perform Classification using Regularized Greedy Forest (RGF).
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = RGFClassifier()

        # Define the parameter grid
        # You can define some hyperparameters here. For example:
        param_grid = {
            'max_leaf': [1000, 1200, 1500],
            'algorithm': ["RGF", "RGF_Opt", "RGF_Sib"],
            'test_interval': [100, 600, 900]
        }

        # If the best parameters exist, use them
        if best_params:
            model.set_params(**best_params)

        return train_and_evaluate_model(model, param_grid, 'RGF', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)
    elif classifier == 'TCN':
        # Step 1.7.6: Perform Classification using TCN. Subflowchart: TCN Subflowchart. Train and validate the network using TCN
        data = (Xtrain, ytrain, Xtest, ytest)

        if enable_tuning:
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": tune.loguniform(1e-4, 1e-1, TRAINING_PARAMS['lr']),
                "weight_decay": tune.loguniform(1e-5, 1e-2, TRAINING_PARAMS['weight_decay']),
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
            }

            scheduler = ASHAScheduler(
                metric="accuracy",
                mode="max",
                max_t=30,
                grace_period=1,
                reduction_factor=2
            )

            reporter = CLIReporter(
                metric_columns=["accuracy", "training_iteration"]
            )

            result = tune.run(
                tune.with_parameters(train_tcn, data=data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning']),
                resources_per_trial={"cpu": 2, "gpu": 1},
                config=config,
                num_samples=10,
                scheduler=scheduler,
                progress_reporter=reporter
            )

            best_trial = result.get_best_trial("accuracy", "max", "last")
            best_params = best_trial.config
            print("Best trial config: {}".format(best_params))
            print("Best trial final validation accuracy: {}".format(
                best_trial.last_result["accuracy"]))

            # Save the best parameters to a JSON file
            save_best_params_to_json(best_params, classifier, id_number)
        else:
            # Define a default config for non-tuning run
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": TRAINING_PARAMS['lr'],
                "weight_decay": TRAINING_PARAMS['weight_decay'],
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
            }
            train_tcn(config, data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'])

            # Save the selected parameters to a JSON file
            save_best_params_to_json(config, classifier, id_number)

    elif classifier == 'LSTM':
        # Step 1.7.7: Perform Classification using LSTM. Subflowchart: LSTM Subflowchart. Train and validate the network using LSTM
        data = (Xtrain, ytrain, Xtest, ytest)

        if enable_tuning:
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": tune.loguniform(1e-4, 1e-1, TRAINING_PARAMS['lr']),
                "weight_decay": tune.loguniform(1e-5, 1e-2, TRAINING_PARAMS['weight_decay']),
                "lstm_hidden_s": TRAINING_PARAMS['lstm_hidden_s'],
                "fc1_hidden_s": TRAINING_PARAMS['fc1_hidden_s'],
                "dropout": tune.uniform(0.2, 0.3, TRAINING_PARAMS['dropout']),
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
            }

            scheduler = ASHAScheduler(
                metric="accuracy",
                mode="max",
                max_t=30,
                grace_period=1,
                reduction_factor=2
            )

            reporter = CLIReporter(
                metric_columns=["accuracy", "training_iteration"]
            )

            result = tune.run(
                tune.with_parameters(train_lstm, data=data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning']),
                resources_per_trial={"cpu": 2, "gpu": 1},
                config=config,
                num_samples=10,
                scheduler=scheduler,
                progress_reporter=reporter
            )

            best_trial = result.get_best_trial("accuracy", "max", "last")
            best_params = best_trial.config
            print("Best trial config: {}".format(best_params))
            print("Best trial final validation accuracy: {}".format(
                best_trial.last_result["accuracy"]))

            # Save the best parameters to a JSON file
            save_best_params_to_json(best_params, classifier, id_number)
        else:
            # Define a default config for non-tuning run
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": TRAINING_PARAMS['lr'],
                "weight_decay": TRAINING_PARAMS['weight_decay'],
                "lstm_hidden_s": TRAINING_PARAMS['lstm_hidden_s'],
                "fc1_hidden_s": TRAINING_PARAMS['fc1_hidden_s'],
                "dropout": TRAINING_PARAMS['dropout'],
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
            }
            train_lstm(config, data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'])

            # Save the selected parameters to a JSON file
            save_best_params_to_json(config, classifier, id_number)

    elif classifier == 'NNet':
        # Step 1.7.7: Perform Classification using NNet. Subflowchart: NNet Subflowchart. Train and validate the network using NNet
        data = (Xtrain, ytrain, Xtest, ytest)

        if enable_tuning:
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": tune.loguniform(1e-4, 1e-1, TRAINING_PARAMS['lr']),
                "weight_decay": tune.loguniform(1e-5, 1e-2, TRAINING_PARAMS['weight_decay']),
                "hidden_dim": TRAINING_PARAMS['hidden_dim'],
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
            }

            scheduler = ASHAScheduler(
                metric="accuracy",
                mode="max",
                max_t=30,
                grace_period=1,
                reduction_factor=2
            )

            reporter = CLIReporter(
                metric_columns=["accuracy", "training_iteration"]
            )

            result = tune.run(
                tune.with_parameters(train_nnet, data=data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning']),
                resources_per_trial={"cpu": 2, "gpu": 1},
                config=config,
                num_samples=10,
                scheduler=scheduler,
                progress_reporter=reporter
            )

            best_trial = result.get_best_trial("accuracy", "max", "last")
            best_params = best_trial.config
            print("Best trial config: {}".format(best_params))
            print("Best trial final validation accuracy: {}".format(
                best_trial.last_result["accuracy"]))

            # Save the best parameters to a JSON file
            save_best_params_to_json(best_params, classifier, id_number)
        else:
            # Define a default config for non-tuning run
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": TRAINING_PARAMS['lr'],
                "weight_decay": TRAINING_PARAMS['weight_decay'],
                "hidden_dim": TRAINING_PARAMS['hidden_dim'],
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
            }
            train_nnet(config, data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'])

            # Save the selected parameters to a JSON file
            save_best_params_to_json(config, classifier, id_number)

    elif classifier == 'DenseNet':
        # Step 1.7.7: Perform Classification using LSTM. Subflowchart: LSTM Subflowchart. Train and validate the network using LSTM
        data = (Xtrain, ytrain, Xtest, ytest)

        if enable_tuning:
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": tune.loguniform(1e-4, 1e-1, TRAINING_PARAMS['lr']),
                "weight_decay": tune.loguniform(1e-5, 1e-2, TRAINING_PARAMS['weight_decay']),
                "hidden_size": TRAINING_PARAMS['hidden_size'],
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
            }

            scheduler = ASHAScheduler(
                metric="accuracy",
                mode="max",
                max_t=30,
                grace_period=1,
                reduction_factor=2
            )

            reporter = CLIReporter(
                metric_columns=["accuracy", "training_iteration"]
            )

            result = tune.run(
                tune.with_parameters(train_densenet, data=data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning']),
                resources_per_trial={"cpu": 2, "gpu": 1},
                config=config,
                num_samples=10,
                scheduler=scheduler,
                progress_reporter=reporter
            )

            best_trial = result.get_best_trial("accuracy", "max", "last")
            best_params = best_trial.config
            print("Best trial config: {}".format(best_params))
            print("Best trial final validation accuracy: {}".format(
                best_trial.last_result["accuracy"]))

            # Save the best parameters to a JSON file
            save_best_params_to_json(best_params, classifier, id_number)
        else:
            # Define a default config for non-tuning run
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": TRAINING_PARAMS['lr'],
                "weight_decay": TRAINING_PARAMS['weight_decay'],
                "hidden_size": TRAINING_PARAMS['hidden_size'],
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
            }
            train_densenet(config, data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'])

            # Save the selected parameters to a JSON file
            save_best_params_to_json(config, classifier, id_number)
    elif classifier == 'MLP_Torch':
        data = (Xtrain, ytrain, Xtest, ytest)
        # Step 1.7.8: Perform Classification using MLP. Subflowchart: MLP Subflowchart. Train and validate the network using MLP
        if enable_tuning:
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": tune.loguniform(1e-4, 1e-1, TRAINING_PARAMS['lr']),
                "weight_decay": tune.loguniform(1e-5, 1e-2, TRAINING_PARAMS['weight_decay']),  # L2 regularization parameter
                "hidden_dim": TRAINING_PARAMS['hidden_dim'],
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
            }

            scheduler = ASHAScheduler(
                metric="accuracy",
                mode="max",
                max_t=30,
                grace_period=1,
                reduction_factor=2
            )

            reporter = CLIReporter(
                metric_columns=["accuracy", "training_iteration"]
            )

            result = tune.run(
                tune.with_parameters(train_mlp, data=data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning']),
                resources_per_trial={"cpu": 2, "gpu": 1},
                config=config,
                num_samples=10,
                scheduler=scheduler,
                progress_reporter=reporter
            )

            best_trial = result.get_best_trial("accuracy", "max", "last")
            best_params = best_trial.config
            print("Best trial config: {}".format(best_params))
            print("Best trial final validation accuracy: {}".format(
                best_trial.last_result["accuracy"]))

            # Save the best parameters to a JSON file
            save_best_params_to_json(best_params, classifier, id_number)
        else:
            # Define a default config for non-tuning run
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": TRAINING_PARAMS['lr'],
                "weight_decay": TRAINING_PARAMS['weight_decay']  # L2 regularization parameter,
                "hidden_dim": TRAINING_PARAMS['hidden_dim'],
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
            }
            train_mlp(config, data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'])

            # Save the selected parameters to a JSON file
            save_best_params_to_json(config, classifier, id_number)

    elif classifier == 'MLP':
        # Step 1.7.8: Perform Classification using MLP.
        model = MLPClassifier()

        # Define the parameter grid
        param_grid = {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
            'max_iter': [200, 500, 1000]
        }

        return train_and_evaluate_model(model, param_grid, 'MLP', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)
    elif classifier == 'DBSCAN':
        # Step 1.7.3: Perform Classification using DBSCAN.
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = DBSCAN()

        # Define the parameter grid
        param_grid = {
            'eps': [0.3, 0.5, 0.7],
            'min_samples': [5, 10, 15],
            'leaf_size': [10, 20, 30],
            'metric': ['euclidean', 'manhattan', 'chebyshev']
        }

        # If the best parameters exist, use them
        if best_params:
            model.set_params(**best_params)
            param_grid = {}

        return train_and_evaluate_model(model, param_grid, 'DBSCAN', X_train, Y_train, X_test, Y_test, args['id_number'], metric, args['search_method'], n_iterations)

def factors(n):
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

def save_params_to_json(df, *args):
    """
    Save the parameters to a JSON file.

    Args:
        df (DataFrame): The input DataFrame.
        *args: Variable length argument list containing the parameter values.

    Returns:
        str: The file path where the parameters are saved.
    """
    # Define parameter names and create a dictionary of params
    param_names = [
        'model', 'id_number', 'years', 'test_type','windowing', 'min_days_hdd', 'days_considered_as_failure',
        'test_train_percentage', 'oversample_undersample', 'balancing_normal_failed',
        'history_signal', 'classifier', 'features_extraction_method', 'cuda_dev',
        'ranking', 'num_features', 'overlap', 'split_technique', 'interpolate_technique',
        'search_method', 'fillna_method', 'pca_components', 'smoothing_level'
    ]

    # Assign values directly from the dictionary
    params = dict(zip(param_names, args))

    # Get column names that start with 'smart_'
    smart_columns = [col for col in df.columns if col.startswith('smart_')]

    # Add smart_columns to params under the key 'smart_attributes'
    params['smart_attributes'] = smart_columns

    script_dir = os.path.dirname(os.path.abspath(__file__))
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_dir = os.path.join(script_dir, '..', 'model', params['id_number'])

    # Create the directory if it doesn't exist
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    logger.info(f'Saving parameters: {params}')

    # Define the file path
    file_path = os.path.join(param_dir, f"{params['classifier'].lower()}_{params['id_number']}_params_{now_str}.json")

    # Write the params dictionary to a JSON file
    with open(file_path, 'w') as f:
        json.dump(params, f)

    logger.info(f'Parameters saved to: {file_path}')

    return file_path

def set_training_params(*args):
    param_names = ['reg', 'batch_size', 'lr', 'weight_decay', 'epochs', 'dropout', 'lstm_hidden_s', 'fc1_hidden_s', 'hidden_dim', 'hidden_size', 'num_layers', 'optimizer_type', 'num_workers']
    # Use the global keyword when modifying global variables
    global TRAINING_PARAMS
    TRAINING_PARAMS = dict(zip(param_names, args))
    # Print out updated parameters to Gradio interface
    return f"Parameters successfully updated:\n" + "\n".join([f"{key}: {value}" for key, value in TRAINING_PARAMS.items()])

def initialize_classification(*args):
    ray.init(num_cpus="12")

    os.environ["MODIN_ENGINE"] = "ray"  # Use ray as the execution engine

    # ------------------ #
    # Feature Selection Subflowchart
    # Step 1: Define empty lists and dictionary
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
        ],
        'iSTEP': [
            'date',
            'serial_number',
            'model',
            'failure',
            'smart_5_raw',
            'smart_3_raw',
            'smart_10_raw',
            'smart_12_raw',
            'smart_4_raw',
            'smart_194_raw',
            'smart_1_raw',
            'smart_9_raw',
            'smart_192_raw',
            'smart_193_raw',
            'smart_197_raw',
            'smart_198_raw',
            'smart_199_raw'
        ]
    }
    
    # Define parameter names and create a dictionary of params
    param_names = [
        'manufacturer', 'model', 'id_number', 'years', 'test_type', 'windowing', 'min_days_hdd', 'days_considered_as_failure',
        'test_train_percentage', 'oversample_undersample', 'balancing_normal_failed',
        'history_signal', 'classifier', 'features_extraction_method', 'cuda_dev',
        'ranking', 'num_features', 'overlap', 'split_technique', 'interpolate_technique',
        'search_method', 'fillna_method', 'pca_components', 'smoothing_level', 'incremental_learning', 'transfer_learning', 'partition_models',
        'enable_tuning', 'enable_ga_algorithm', 'number_pop', 'number_gen'
    ]

    # Assign values directly from the dictionary
    (
        manufacturer, model, id_number, years, test_type, windowing, min_days_HDD, days_considered_as_failure,
        test_train_perc, oversample_undersample, balancing_normal_failed,
        history_signal, classifier, features_extraction_method, CUDA_DEV,
        ranking, num_features, overlap, split_technique, interpolate_technique,
        search_method, fillna_method, pca_components, smoothing_level, incremental_learning, transfer_learning, partition_models,
        enable_tuning, enable_ga_algorithm, number_pop, number_gen
    ) = dict(zip(param_names, args)).values()
    models = [m.strip() for m in model.split(',')]
    model_string = "_".join(models)
    # here you can select the model. This is the one tested.
    # Correct years for the model
    # Select the statistical methods to extract features
    # many parameters that could be changed, both for unbalancing, for networks and for features.
    # minimum number of days for a HDD to be filtered out
    # Days considered as failure
    # percentage of the test set
    # type of oversampling: 0 means undersample, 1 means oversample, 2 means no balancing technique applied
    # The balance factor (major/minor = balancing_normal_failed) is used to balance the number of normal and failed samples in the dataset, default as 'auto'
    # length of the window
    # type of classifier
    # if you extract features for RF for example. Not tested
    # cuda device
    # if automatically select best features
    # number of SMART features to select
    # overlap option of the window
    # split technique for dataset partitioning
    # interpolation technique for the rows with missing dates

    try:
        # Try to convert to float
        balancing_normal_failed = float(balancing_normal_failed)
    except ValueError:
        # If it fails, it's a string and we leave it as is
        pass

    logger.info(f'Logger initialized successfully! Current id number is: {id_number}')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'output')
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Step 1: Load the dataset from pkl file.
        df = pd.read_pickle(os.path.join(script_dir, '..', 'output', f'{model_string}_Dataset_selected_windowed_{history_signal}_rank_{ranking}_{num_features}_overlap_{overlap}.pkl'))
        logger.info(f'Loading file from pickle file: {model_string}_Dataset_selected_windowed_{history_signal}_rank_{ranking}_{num_features}_overlap_{overlap}.pkl')
    except:
        # Step 1.1: Import the dataset from the raw data.
        if ranking == 'None':
            df = import_data(years=years, models=models, name='iSTEP', features=features, manufacturer=manufacturer)
        else:
            df = import_data(years=years, models=models, name='iSTEP', manufacturer=manufacturer)

        # Check if the DataFrame is empty
        if df.empty:
            raise ValueError("The DataFrame is empty. Please check your data source.")

        logger.info('Data imported successfully, processing smart attributes...')
        for column in list(df):
            missing = round(df[column].notna().sum() / df.shape[0] * 100, 2)
            logger.info(f"{column:<27}.{missing}%")
        # Step 1.2: Filter out the bad HDDs.
        bad_missing_hds, bad_power_hds, df = filter_HDs_out(df, min_days=min_days_HDD, time_window='30D', tolerance=2)
        # predict_val represents the prediction value of the failure
        # validate_val represents the validation value of the failure
        # Step 1.3: Define RUL(Remain useful life) Piecewise
        df, pred_list, valid_list = generate_failure_predictions(df, days=days_considered_as_failure, window=history_signal)

        # Create a new DataFrame with the results since the previous function may filter out some groups from the DataFrame
        df['predict_val'] = pred_list
        df['validate_val'] = valid_list

        if ranking != 'None':
            enable_ga_algorithm = True
            # Step 1.4: Feature Selection: Subflow chart of Main Classification Process
            # n_pop: Number of individuals in each generation
            # n_gen: Stop the genetic algorithm after certain generations
            df = feature_selection(df, num_features, test_type, enable_ga_algorithm, n_pop=number_pop, n_gen=number_gen)
        logger.info('Used features')
        for column in list(df):
            logger.info(f'{column:<27}.')
        logger.info(f'Saving to pickle file: {model_string}_Dataset_selected_windowed_{history_signal}_rank_{ranking}_{num_features}_overlap_{overlap}.pkl')
        df.to_pickle(os.path.join(output_dir, f'{model_string}_Dataset_selected_windowed_{history_signal}_rank_{ranking}_{num_features}_overlap_{overlap}.pkl'))

    # Interpolate data for the rows with missing dates
    if interpolate_technique != 'None':
        df = interpolate_ts(df, method=interpolate_technique)

    if manufacturer != 'custom' and partition_models == True:
        relevant_models, irrelevant_models = find_relevant_models(df)
        # Filter the original DataFrame based on the 'model' column
        relevant_df = df[df['model'].isin(relevant_models)]
        irrelevant_df = df[df['model'].isin(irrelevant_models)]

    # Saving parameters to json file
    logger.info('Saving parameters to json file...')
    param_path = save_params_to_json(
        df, model_string, id_number, years, test_type, windowing, min_days_HDD, days_considered_as_failure,
        test_train_perc, oversample_undersample, balancing_normal_failed,
        history_signal, classifier, features_extraction_method, CUDA_DEV,
        ranking, num_features, overlap, split_technique, interpolate_technique,
        search_method, enable_tuning, fillna_method, pca_components, smoothing_level
    )

    if transfer_learning:
        if partition_models == True:
            # Partition the dataset into training and testing sets for the relevant_df
            Xtrain, ytrain, Xtest, ytest = initialize_partitioner(
                relevant_df, output_dir, model_string, windowing, test_train_perc, 
                oversample_undersample, balancing_normal_failed, history_signal, 
                classifier, features_extraction_method, ranking, num_features, 
                overlap, split_technique, fillna_method, pca_components, smoothing_level
            )

            # Perform classification for the relevant_df
            perform_classification(Xtrain, ytrain, Xtest, ytest, id_number, 
                classifier, CUDA_DEV, search_method, enable_tuning, incremental_learning, False, param_path
            )

            # Partition the dataset into training and testing sets for the irrelevant_df
            Xtrain, ytrain, Xtest, ytest = initialize_partitioner(
                irrelevant_df, output_dir, model_string, windowing, test_train_perc, 
                oversample_undersample, balancing_normal_failed, history_signal, 
                classifier, features_extraction_method, ranking, num_features, 
                overlap, split_technique, fillna_method, pca_components, smoothing_level
            )

            # Perform classification for the irrelevant_df
            return perform_classification(Xtrain, ytrain, Xtest, ytest, id_number, 
                classifier, CUDA_DEV, search_method, enable_tuning, enable_tuning, incremental_learning, True, param_path
            )
        else:
            # Partition the dataset into training and testing sets for the irrelevant_df
            Xtrain, ytrain, Xtest, ytest = initialize_partitioner(
                df, output_dir, model_string, windowing, test_train_perc, 
                oversample_undersample, balancing_normal_failed, history_signal, 
                classifier, features_extraction_method, ranking, num_features, 
                overlap, split_technique, fillna_method, pca_components, smoothing_level
            )

            # Perform classification for the irrelevant_df
            return perform_classification(Xtrain, ytrain, Xtest, ytest, id_number, 
                classifier, CUDA_DEV, search_method, enable_tuning, incremental_learning, True, param_path
            )
    else:
        if partition_models == False:
            # Partition the dataset into training and testing sets for entire df
            Xtrain, ytrain, Xtest, ytest = initialize_partitioner(
                df, output_dir, model_string, windowing, test_train_perc, 
                oversample_undersample, balancing_normal_failed, history_signal, 
                classifier, features_extraction_method, ranking, num_features, 
                overlap, split_technique, fillna_method, pca_components, smoothing_level
            )

        else:
            # Partition the dataset into training and testing sets for relevant_df
            Xtrain, ytrain, Xtest, ytest = initialize_partitioner(
                relevant_df, output_dir, model_string, windowing, test_train_perc, 
                oversample_undersample, balancing_normal_failed, history_signal, 
                classifier, features_extraction_method, ranking, num_features, 
                overlap, split_technique, fillna_method, pca_components, smoothing_level
            )

        # Perform classification for the relevant_df
        return perform_classification(Xtrain, ytrain, Xtest, ytest, id_number, 
            classifier, CUDA_DEV, search_method, enable_tuning, incremental_learning, False, param_path
        )

def initialize_partitioner(df, *args):
    # Define parameter names and create a dictionary of params
    param_names = [
        'output_dir', 'model_string', 'windowing',
        'test_train_percentage', 'oversample_undersample', 'balancing_normal_failed',
        'history_signal', 'classifier', 'features_extraction_method',
        'ranking', 'num_features', 'overlap', 'split_technique', 'interpolate_technique',
        'search_method', 'fillna_method', 'pca_components', 'smoothing_level'
    ]

    # Assign values directly from the dictionary
    (
        output_dir, model_string, windowing, 
        test_train_perc, oversample_undersample, balancing_normal_failed,
        history_signal, classifier, features_extraction_method,
        ranking, num_features, overlap, split_technique,
        fillna_method, pca_components, smoothing_level
    ) = dict(zip(param_names, args)).values()
    ## -------- ##
    # random: stratified without keeping time order
    # hdd --> separate different hdd (need FIXes)
    # temporal --> separate by time (need FIXes)
    # Step 1.5: Partition the dataset into training and testing sets. Partition Dataset: Subflow chart of Main Classification Process
    Xtrain, Xtest, ytrain, ytest = DatasetPartitioner(
        df,
        model_string,
        overlap=overlap,
        rank=ranking,
        num_features=num_features,
        technique=split_technique,
        test_train_perc=test_train_perc,
        windowing=windowing,
        window_dim=history_signal,
        resampler_balancing=balancing_normal_failed,
        oversample_undersample=oversample_undersample,
        fillna_method=fillna_method,
        smoothing_level=smoothing_level
    )

    # Print the line of Xtrain and Xtest
    logger.info(f'Xtrain shape: {Xtrain.shape}, Xtest shape: {Xtest.shape}')

    try:
        data = np.load(os.path.join(output_dir, f'{model_string}_training_and_testing_data_{history_signal}_rank_{ranking}_{num_features}_overlap_{overlap}_features_extraction_method_{features_extraction_method}_oversample_undersample_{oversample_undersample}.npz'))
        Xtrain, Xtest, ytrain, ytest = data['Xtrain'], data['Xtest'], data['Ytrain'], data['Ytest']
    except:
        # Step x.1: Feature Extraction
        if features_extraction_method == 'custom':
            # Extract features for the train and test set
            Xtrain = feature_extraction(Xtrain)
            Xtest = feature_extraction(Xtest)
        elif features_extraction_method == 'PCA':
            Xtrain = feature_extraction_PCA(Xtrain, pca_components)
            Xtest = feature_extraction_PCA(Xtest, pca_components)
        elif features_extraction_method == 'None':
            logger.info('Skipping features extraction for training data.')
        else:
            raise ValueError('Invalid features extraction method. Please choose either "custom" or "pca" or "None".')
        logger.info(f'Saving training and testing data to file: {model_string}_training_and_testing_data_{history_signal}_rank_{ranking}_{num_features}_overlap_{overlap}_features_extraction_method_{features_extraction_method}_oversample_undersample_{oversample_undersample}.npz')
        # Save the arrays to a .npz file
        np.savez(os.path.join(output_dir, f'{model_string}_training_and_testing_data_{history_signal}_rank_{ranking}_{num_features}_overlap_{overlap}_features_extraction_method_{features_extraction_method}_oversample_undersample_{oversample_undersample}.npz'), Xtrain=Xtrain, Xtest=Xtest, Ytrain=ytrain, Ytest=ytest)

    ## ---------------------------- ##
    # Step x.2: Reshape the data for RandomForest: We jumped from Step 1.6.1, use third-party RandomForest library
    classifiers = [
        'RandomForest', 
        'KNeighbors', 
        'DecisionTree', 
        'LogisticRegression', 
        'SVM', 
        'XGB', 
        'MLP', 
        'IsolationForest', 
        'ExtraTrees', 
        'GradientBoosting', 
        'NaiveBayes', 
        'DBSCAN',
        'RGF'
    ]
    if classifier in classifiers and windowing == 1:
        Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
        Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])

def perform_classification(*args):
    # Define parameter names and create a dictionary of params
    param_names = [
        'Xtrain', 'ytrain', 'Xtest', 'ytest',
        'id_number', 'classifier', 'cuda_dev',
        'search_method', 'enable_tuning', 'incremental_learning',
        'transfer_learning', 'param_path'
    ]

    # Assign values directly from the dictionary
    (
        Xtrain, ytrain, Xtest, ytest,
        id_number, classifier, CUDA_DEV,
        search_method, enable_tuning, incremental_learning,
        transfer_learning, param_path
    ) = dict(zip(param_names, args)).values()

    # Step 1.6: Classifier Selection: set training parameters
    ####### CLASSIFIER PARAMETERS #######
    if CUDA_DEV != 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEV

    try:
        # Parameters for TCN and LSTM networks
        model_path = classification(
            X_train=Xtrain,
            Y_train=ytrain,
            X_test=Xtest,
            Y_test=ytest,
            classifier=classifier,
            metric=['RMSE', 'MAE', 'FDR', 'FAR', 'F1', 'recall', 'precision'],
            net=net,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            reg=reg,
            id_number=id_number,
            num_workers=num_workers,
            enable_tuning=enable_tuning,
            incremental_learning=incremental_learning,
            transfer_learning=incremental_learning
        )
    except:
        # Parameters for RandomForest
        model_path = classification(
            X_train=Xtrain,
            Y_train=ytrain,
            X_test=Xtest,
            Y_test=ytest,
            classifier=classifier,
            # FDR, FAR, F1, recall, precision are not calculated for some algorithms, it will report as 0.0
            metric=['RMSE', 'MAE', 'FDR', 'FAR', 'F1', 'recall', 'precision'],
            search_method=search_method,
            id_number=id_number
        )

    return logger.get_log_file_path(), model_path, param_path