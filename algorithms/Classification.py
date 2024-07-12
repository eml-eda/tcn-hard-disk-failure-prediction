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
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from rgf.sklearn import RGFClassifier
import logger
import ray
from ray import tune
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune import CLIReporter
from json_param import save_best_params_to_json, load_best_params_from_json, save_params_to_json
from network_training import train_and_evaluate_model, train_dl_model


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
    'num_workers': 8,
    'scheduler_type': 'ReduceLROnPlateau',
    'scheduler_factor': 0.1,
    'scheduler_patience': 10,
    'scheduler_step_size': 30,
    'scheduler_gamma': 0.9,
    'loss_function': 'CrossEntropy'
}

def classification(X_train, Y_train, X_test, Y_test, classifier, **args):
    """
    Perform classification using the specified classifier.
    --- Step 1.7: Perform Classification
    Parameters:
    - X_train (array-like): Training data features.
    - Y_train (array-like): Training data labels.
    - X_test (array-like): Test data features.
    - Y_test (array-like): Test data labels.
    - classifier (str): The classifier to use. Options: 'RandomForest', 'TCN', 'LSTM'.
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

        # Default parameters
        default_params = {
            'n_estimators': 1000,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto',
            'max_depth': None,
            'criterion': 'gini',
            'bootstrap': True
        }

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'RandomForest',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

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

        # Default parameters
        default_params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'metric': 'minkowski'
        }

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'KNeighbors',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

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

        # Default parameters
        default_params = {
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None
        }

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'DecisionTree',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

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

        # Default parameters
        default_params = {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'lbfgs',
            'max_iter': 100
        }

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'LogisticRegression',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

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

        # Default parameters
        default_params = {
            'C': 1.0,
            'gamma': 'scale',
            'kernel': 'rbf'
        }

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'SVM',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

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

        # Default parameters
        default_params = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 3,
            'min_child_weight': 1,
            'gamma': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'objective': 'binary:logistic'
        }

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'XGB',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

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

        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_samples': 'auto',
            'contamination': 'auto',
            'max_features': 1.0,
            'bootstrap': False
        }

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'IsolationForest',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

    elif classifier == 'ExtraTrees':
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = ExtraTreesClassifier()

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }

        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_features': 'auto',
            'bootstrap': False
        }

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'ExtraTreesClassifier',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

    elif classifier == 'GradientBoosting':
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = GradientBoostingClassifier()

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.1, 0.05, 0.01],
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10]
        }

        # Default parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'GradientBoostingClassifier',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

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

        # Default parameters
        default_params = {}

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'NaiveBayes',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

    elif classifier == 'RGF':
        # Step 1.7.7: Perform Classification using Regularized Greedy Forest (RGF).
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = RGFClassifier()

        # Define the parameter grid
        param_grid = {
            'max_leaf': [1000, 1200, 1500],
            'algorithm': ["RGF", "RGF_Opt", "RGF_Sib"],
            'test_interval': [100, 600, 900]
        }

        # Default parameters
        default_params = {
            'max_leaf': 1000,
            'algorithm': 'RGF',
            'test_interval': 100
        }

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'RGF',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

    elif classifier == 'TCN':
        # Step 1.7.6: Perform Classification using TCN. Subflowchart: TCN Subflowchart. Train and validate the network using TCN
        data = (Xtrain, ytrain, Xtest, ytest)

        if args['enable_tuning']:
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": tune.loguniform(TRAINING_PARAMS['lr'] / 10, TRAINING_PARAMS['lr'] * 10),
                "weight_decay": tune.loguniform(TRAINING_PARAMS['weight_decay'] / 10, TRAINING_PARAMS['weight_decay'] * 10),
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
                "scheduler_type": TRAINING_PARAMS['scheduler_type'],
                "loss_function": TRAINING_PARAMS['loss_function']
            }
            if TRAINING_PARAMS['scheduler_type'] == 'StepLR':
                config["scheduler_step_size"] = TRAINING_PARAMS['scheduler_step_size']
                config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
            elif TRAINING_PARAMS['scheduler_type'] == 'ExponentialLR':
                config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
            elif TRAINING_PARAMS['scheduler_type'] == 'ReduceLROnPlateau':
                config["scheduler_factor"] = TRAINING_PARAMS['scheduler_factor']
                config["scheduler_patience"] = TRAINING_PARAMS['scheduler_patience']

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
                tune.with_parameters(train_dl_model, data=data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'], classifier='TCN', id_number=args['id_number']),
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
            save_best_params_to_json(best_params, classifier, args['id_number'])
        else:
            try:
                config = load_best_params_from_json(classifier, args['id_number'])
            except FileNotFoundError:
                # Define a default config for non-tuning run
                config = {
                    "epochs": TRAINING_PARAMS['epochs'],
                    "batch_size": TRAINING_PARAMS['batch_size'],
                    "lr": TRAINING_PARAMS['lr'],
                    "weight_decay": TRAINING_PARAMS['weight_decay'],
                    "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                    "reg": TRAINING_PARAMS['reg'],
                    "num_workers": TRAINING_PARAMS['num_workers'],
                    "scheduler_type": TRAINING_PARAMS['scheduler_type'],
                    "loss_function": TRAINING_PARAMS['loss_function']
                }
                if TRAINING_PARAMS['scheduler_type'] == 'StepLR':
                    config["scheduler_step_size"] = TRAINING_PARAMS['scheduler_step_size']
                    config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
                elif TRAINING_PARAMS['scheduler_type'] == 'ExponentialLR':
                    config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
                elif TRAINING_PARAMS['scheduler_type'] == 'ReduceLROnPlateau':
                    config["scheduler_factor"] = TRAINING_PARAMS['scheduler_factor']
                    config["scheduler_patience"] = TRAINING_PARAMS['scheduler_patience']
            train_dl_model(config, data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'], classifier='TCN', id_number=args['id_number'])

            # Save the selected parameters to a JSON file
            save_best_params_to_json(config, classifier, args['id_number'])

    elif classifier == 'LSTM':
        # Step 1.7.7: Perform Classification using LSTM. Subflowchart: LSTM Subflowchart. Train and validate the network using LSTM
        data = (Xtrain, ytrain, Xtest, ytest)

        if args['enable_tuning']:
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": tune.loguniform(TRAINING_PARAMS['lr'] / 10, TRAINING_PARAMS['lr'] * 10),
                "weight_decay": tune.loguniform(TRAINING_PARAMS['weight_decay'] / 10, TRAINING_PARAMS['weight_decay'] * 10),
                "lstm_hidden_s": TRAINING_PARAMS['lstm_hidden_s'],
                "fc1_hidden_s": TRAINING_PARAMS['fc1_hidden_s'],
                "dropout": tune.uniform(0.2, 0.3, TRAINING_PARAMS['dropout']),
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
                "scheduler_type": TRAINING_PARAMS['scheduler_type'],
                "loss_function": TRAINING_PARAMS['loss_function']
            }
            if TRAINING_PARAMS['scheduler_type'] == 'StepLR':
                config["scheduler_step_size"] = TRAINING_PARAMS['scheduler_step_size']
                config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
            elif TRAINING_PARAMS['scheduler_type'] == 'ExponentialLR':
                config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
            elif TRAINING_PARAMS['scheduler_type'] == 'ReduceLROnPlateau':
                config["scheduler_factor"] = TRAINING_PARAMS['scheduler_factor']
                config["scheduler_patience"] = TRAINING_PARAMS['scheduler_patience']

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
                tune.with_parameters(train_dl_model, data=data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'], classifier='FPLSTM', id_number=args['id_number']),
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
            save_best_params_to_json(best_params, classifier, args['id_number'])
        else:
            try:
                config = load_best_params_from_json(classifier, args['id_number'])
            except FileNotFoundError:
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
                    "scheduler_type": TRAINING_PARAMS['scheduler_type'],
                    "loss_function": TRAINING_PARAMS['loss_function']
                }
                if TRAINING_PARAMS['scheduler_type'] == 'StepLR':
                    config["scheduler_step_size"] = TRAINING_PARAMS['scheduler_step_size']
                    config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
                elif TRAINING_PARAMS['scheduler_type'] == 'ExponentialLR':
                    config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
                elif TRAINING_PARAMS['scheduler_type'] == 'ReduceLROnPlateau':
                    config["scheduler_factor"] = TRAINING_PARAMS['scheduler_factor']
                    config["scheduler_patience"] = TRAINING_PARAMS['scheduler_patience']
            train_dl_model(config, data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'], classifier='FPLSTM', id_number=args['id_number'])

            # Save the selected parameters to a JSON file
            save_best_params_to_json(config, classifier, args['id_number'])

    elif classifier == 'NNet':
        # Step 1.7.7: Perform Classification using NNet. Subflowchart: NNet Subflowchart. Train and validate the network using NNet
        data = (Xtrain, ytrain, Xtest, ytest)

        if args['enable_tuning']:
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": tune.loguniform(TRAINING_PARAMS['lr'] / 10, TRAINING_PARAMS['lr'] * 10),
                "weight_decay": tune.loguniform(TRAINING_PARAMS['weight_decay'] / 10, TRAINING_PARAMS['weight_decay'] * 10),
                "hidden_dim": TRAINING_PARAMS['hidden_dim'],
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
                "scheduler_type": TRAINING_PARAMS['scheduler_type'],
                "loss_function": TRAINING_PARAMS['loss_function']
            }
            if TRAINING_PARAMS['scheduler_type'] == 'StepLR':
                config["scheduler_step_size"] = TRAINING_PARAMS['scheduler_step_size']
                config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
            elif TRAINING_PARAMS['scheduler_type'] == 'ExponentialLR':
                config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
            elif TRAINING_PARAMS['scheduler_type'] == 'ReduceLROnPlateau':
                config["scheduler_factor"] = TRAINING_PARAMS['scheduler_factor']
                config["scheduler_patience"] = TRAINING_PARAMS['scheduler_patience']

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
                tune.with_parameters(train_dl_model, data=data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'], classifier='NNet', id_number=args['id_number']),
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
            save_best_params_to_json(best_params, classifier, args['id_number'])
        else:
            try:
                config = load_best_params_from_json(classifier, args['id_number'])
            except FileNotFoundError:
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
                    "scheduler_type": TRAINING_PARAMS['scheduler_type'],
                    "loss_function": TRAINING_PARAMS['loss_function']
                }
                if TRAINING_PARAMS['scheduler_type'] == 'StepLR':
                    config["scheduler_step_size"] = TRAINING_PARAMS['scheduler_step_size']
                    config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
                elif TRAINING_PARAMS['scheduler_type'] == 'ExponentialLR':
                    config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
                elif TRAINING_PARAMS['scheduler_type'] == 'ReduceLROnPlateau':
                    config["scheduler_factor"] = TRAINING_PARAMS['scheduler_factor']
                    config["scheduler_patience"] = TRAINING_PARAMS['scheduler_patience']
            train_dl_model(config, data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'], classifier='NNet', id_number=args['id_number'])

            # Save the selected parameters to a JSON file
            save_best_params_to_json(config, classifier, args['id_number'])

    elif classifier == 'DenseNet':
        # Step 1.7.7: Perform Classification using LSTM. Subflowchart: LSTM Subflowchart. Train and validate the network using LSTM
        data = (Xtrain, ytrain, Xtest, ytest)

        if args['enable_tuning']:
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": tune.loguniform(1e-4, 1e-1, TRAINING_PARAMS['lr']),
                "weight_decay": tune.loguniform(1e-5, 1e-2, TRAINING_PARAMS['weight_decay']),
                "hidden_size": TRAINING_PARAMS['hidden_size'],
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
                "scheduler_type": TRAINING_PARAMS['scheduler_type'],
                "loss_function": TRAINING_PARAMS['loss_function']
            }
            if TRAINING_PARAMS['scheduler_type'] == 'StepLR':
                config["scheduler_step_size"] = TRAINING_PARAMS['scheduler_step_size']
                config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
            elif TRAINING_PARAMS['scheduler_type'] == 'ExponentialLR':
                config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
            elif TRAINING_PARAMS['scheduler_type'] == 'ReduceLROnPlateau':
                config["scheduler_factor"] = TRAINING_PARAMS['scheduler_factor']
                config["scheduler_patience"] = TRAINING_PARAMS['scheduler_patience']

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
                tune.with_parameters(train_dl_model, data=data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'], classifier='DenseNet', id_number=args['id_number']),
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
            save_best_params_to_json(best_params, classifier, args['id_number'])
        else:
            try:
                config = load_best_params_from_json(classifier, args['id_number'])
            except FileNotFoundError:
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
                    "scheduler_type": TRAINING_PARAMS['scheduler_type'],
                    "loss_function": TRAINING_PARAMS['loss_function']
                }
                if TRAINING_PARAMS['scheduler_type'] == 'StepLR':
                    config["scheduler_step_size"] = TRAINING_PARAMS['scheduler_step_size']
                    config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
                elif TRAINING_PARAMS['scheduler_type'] == 'ExponentialLR':
                    config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
                elif TRAINING_PARAMS['scheduler_type'] == 'ReduceLROnPlateau':
                    config["scheduler_factor"] = TRAINING_PARAMS['scheduler_factor']
                    config["scheduler_patience"] = TRAINING_PARAMS['scheduler_patience']
            train_dl_model(config, data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'], classifier='DenseNet', id_number=args['id_number'])

            # Save the selected parameters to a JSON file
            save_best_params_to_json(config, classifier, args['id_number'])
    elif classifier == 'MLP_Torch':
        data = (Xtrain, ytrain, Xtest, ytest)
        # Step 1.7.8: Perform Classification using MLP. Subflowchart: MLP Subflowchart. Train and validate the network using MLP
        if args['enable_tuning']:
            config = {
                "epochs": TRAINING_PARAMS['epochs'],
                "batch_size": TRAINING_PARAMS['batch_size'],
                "lr": tune.loguniform(TRAINING_PARAMS['lr'] / 10, TRAINING_PARAMS['lr'] * 10),
                "weight_decay": tune.loguniform(TRAINING_PARAMS['weight_decay'] / 10, TRAINING_PARAMS['weight_decay'] * 10),  # L2 regularization parameter
                "hidden_dim": TRAINING_PARAMS['hidden_dim'],
                "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                "reg": TRAINING_PARAMS['reg'],
                "num_workers": TRAINING_PARAMS['num_workers'],
                "scheduler_type": TRAINING_PARAMS['scheduler_type'],
                "loss_function": TRAINING_PARAMS['loss_function']
            }
            if TRAINING_PARAMS['scheduler_type'] == 'StepLR':
                config["scheduler_step_size"] = TRAINING_PARAMS['scheduler_step_size']
                config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
            elif TRAINING_PARAMS['scheduler_type'] == 'ExponentialLR':
                config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
            elif TRAINING_PARAMS['scheduler_type'] == 'ReduceLROnPlateau':
                config["scheduler_factor"] = TRAINING_PARAMS['scheduler_factor']
                config["scheduler_patience"] = TRAINING_PARAMS['scheduler_patience']

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
                tune.with_parameters(train_dl_model, data=data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'], classifier='MLP', id_number=args['id_number']),
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
            save_best_params_to_json(best_params, classifier, args['id_number'])
        else:
            try:
                config = load_best_params_from_json(classifier, args['id_number'])
            except FileNotFoundError:
                # Define a default config for non-tuning run
                config = {
                    "epochs": TRAINING_PARAMS['epochs'],
                    "batch_size": TRAINING_PARAMS['batch_size'],
                    "lr": tune.loguniform(TRAINING_PARAMS['lr'] / 10, TRAINING_PARAMS['lr'] * 10),
                    "weight_decay": tune.loguniform(TRAINING_PARAMS['weight_decay'] / 10, TRAINING_PARAMS['weight_decay'] * 10),  # L2 regularization parameter,
                    "hidden_dim": TRAINING_PARAMS['hidden_dim'],
                    "optimizer_type": TRAINING_PARAMS['optimizer_type'],
                    "reg": TRAINING_PARAMS['reg'],
                    "num_workers": TRAINING_PARAMS['num_workers'],
                    "scheduler_type": TRAINING_PARAMS['scheduler_type'],
                    "loss_function": TRAINING_PARAMS['loss_function']
                }
                if TRAINING_PARAMS['scheduler_type'] == 'StepLR':
                    config["scheduler_step_size"] = TRAINING_PARAMS['scheduler_step_size']
                    config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
                elif TRAINING_PARAMS['scheduler_type'] == 'ExponentialLR':
                    config["scheduler_gamma"] = TRAINING_PARAMS['scheduler_gamma']
                elif TRAINING_PARAMS['scheduler_type'] == 'ReduceLROnPlateau':
                    config["scheduler_factor"] = TRAINING_PARAMS['scheduler_factor']
                    config["scheduler_patience"] = TRAINING_PARAMS['scheduler_patience']
            train_dl_model(config, data, enable_tuning=args['enable_tuning'], incremental_learning=args['incremental_learning'], transfer_learning=args['transfer_learning'], classifier='MLP', id_number=args['id_number'])

            # Save the selected parameters to a JSON file
            save_best_params_to_json(config, classifier, args['id_number'])

    elif classifier == 'MLP':
        # Step 1.7.8: Perform Classification using MLP.
        try:
            best_params = load_best_params_from_json(classifier, args['id_number'])
        except FileNotFoundError:
            best_params = None

        model = MLPClassifier()

        # Define the parameter grid
        param_grid = {
            'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [200, 500, 1000]
        }

        # Default parameters
        default_params = {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'constant',
            'max_iter': 200
        }

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'MLP',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

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

        # Default parameters
        default_params = {
            'eps': 0.5,
            'min_samples': 5,
            'leaf_size': 30,
            'metric': 'euclidean'
        }

        if not args['enable_tuning']:
            if best_params:
                model.set_params(**best_params)
            else:
                model.set_params(**default_params)
            param_grid = {}

        return train_and_evaluate_model(
            model,
            param_grid,
            'DBSCAN',
            X_train,
            Y_train,
            X_test,
            Y_test,
            args['id_number'],
            args['metric'],
            args['search_method'],
            args['n_iterations'],
            args['launch_dashboard']
        )

def set_training_params(*args):
    """
    Set the training parameters for the model.

    Args:
        *args: Variable number of arguments representing the training parameters.

    Returns:
        str: A string indicating that the parameters have been successfully updated.

    """
    param_names = [
        'reg',
        'batch_size',
        'lr',
        'weight_decay',
        'epochs',
        'dropout',
        'lstm_hidden_s',
        'fc1_hidden_s',
        'hidden_dim',
        'hidden_size',
        'num_layers',
        'optimizer_type',
        'num_workers',
        'scheduler_type',
        'scheduler_factor',
        'scheduler_patience',
        'scheduler_step_size',
        'scheduler_gamma',
        'loss_function'
    ]
    # Use the global keyword when modifying global variables
    global TRAINING_PARAMS # pylint: disable=global-statement
    TRAINING_PARAMS = dict(zip(param_names, args))
    # Print out updated parameters to Gradio interface
    return "Parameters successfully updated:\n" + "\n".join([f"{key}: {value}" for key, value in TRAINING_PARAMS.items()])

def initialize_classification(*args):
    ray.init()

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
        'enable_tuning', 'enable_ga_algorithm', 'number_pop', 'number_gen', 'apply_weighted_feature', 'max_wavelet_scales'
        'launch_dashboard'
    ]

    # Assign values directly from the dictionary
    (
        manufacturer, model, id_number, years, test_type, windowing, min_days_HDD, days_considered_as_failure,
        test_train_perc, oversample_undersample, balancing_normal_failed,
        history_signal, classifier, features_extraction_method, CUDA_DEV,
        ranking, num_features, overlap, split_technique, interpolate_technique,
        search_method, fillna_method, pca_components, smoothing_level, incremental_learning, transfer_learning, partition_models,
        enable_tuning, enable_ga_algorithm, number_pop, number_gen, apply_weighted_feature, max_wavelet_scales,
        launch_dashboard
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
    except: # pylint: disable=bare-except
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
            df, feature_weights = feature_selection(df, num_features, test_type, enable_ga_algorithm, n_pop=number_pop, n_gen=number_gen)
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
        search_method, enable_tuning, fillna_method, pca_components, smoothing_level, max_wavelet_scales
    )

    if transfer_learning:
        if partition_models == True:
            # Partition the dataset into training and testing sets for the relevant_df
            Xtrain, ytrain, Xtest, ytest = initialize_partitioner(
                relevant_df, output_dir, model_string, windowing, test_train_perc, 
                oversample_undersample, balancing_normal_failed, history_signal, 
                classifier, features_extraction_method, ranking, num_features, 
                overlap, split_technique, fillna_method, pca_components, smoothing_level,
                apply_weighted_feature, feature_weights, max_wavelet_scales
            )

            # Perform classification for the relevant_df
            perform_classification(Xtrain, ytrain, Xtest, ytest, id_number, 
                classifier, CUDA_DEV, search_method, enable_tuning, incremental_learning, False,
                launch_dashboard, param_path
            )

            # Partition the dataset into training and testing sets for the irrelevant_df
            Xtrain, ytrain, Xtest, ytest = initialize_partitioner(
                irrelevant_df, output_dir, model_string, windowing, test_train_perc, 
                oversample_undersample, balancing_normal_failed, history_signal, 
                classifier, features_extraction_method, ranking, num_features, 
                overlap, split_technique, fillna_method, pca_components, smoothing_level,
                apply_weighted_feature, feature_weights, max_wavelet_scales
            )

            # Perform classification for the irrelevant_df
            return perform_classification(Xtrain, ytrain, Xtest, ytest, id_number, 
                classifier, CUDA_DEV, search_method, enable_tuning, incremental_learning, True,
                launch_dashboard, param_path
            )
        else:
            # Partition the dataset into training and testing sets for the irrelevant_df
            Xtrain, ytrain, Xtest, ytest = initialize_partitioner(
                df, output_dir, model_string, windowing, test_train_perc, 
                oversample_undersample, balancing_normal_failed, history_signal, 
                classifier, features_extraction_method, ranking, num_features, 
                overlap, split_technique, fillna_method, pca_components, smoothing_level,
                apply_weighted_feature, feature_weights, max_wavelet_scales
            )

            # Perform classification for the irrelevant_df
            return perform_classification(Xtrain, ytrain, Xtest, ytest, id_number, 
                classifier, CUDA_DEV, search_method, enable_tuning, incremental_learning, True,
                launch_dashboard, param_path
            )
    else:
        if partition_models == False:
            # Partition the dataset into training and testing sets for entire df
            Xtrain, ytrain, Xtest, ytest = initialize_partitioner(
                df, output_dir, model_string, windowing, test_train_perc, 
                oversample_undersample, balancing_normal_failed, history_signal, 
                classifier, features_extraction_method, ranking, num_features, 
                overlap, split_technique, fillna_method, pca_components, smoothing_level,
                apply_weighted_feature, feature_weights, max_wavelet_scales
            )

        else:
            # Partition the dataset into training and testing sets for relevant_df
            Xtrain, ytrain, Xtest, ytest = initialize_partitioner(
                relevant_df, output_dir, model_string, windowing, test_train_perc, 
                oversample_undersample, balancing_normal_failed, history_signal, 
                classifier, features_extraction_method, ranking, num_features, 
                overlap, split_technique, fillna_method, pca_components, smoothing_level,
                apply_weighted_feature, feature_weights, max_wavelet_scales
            )

        # Perform classification for the relevant_df
        return perform_classification(Xtrain, ytrain, Xtest, ytest, id_number, 
            classifier, CUDA_DEV, search_method, enable_tuning, incremental_learning, False,
            launch_dashboard, param_path
        )

def apply_feature_weights(data, feature_weights):
    """
    Applies the feature weights to the input training data.

    Args:
        data (pd.DataFrame): The input training data.
        feature_weights (dict): A dictionary with feature names as keys and their weights as values.

    Returns:
        pd.DataFrame: The weighted input training data.
    """
    if feature_weights is not None:
        for feature, weight in feature_weights.items():
            if feature in data.columns:
                data[feature] *= weight
    return data

def initialize_partitioner(df, *args):
    # Define parameter names and create a dictionary of params
    param_names = [
        'output_dir', 'model_string', 'windowing',
        'test_train_percentage', 'oversample_undersample', 'balancing_normal_failed',
        'history_signal', 'classifier', 'features_extraction_method',
        'ranking', 'num_features', 'overlap', 'split_technique', 'interpolate_technique',
        'search_method', 'fillna_method', 'pca_components', 'smoothing_level',
        'apply_weighted_feature', 'feature_weights', 'max_wavelet_scales'
    ]

    # Assign values directly from the dictionary
    (
        output_dir, model_string, windowing, 
        test_train_perc, oversample_undersample, balancing_normal_failed,
        history_signal, classifier, features_extraction_method,
        ranking, num_features, overlap, split_technique,
        fillna_method, pca_components, smoothing_level,
        apply_weighted_feature, feature_weights, max_wavelet_scales
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
        smoothing_level=smoothing_level,
        max_wavelet_scales=max_wavelet_scales
    )

    # Print the line of Xtrain and Xtest
    logger.info(f'Xtrain shape: {Xtrain.shape}, Xtest shape: {Xtest.shape}')

    try:
        data = np.load(os.path.join(output_dir, f'{model_string}_training_and_testing_data_{history_signal}_rank_{ranking}_{num_features}_overlap_{overlap}_features_extraction_method_{features_extraction_method}_oversample_undersample_{oversample_undersample}.npz'))
        Xtrain, Xtest, ytrain, ytest = data['Xtrain'], data['Xtest'], data['Ytrain'], data['Ytest']
    except: # pylint: disable=bare-except
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
            if apply_weighted_feature == True:
                # Apply feature weights to the training and testing data
                Xtrain = apply_feature_weights(Xtrain, feature_weights)
                Xtest = apply_feature_weights(Xtest, feature_weights)
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
    return Xtrain, ytrain, Xtest, ytest

def perform_classification(*args):
    # Define parameter names and create a dictionary of params
    param_names = [
        'Xtrain', 'ytrain', 'Xtest', 'ytest',
        'id_number', 'classifier', 'cuda_dev',
        'search_method', 'enable_tuning', 'incremental_learning',
        'transfer_learning', 'launch_dashboard', 'param_path'
    ]

    # Assign values directly from the dictionary
    (
        Xtrain, ytrain, Xtest, ytest,
        id_number, classifier, CUDA_DEV,
        search_method, enable_tuning, incremental_learning,
        transfer_learning, launch_dashboard, param_path
    ) = dict(zip(param_names, args)).values()

    # Step 1.6: Classifier Selection: set training parameters
    ####### CLASSIFIER PARAMETERS #######
    if CUDA_DEV != 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEV

    classifiers = [
        'FPLSTM',
        'NNet',
        'TCN',
        'DenseNet',
        'MLP_Torch'
    ]

    if classifier in classifiers:
        # Parameters for deep learning networks
        model_path = classification(
            X_train=Xtrain,
            Y_train=ytrain,
            X_test=Xtest,
            Y_test=ytest,
            classifier=classifier,
            metric=['RMSE', 'MAE', 'FDR', 'FAR', 'F1', 'recall', 'precision'],
            id_number=id_number,
            enable_tuning=enable_tuning,
            incremental_learning=incremental_learning,
            transfer_learning=transfer_learning
        )
    else:
        # Parameters for traditional learning
        model_path = classification(
            X_train=Xtrain,
            Y_train=ytrain,
            X_test=Xtest,
            Y_test=ytest,
            classifier=classifier,
            # FDR, FAR, F1, recall, precision are not calculated for some algorithms, it will report as 0.0
            metric=['RMSE', 'MAE', 'FDR', 'FAR', 'F1', 'recall', 'precision'],
            search_method=search_method,
            id_number=id_number,
            launch_dashboard=launch_dashboard
        )

    return logger.get_log_file_path(), model_path, param_path