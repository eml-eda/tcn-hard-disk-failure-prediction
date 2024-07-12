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
import socket
from explainerdashboard import ClassifierExplainer, ExplainerDashboard


def launch_explainer_dashboard(model, X_test, Y_test, classifier_name, port=8050):
    """
    Launches an ExplainerDashboard for a given trained model.
    Note: The ExplainerDashboard primarily works with scikit-learn models.

    Args:
        model (sklearn.base.BaseEstimator): The trained model.
        X_test (array-like): Test features.
        Y_test (array-like): Test labels.
        classifier_name (str): Name of the classifier to be used as the title.
        port (int): The port on which to run the dashboard.
    """
    # Function to check if a port is available
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    # Find an available port
    while is_port_in_use(port):
        port += 1

    explainer = ClassifierExplainer(model, X_test, Y_test)
    db = ExplainerDashboard(explainer, title=f"{classifier_name} Model Explainer", whatif=False)  # Customize features as needed
    db.run(port=port)
    logger.info(f"Dashboard is running on http://localhost:{port}")

def save_model(model, classifier_name, id_number, n_iterations):
    """
    Saves the trained model to a file.

    Args:
        model (object): The trained model.
        classifier_name (str): The name of the classifier.
        id_number (str): The ID number for the model.
        n_iterations (int): The number of iterations for training.

    Returns:
        str: The path to the saved model file.
    """
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', id_number)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f'{classifier_name}_{id_number}_iterations_{n_iterations}_{now_str}.joblib')
    dump(model, model_path)
    logger.info(f'Model saved as: {model_path}')
    return model_path

def train_and_evaluate_model(
    model,
    param_grid,
    classifier_name,
    X_train,
    Y_train,
    X_test,
    Y_test,
    id_number=1,
    metric=['RMSE', 'MAE', 'FDR', 'FAR', 'F1', 'recall', 'precision'],
    search_method='randomized',
    n_iterations=100,
    launch_dashboard=False
):
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
    supervised_classifiers = [
        'RandomForest',
        'KNeighbors',
        'DecisionTree',
        'LogisticRegression',
        'SVM',
        'XGB',
        'MLP',
        'ExtraTrees',
        'GradientBoosting',
        'NaiveBayes',
        'RGF'
    ]

    # Define scoring metrics based on the type of classifier
    if classifier_name in supervised_classifiers:
        scoring = {'accuracy': make_scorer(accuracy_score), 'f1': make_scorer(f1_score)}
    else:
        scoring = {'silhouette': make_scorer(silhouette_score)}

    search = None
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

    model_path = save_model(best_model, classifier_name, id_number, n_iterations)

    if launch_dashboard:
        launch_explainer_dashboard(best_model, X_test, Y_test, classifier_name)

    return model_path

def train_dl_model(
    config,
    data,
    enable_tuning=True,
    incremental_learning=False,
    transfer_learning=False,
    classifier='FPLSTM',
    id_number=1
):
    """
    Train a deep learning model.

    Args:
        config (dict): Configuration parameters for training the model.
        data (tuple): Tuple containing the training and testing data.
        enable_tuning (bool, optional): Flag to enable tuning. Defaults to True.
        incremental_learning (bool, optional): Flag to enable incremental learning. Defaults to False.
        transfer_learning (bool, optional): Flag to enable transfer learning. Defaults to False.
        classifier (str, optional): Type of classifier to use. Defaults to 'FPLSTM'.
        id_number (int, optional): ID number for the model. Defaults to 1.

    Returns:
        None
    """
    Xtrain, ytrain, Xtest, ytest = data

    # Set training parameters
    lr = config['lr']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    epochs = config['epochs']
    dropout = config['dropout']
    optimizer_type = config['optimizer_type']
    reg = config['reg']
    num_workers = config['num_workers']
    scheduler_type = config['scheduler_type']
    loss_function = config['loss_function']

    if config['scheduler_type'] == 'StepLR':
        scheduler_step_size = config['scheduler_step_size']
        scheduler_gamma = config['scheduler_gamma']
    elif config['scheduler_type'] == 'ExponentialLR':
        scheduler_gamma = config['scheduler_gamma']
    elif config['scheduler_type'] == 'ReduceLROnPlateau':
        scheduler_factor = config['scheduler_factor']
        scheduler_patience = config['scheduler_patience']

    num_inputs = Xtrain.shape[1]
    data_dim = Xtrain.shape[2] if classifier not in ['FPLSTM', 'DenseNet'] else None
    hidden_dim = config.get('hidden_dim')
    num_layers = config.get('num_layers')
    lstm_hidden_s = config.get('lstm_hidden_s')
    fc1_hidden_s = config.get('fc1_hidden_s')
    hidden_size = config.get('hidden_size')

    if classifier == 'FPLSTM':
        net = FPLSTM(lstm_hidden_s, fc1_hidden_s, num_inputs, 2, dropout)
    elif classifier == 'NNet':
        net = NNet(input_size=data_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    elif classifier == 'TCN':
        net = TCN_Network(data_dim, num_inputs)
    elif classifier == 'DenseNet':
        net = DenseNet(input_size=num_inputs, hidden_size=hidden_size)
    elif classifier == 'MLP':
        input_dim = Xtrain.shape[1] * Xtrain.shape[2]
        net = MLP(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError('Invalid classifier type. Please choose a valid classifier.')

    if incremental_learning:
        net.load_state_dict(torch.load(f'{classifier.lower()}_{id_number}_epochs_{epochs}_batchsize_{batch_size}_lr_{lr}_*.pth'))
        if transfer_learning:
            for name, param in net.named_parameters():
                if classifier == 'FPLSTM' and ('lstm' in name or 'do1' in name):
                    param.requires_grad = False
                elif classifier == 'NNet' and 'rnn' in name:
                    param.requires_grad = False
                elif classifier == 'TCN' and ('b0_' in name or 'b1_' in name or 'b2_' in name):
                    param.requires_grad = False
                elif classifier == 'DenseNet' and ('layers.0' in name or 'layers.2' in name):
                    param.requires_grad = False
                elif classifier == 'MLP' and ('lin1' in name or 'lin2' in name):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Moving model to {device}')
    net.to(device)

    optimizer_cls = optim.Adam if optimizer_type == 'Adam' else optim.SGD
    optimizer = optimizer_cls(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)

    trainer = UnifiedTrainer(
        model=net,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        reg=reg,
        id_number=id_number,
        model_type=classifier,
        num_workers=num_workers,
        scheduler_type=scheduler_type,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        loss_function=loss_function
    )

    trainer.run(Xtrain, ytrain, Xtest, ytest)

    if enable_tuning:
        tune.report(accuracy=trainer.test_accuracy)
