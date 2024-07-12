import gradio as gr
from Classification import initialize_classification, set_training_params
import argparse
import sys

# Parse command-line arguments, default type as int
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ST3000DM001', help='Enter the model type(s) for training. For multiple models, separate them with commas.')
parser.add_argument('--id_number', default='01234567', help='Enter the ID number(s).')
parser.add_argument('--years', nargs='*', default=['2013'], help='Select the years to consider.')
parser.add_argument('--statistical_tests', default='t-test', help='Select the statistical tests to extract features.')
parser.add_argument('--windowing', default=1, type=int, help='Select windowing technique.')
parser.add_argument('--min_days_hdd', default=115, type=int, help='Minimum number of days for HDD.')
parser.add_argument('--days_considered_as_failure', default=7, type=int, help='Number of days considered as failure.')
parser.add_argument('--test_train_percentage', default=0.3, type=int, help='Proportion for test/train split.')
parser.add_argument('--oversample_undersample', default=1, type=int, help='Select oversample/undersample technique.')
parser.add_argument('--balancing_normal_failed', default='auto', help='Balancing factor for normal and failed states, input auto for automatic.')
parser.add_argument('--history_signal', default=32, type=int, help='Length of the history signal.')
parser.add_argument('--classifier', default='TCN', help='Select the classifier algorithm.')
parser.add_argument('--features_extraction_method', default='None', help='Select the features extraction method.')
parser.add_argument('--cuda_dev', default='0', help='Select CUDA device, None for CPU.')
parser.add_argument('--ranking', default='Ok', help='Select ranking method.')
parser.add_argument('--num_features', default=18, help='Number of features to use.')
parser.add_argument('--overlap', default=1, type=int, help='Select Overlap technique.')
parser.add_argument('--split_technique', default='random', help='Select the data split technique.')
parser.add_argument('--interpolate_technique', default='linear', help='Select the interpolation technique.')
parser.add_argument('--search_technique', default='randomized', help='Select the search technique.')
parser.add_argument('--fill_na_method', default='None', help='Select the method to fill NA values.')
parser.add_argument('--pca_components', default=8, type=int, help='Select the number of PCA components to generate.')
parser.add_argument('--smoothing_level', default=0.3, type=int, help='Select the smoothing level.')
parser.add_argument('--incremental_learning', default=False, help='Check to enable incremental learning.')
parser.add_argument('--transfer_learning', default=False, help='Check to enable transfer learning.')
parser.add_argument('--partion_model', default=False, help='Check to partition model and enable initial training for the relevant models and transfer learning for irrelevant models.')
parser.add_argument('--hyperparameter_tuning', default=False, help='Enable hyperparameter tuning for the model.')
parser.add_argument('--genetic_algorithm', default=False, help='Enable Genetic Algorithm for Primary Feature Selection.')
parser.add_argument('--population_number', default=10, type=int, help='Number of individuals in each generation.')
parser.add_argument('--stop_criteria', default=2, type=int, help='Stop the genetic algorithm after certain generations.')
parser.add_argument('--weighted_feature_training', default=False, help='Check to enable weighted feature training.')
parser.add_argument('--max_wavelet_scales', default=50, type=int, help='Maximum number of wavelet scales to consider.')
parser.add_argument('--launch_dashboard', default=True, help='Check to launch the dashboard for training, currently only support sklearn API.')
# Add more arguments as needed
args = parser.parse_args()

# Add tabbed interface for ML training parameters
main_iface = gr.Interface(
    fn=initialize_classification,
    inputs=[
        gr.Radio(choices=['ST', 'WDC', 'Hitachi', 'custom'], value='custom', label='Manufacturer', info='Select the manufacturer of the hard disk, custom for custom model.'),
        gr.Textbox(value='ST3000DM001', label='Model', info='Enter the model type(s) for training. For multiple models, separate them with commas.'),
        gr.Textbox(value='01234567', label='ID Number', info='Enter the ID number(s).'),
        gr.CheckboxGroup(choices=['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'], value=['2013'], label='Years', info='Select the years to consider.'),
        gr.Radio(choices=['t-test', 'mannwhitneyu'], value='t-test', label='Statistical Tests', info='Select the statistical tests to extract features.'),
        gr.Radio(choices=[0, 1], value=1, label='Windowing', info='Select windowing technique.'),
        gr.Slider(minimum=1, maximum=365, step=1, value=115, label='Min Days HDD', info='Minimum number of days for HDD.'),
        gr.Slider(minimum=1, maximum=30, step=1, value=7, label='Days Considered as Failure', info='Number of days considered as failure.'),
        gr.Slider(minimum=0, maximum=1, step=0.1, value=0.3, label='Test Train Proportion', info='Proportion for test/train split.'),
        gr.Radio(choices=['None', 'Yes'], value='None', label='Oversample Undersample', info='Select oversample/undersample technique.'),
        gr.Textbox(value='auto', label='Balancing Normal Failed', info='Balancing factor for normal and failed states, input auto for automatic.'),
        gr.Slider(minimum=1, maximum=100, step=1, value=32, label='History Signal', info='Length of the history signal.'),
        gr.Radio(choices=['TCN', 'LSTM', 'NNet', 'DenseNet', 'MLP', 'RandomForest', 'KNeighbors', 'DecisionTree', 'LogisticRegression', 'SVM', 'MLP_Torch', 'XGB', 'IsolationForest', 'ExtraTrees', 'GradientBoosting', 'NaiveBayes', 'DBSCAN', 'RGF'], value='TCN', label='Classifier', info='Select the classifier algorithm.'),
        gr.Radio(choices=['custom', 'PCA', 'None'], value='None', label='Features Extraction Method', info='Select the features extraction method.'),
        gr.Radio(choices=['0', '1', 'None'], value='0', label='CUDA DEV', info='Select CUDA device, None for CPU.'),
        gr.Radio(choices=['Ok', 'None'], value='Ok', label='Ranking', info='Select ranking method.'),
        gr.Slider(minimum=1, maximum=50, step=1, value=18, label='Num Features', info='Number of features to use.'),
        gr.Radio(choices=[0, 1, 2], value=1, label='Overlap', info='Select Overlap technique.'),
        gr.Radio(choices=['random', 'hdd', 'date'], value='random', label='Split Technique', info='Select the data split technique.'),
        gr.Radio(choices=['linear', 'time', 'None'], value='linear', label='Interpolate Technique', info='Select the interpolation technique.'),
        gr.Radio(choices=['randomized', 'grid', 'None'], value='randomized', label='Search Technique', info='Select the search technique.'),
        gr.Radio(choices=['ffill', 'None'], value='None', label='Fill NA Method', info='Select the method to fill NA values.'),
        gr.Slider(minimum=1, maximum=8, step=1, value=8, label='PCA Components', info='Select the number of PCA components to generate.'),
        gr.Slider(minimum=0, maximum=1, step=0.1, value=0.3, label='Smoothing Level', info='Select the smoothing level.'),
        gr.Checkbox(value=False, label='Enable Incremental Learning', info='Check to enable incremental learning.'),
        gr.Checkbox(value=False, label='Enable Transfer Learning', info='Check to enable transfer learning.'),
        gr.Checkbox(value=False, label='Enable Partition Model', info='Check to partition model and enable initial training for the relevant models and transfer learning for irrelevant models'),
        gr.Checkbox(value=False, label='Enable Hyperparameter Tuning', info='Enable hyperparameter tuning for the model.'),
        gr.Checkbox(value=False, label='Enable Genetic Algorithm', info='Enable Genetic Algorithm for Primary Feature Selection.'),
        gr.Slider(minimum=5, maximum=50, step=1, value=10, label='Population Number', info='Number of individuals in each generation.'),
        gr.Slider(minimum=1, maximum=10, step=1, value=2, label='Stop Criteria', info='Stop the genetic algorithm after certain generations.'),
        gr.Checkbox(value=False, label='Enable Weighted Feature Training', info='Check to enable weighted feature training.'),
        gr.Slider(minimum=10, maximum=200, step=5, value=50, label='Max Wavelet Scales', info='Maximum number of wavelet scales to consider.'),
        gr.Checkbox(value=True, label='Launch the Dashboard for Training', info='Check to launch the dashboard for training, currently only support sklearn API'),
    ],
    outputs=[
        gr.File(label='Download Log File', type='filepath'),
        gr.File(label='Download Model File', type='filepath'),
        gr.File(label='Download Parameter File', type='filepath'),
    ],
    description="Main Parameters for Predicting System Failures using Machine Learning Techniques",  # Description of the interface
)

training_param_iface = gr.Interface(
    fn=set_training_params,
    inputs=[
        gr.Slider(minimum=0, maximum=1, step=0.1, value=0.1, label='Regularization', info='Regularization factor for training. (Set 1 to disable the penalty)'),
        gr.Slider(minimum=64, maximum=512, step=1, value=256, label='Batch Size', info='Size of the batch for training.'),
        gr.Slider(minimum=0.001, maximum=0.010, step=0.001, value=0.001, label='Learning Rate', info='Initial learning rate for training.'),
        gr.Slider(minimum=0.000, maximum=0.010, step=0.001, value=0.005, label='L2 Weight Decay', info='L2 weight decay for training.'),
        gr.Slider(minimum=50, maximum=500, step=1, value=200, label='Epochs', info='Number of epochs for training.'),
        gr.Slider(minimum=0.05, maximum=0.5, step=0.01, value=0.1, label='LSTM & NNet Dropout', info='LSTM & NNet dropout rate for training.'),
        gr.Slider(minimum=16, maximum=128, step=1, value=64, label='LSTM Hidden Dimension', info='LSTM hidden dimension for training.'),
        gr.Slider(minimum=16, maximum=128, step=1, value=16, label='LSTM FC Dimension', info='LSTM fully connected dimension for training.'),
        gr.Slider(minimum=16, maximum=128, step=1, value=128, label='MLP & NNet Hidden Dimension', info='MLP & NNet hidden dimension for training.'),
        gr.Slider(minimum=1, maximum=32, step=1, value=8, label='DenseNet Hidden Dimension', info='DenseNet hidden dimension for training. (x*x)'),
        gr.Slider(minimum=1, maximum=4, step=1, value=1, label='NNet Number of Layers', info='NNet number of layers for training.'),
        gr.Radio(choices=['Adam', 'SGD'], value='Adam', label='Optimizer', info='Select the optimizer for training.'),
        gr.Slider(minimum=1, maximum=16, step=1, value=8, label='Number of Workers for DataLoader', info='Number of workers for DataLoader.'),
        gr.Radio(choices=['ReduceLROnPlateau', 'ExponentialLR', 'StepLR'], value='ReduceLROnPlateau', label='Scheduler Type', info='Select the scheduler for training.'),
        gr.Slider(minimum=0.1, maximum=0.5, step=0.1, value=0.1, label='Scheduler Factor', info='Select the scheduler factor for training, this is the factor by which the learning rate will be reduced.'),
        gr.Slider(minimum=5, maximum=50, step=1, value=10, label='Scheduler Patience', info='Select the scheduler patience for training, this is the number of epochs with no improvement after which learning rate will be reduced.'),
        gr.Slider(minimum=10, maximum=50, step=1, value=30, label='Scheduler Step Size', info='Select the scheduler step size for training, this is the number of epochs after which the learning rate is multiplied by gamma.'),
        gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label='Scheduler Gamma', info='Select the scheduler gamma for training, this is the factor by which the learning rate is multiplied after each step_size epochs.'),
        gr.Radio(choices=['CrossEntropy', 'BCEWithLogits'], value='CrossEntropy', label='Loss Function', info='Select the loss function for training.'),
    ],
    outputs=gr.Textbox(placeholder="See updated parameters below.", label="Updated Parameters"),
    description="Training Parameters for Predicting System Failures using Machine Learning Techniques",  # Description of the interface
)

iface = gr.TabbedInterface(
    [main_iface, training_param_iface],
    ["Main Params", "Training Params"],
    title="Prognostika - Hard Disk Failure Prediction Model Training Dashboard",  # Title of the interface
)

if __name__ == '__main__':
    # Check if any arguments were passed from the command line
    if len(sys.argv) > 1:
        # Convert args to a dictionary and get the values
        args_values = list(vars(args).values())

        # Call initialize_classification with the unpacked list of values
        initialize_classification(*args_values)
    else:
        iface.launch()