import gradio as gr
from Classification import initialize_classification, set_training_params
import argparse
import sys

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ST3000DM001')
parser.add_argument('--id_number', default='01234567')
parser.add_argument('--years', nargs='*', default=['2013'])
parser.add_argument('--statistical_tests', default='t-test')
parser.add_argument('--windowing', default=1)
parser.add_argument('--min_days_hdd', default=115)
parser.add_argument('--days_considered_as_failure', default=7)
parser.add_argument('--test_train_percentage', default=0.3)
parser.add_argument('--oversample_undersample', default=1)
parser.add_argument('--balancing_normal_failed', default='auto')
parser.add_argument('--history_signal', default=32)
parser.add_argument('--classifier', default='TCN')
parser.add_argument('--features_extraction_method', default='None')
parser.add_argument('--cuda_dev', default='0')
parser.add_argument('--ranking', default='Ok')
parser.add_argument('--num_features', default=18)
parser.add_argument('--overlap', default=1)
parser.add_argument('--split_technique', default='random')
parser.add_argument('--interpolate_technique', default='linear')
parser.add_argument('--search_technique', default='randomized')
parser.add_argument('--fill_na_method', default='None')
parser.add_argument('--pca_components', default=8)
parser.add_argument('--smoothing_level', default=0.3)
parser.add_argument('--transfer_learning', default=False)
parser.add_argument('--train_on_all_models', default=False)
# Add more arguments as needed
args = parser.parse_args()

# Add tabbed interface for ML training parameters
main_iface = gr.Interface(
    fn=initialize_classification,
    inputs=[
        gr.Dropdown(choices=['ST', 'WDC', 'Hitachi', 'custom'], value='custom', label='Manufacturer', info='Select the manufacturer of the hard disk, custom for custom model.'),
        gr.Textbox(value='ST3000DM001', label='Model', info='Enter the model type(s) for training. For multiple models, separate them with commas.'),
        gr.Textbox(value='01234567', label='ID Number', info='Enter the ID number(s).'),
        gr.CheckboxGroup(choices=['2013', '2014', '2015', '2016', '2017', '2018', '2019'], value=['2013'], label='Years', info='Select the years to consider.'),
        gr.Dropdown(choices=['t-test', 'mannwhitneyu'], value='t-test', label='Statistical Tests', info='Select the statistical tests to extract features.'),
        gr.Dropdown(choices=[0, 1], value=1, label='Windowing', info='Select windowing technique.'),
        gr.Slider(minimum=1, maximum=365, step=1, value=115, label='Min Days HDD', info='Minimum number of days for HDD.'),
        gr.Slider(minimum=1, maximum=30, step=1, value=7, label='Days Considered as Failure', info='Number of days considered as failure.'),
        gr.Slider(minimum=0, maximum=1, step=0.1, value=0.3, label='Test Train Proportion', info='Proportion for test/train split.'),
        gr.Dropdown(choices=['None', 'Yes'], value='None', label='Oversample Undersample', info='Select oversample/undersample technique.'),
        gr.Textbox(value='auto', label='Balancing Normal Failed', info='Balancing factor for normal and failed states, input auto for automatic.'),
        gr.Slider(minimum=1, maximum=100, step=1, value=32, label='History Signal', info='Length of the history signal.'),
        gr.Dropdown(choices=['TCN', 'LSTM', 'NNet', 'DenseNet', 'MLP', 'RandomForest', 'KNeighbors', 'DecisionTree', 'LogisticRegression', 'SVM', 'MLP_Torch', 'XGB', 'IsolationForest', 'ExtraTrees', 'GradientBoosting', 'NaiveBayes', 'DBSCAN', 'RGF'], value='TCN', label='Classifier', info='Select the classifier algorithm.'),
        gr.Dropdown(choices=['custom', 'PCA', 'None'], value='None', label='Features Extraction Method', info='Select the features extraction method.'),
        gr.Dropdown(choices=['0', '1', 'None'], value='0', label='CUDA DEV', info='Select CUDA device, None for CPU.'),
        gr.Dropdown(choices=['Ok', 'None'], value='Ok', label='Ranking', info='Select ranking method.'),
        gr.Slider(minimum=1, maximum=50, step=1, value=18, label='Num Features', info='Number of features to use.'),
        gr.Dropdown(choices=[0, 1, 2], value=1, label='Overlap', info='Select Overlap technique.'),
        gr.Dropdown(choices=['random', 'hdd', 'date'], value='random', label='Split Technique', info='Select the data split technique.'),
        gr.Dropdown(choices=['linear', 'time', 'None'], value='linear', label='Interpolate Technique', info='Select the interpolation technique.'),
        gr.Dropdown(choices=['randomized', 'grid', 'None'], value='randomized', label='Search Technique', info='Select the search technique.'),
        gr.Dropdown(choices=['ffill', 'None'], value='None', label='Fill NA Method', info='Select the method to fill NA values.'),
        gr.Slider(minimum=1, maximum=8, step=1, value=8, label='PCA Components', info='Select the number of PCA components to generate.'),
        gr.Slider(minimum=0, maximum=1, step=0.1, value=0.3, label='Smoothing Level', info='Select the smoothing level.'),
        gr.Checkbox(value=False, label='Enable Transfer Learning', info='Check to enable transfer learning.'),
        gr.Checkbox(value=False, label='Train on All Models', info='Check to train on all models.'),
        gr.Checkbox(value=False, label='Enable Genetic Algorithm', info='Enable Genetic Algorithm for Primary Feature Selection.'),
        gr.Slider(minimum=5, maximum=50, step=1, value=10, label='Population Number', info='Number of individuals in each generation.'),
        gr.Slider(minimum=1, maximum=10, step=1, value=2, label='Stop Criteria', info='Stop the genetic algorithm after certain generations.'),
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
        gr.Dropdown(choices=['Adam', 'SGD'], value='Adam', label='Optimizer', info='Select the optimizer for training.'),
        gr.Slider(minimum=1, maximum=16, step=1, value=8, label='Number of Workers for DataLoader', info='Number of workers for DataLoader.'),
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