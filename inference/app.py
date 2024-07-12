import gradio as gr
import argparse
from Inference import initialize_inference, set_inference_params
import sys

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--id_number', default='01234567')
parser.add_argument('--classifier', default='TCN')
parser.add_argument('--features_extraction_method', default='None')
parser.add_argument('--cuda_dev', default='0')
parser.add_argument('--file_path', default='data.csv')
# Add more arguments as needed
args = parser.parse_args()

main_iface = gr.Interface(
    fn=initialize_inference,
    inputs=[
        gr.Textbox(value='01234567', label='ID Number', info='Enter the ID number(s).'),
        gr.Textbox(value='', label='Model', info='Enter the model type(s) for infering.'),
        gr.Dropdown(choices=['TCN', 'LSTM', 'MLP', 'RandomForest', 'KNeighbors', 'DecisionTree', 'LogisticRegression', 'SVM', 'MLP_Torch', 'XGB', 'IsolationForest', 'ExtraTrees', 'GradientBoosting', 'NaiveBayes'], value='TCN', label='Classifier', info='Select the classifier type.'),
        gr.Dropdown(choices=['custom', 'PCA', 'None'], value='None', label='Features Extraction Method', info='Select the features extraction method.'),
        gr.Dropdown(choices=['0', '1', ''], value='0', label='CUDA DEV', info='Select CUDA device.'),
        gr.Slider(minimum=1, maximum=8, step=1, value=8, label='PCA Components', info='Select the number of PCA components to generate.'),
        gr.Slider(minimum=0, maximum=1, step=0.1, value=0.3, label='Smoothing Level', info='Select the smoothing level.'),
        gr.Slider(choices=['duplicate', 'interpolate'], value='interpolate', label='Augmentation Method', info='Select the augmentation method.'),
        gr.File(label="Upload CSV"),  # File upload option
    ],
    outputs=gr.Textbox(),
    title="Prognostika - Hard Disk Failure Prediction Model Inference Dashboard",  # Title of the interface
    description="Predicting System Failures using Machine Learning Techniques",  # Description of the interface
)

inference_param_iface = gr.Interface(
    fn=set_inference_params,
    inputs=[
        gr.Slider(minimum=0.05, maximum=0.5, step=0.01, value=0.1, label='LSTM & NNet Dropout', info='LSTM & NNet dropout rate for training.'),
        gr.Slider(minimum=16, maximum=128, step=1, value=64, label='LSTM Hidden Dimension', info='LSTM hidden dimension for training.'),
        gr.Slider(minimum=16, maximum=128, step=1, value=16, label='LSTM FC Dimension', info='LSTM fully connected dimension for training.'),
        gr.Slider(minimum=16, maximum=128, step=1, value=128, label='MLP & NNet Hidden Dimension', info='MLP & NNet hidden dimension for training.'),
        gr.Slider(minimum=1, maximum=32, step=1, value=8, label='DenseNet Hidden Dimension', info='DenseNet hidden dimension for training. (x*x)'),
        gr.Slider(minimum=1, maximum=4, step=1, value=1, label='NNet Number of Layers', info='NNet number of layers for training.'),
    ],
    outputs=gr.Textbox(placeholder="See updated parameters below.", label="Updated Parameters"),
    description="Training Parameters for Predicting System Failures using Machine Learning Techniques",  # Description of the interface
)

iface = gr.TabbedInterface(
    [main_iface, inference_param_iface],
    ["Main Params", "Inference Params"],
    title="Prognostika - Hard Disk Failure Prediction Model Inference Dashboard",  # Title of the interface
)

if __name__ == '__main__':
    # Check if any arguments were passed from the command line
    if len(sys.argv) > 1:
        # Convert args to a dictionary and get the values
        args_values = list(vars(args).values())

        # Call initialize_classification with the unpacked list of values
        initialize_inference(*args_values)
    else:
        iface.launch()