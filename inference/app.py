import gradio as gr
from Inference import initialize_inference


if __name__ == '__main__':
    iface = gr.Interface(
        fn=initialize_inference,
        inputs=[
            gr.Textbox(value='01234567', label='ID Number', info='Enter the ID number(s).'),
            gr.Dropdown(choices=['TCN', 'LSTM', 'MLP', 'RandomForest', 'KNeighbors', 'DecisionTree', 'LogisticRegression', 'SVM', 'MLP_Manual', 'XGB', 'IsolationForest', 'ExtraTrees', 'GradientBoosting', 'NaiveBayes'], value='TCN', label='Classifier', info='Select the classifier type.'),
            gr.Dropdown(choices=['0', '1', ''], value='0', label='CUDA DEV', info='Select CUDA device.'),
            gr.File(label="Upload CSV"),  # File upload option
        ],
        outputs=gr.Textbox(),
        title="Prognostika - Hard Disk Failure Prediction Model Inference Dashboard",  # Title of the interface
        description="Predicting System Failures using Machine Learning Techniques",  # Description of the interface
    )

    iface.launch()