import gradio as gr
from Classification import initialize_classification


if __name__ == '__main__':
    iface = gr.Interface(
        fn=initialize_classification,
        inputs=[
            gr.Textbox(value='ST3000DM001', label='Model', info='Enter the model type(s). For multiple models, separate them with commas.'),
            gr.Textbox(value='01234567', label='ID Number', info='Enter the ID number(s).'),
            gr.CheckboxGroup(choices=['2013', '2014', '2015', '2016', '2017', '2018', '2019'], value=['2013'], label='Years', info='Select the years to consider.'),
            gr.Dropdown(choices=[0, 1], value=1, label='Windowing', info='Select windowing technique.'),
            gr.Slider(minimum=1, maximum=365, step=1, value=115, label='Min Days HDD', info='Minimum number of days for HDD.'),
            gr.Slider(minimum=1, maximum=30, step=1, value=7, label='Days Considered as Failure', info='Number of days considered as failure.'),
            gr.Slider(minimum=0, maximum=1, step=0.1, value=0.3, label='Test Train Percentage', info='Percentage for test/train split.'),
            gr.Dropdown(choices=[0, 1, 2], value=0, label='Oversample Undersample', info='Select oversample/undersample technique.'),
            gr.Textbox(value='auto', label='Balancing Normal Failed', info='Balancing factor for normal and failed states, input auto for automatic.'),
            gr.Slider(minimum=1, maximum=100, step=1, value=32, label='History Signal', info='Length of the history signal.'),
            gr.Dropdown(choices=['TCN', 'LSTM', 'MLP', 'RandomForest', 'KNeighbors', 'DecisionTree', 'LogisticRegression', 'SVM', 'MLP_Manual', 'XGB', 'IsolationForest', 'ExtraTrees', 'GradientBoosting', 'NaiveBayes', 'DBSCAN'], value='TCN', label='Classifier', info='Select the classifier type.'),
            gr.Checkbox(value=False, label='Perform Features Extraction', info='Check to perform feature extraction.'),
            gr.Dropdown(choices=['0', '1', ''], value='0', label='CUDA DEV', info='Select CUDA device.'),
            gr.Dropdown(choices=['Ok', 'None'], value='Ok', label='Ranking', info='Select ranking method.'),
            gr.Slider(minimum=1, maximum=50, step=1, value=18, label='Num Features', info='Number of features to use.'),
            gr.Dropdown(choices=[0, 1, 2], value=1, label='Overlap', info='Select Overlap technique.'),
            gr.Dropdown(choices=['random', 'hdd', 'date'], value='random', label='Split Technique', info='Data split technique.'),
            gr.Dropdown(choices=['linear', 'time', 'None'], value='linear', label='Interpolate Technique', info='Interpolation technique.'),
            gr.Dropdown(choices=['randomized', 'grid', 'None'], value='randomized', label='Search Technique', info='Optimal parameters search technique.'),
            gr.Dropdown(choices=['ffill', 'None'], value='None', label='Fill NA Method', info='Method to fill NA values.'),
        ],
        outputs=[
            gr.File(label='Download Log File', type='filepath'),
            gr.File(label='Download Model File', type='filepath'),
            gr.File(label='Download Parameter File', type='filepath'),
        ],
        title="Prognostika - Hard Disk Failure Prediction Model Training Dashboard",  # Title of the interface
        description="Predicting System Failures using Machine Learning Techniques",  # Description of the interface
    )

    iface.launch()