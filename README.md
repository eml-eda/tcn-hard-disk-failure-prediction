# tcn-hard-disk-failure-prediction

## Table of Contents

- [tcn-hard-disk-failure-prediction](#tcn-hard-disk-failure-prediction)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Code Structure](#code-structure)
  - [Description of each file](#description-of-each-file)
  - [Wiki](#wiki)
  - [How to run the code](#how-to-run-the-code)
  - [Core Parts of this Algorithm](#core-parts-of-this-algorithm)
  - [Future Work](#future-work)

## Introduction

This repository contains the code to reproduce the experiments of the paper "Predicting Hard Disk Failures in Data Centers using Temporal Convolutional Neural Networks", Burrello et al., Euro-Par 2020.
Please cite the paper as:

@InProceedings{10.1007/978-3-030-71593-9_22,

author="Burrello, Alessio

and Pagliari, Daniele Jahier

and Bartolini, Andrea

and Benini, Luca

and Macii, Enrico

and Poncino, Massimo",

title="Predicting Hard Disk Failures in Data Centers Using Temporal Convolutional Neural Networks",

booktitle="Euro-Par 2020: Parallel Processing Workshops",

year="2021",

publisher="Springer International Publishing",

address="Cham",

pages="277--289",

isbn="978-3-030-71593-9"

}

## Code Structure

The code is structured as follows:
```
tcn-hard-disk-failure-prediction
│
├── algorithms
│   ├── Classification.py
│   ├── Dataset_manipulation.py
│   ├── Networks_pytorch.py
│   └── README.txt
│
├── datasets_creation
│   ├── files_to_failed.py
│   ├── find_failed.py
│   ├── get_dataset.py
│   ├── README.txt
│   └── toList.py
│
├── .gitignore
└── README.md
```

## Description of each file

- `Classification.py`: This script contains the code to train and test various classification models including RandomForest, TCN, and LSTM.

- `Dataset_manipulation.py`: This script is used for manipulating the dataset. It could include tasks such as cleaning, preprocessing, feature extraction, etc.

- `Networks_pytorch.py`: This script contains the implementation of the TCN and LSTM networks using PyTorch.

- `README.txt`: This file provides an overview of the algorithms directory.

- `files_to_failed.py`: This script is used to convert files to a failed state, as part of the dataset creation process.

- `find_failed.py`: This script is used to find and mark failed instances in the dataset.

- `get_dataset.py`: This script is used to fetch and possibly preprocess the dataset for the hard disk failure prediction task.

- `README.txt`: This file provides an overview of the datasets_creation directory.

- `toList.py`: This script is used to convert certain data structures to a list format, possibly for easier manipulation or usage in the project.

## Wiki

For more information, please refer to the [wiki](https://github.com/Disk-Failure-Prediction/tcn-hard-disk-failure-prediction/wiki).

## How to run the code

1. Clone the repository:

   ```bash
   git clone git@github.com:Prognostika/tcn-hard-disk-failure-prediction.git
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset via the `get_dataset.py` script:

   ```bash
   python .\datasets_creation\get_dataset.py
   ```

   After running the script, the dataset will be downloaded and saved in the `HDD_dataset` directory in the parent folder of the repository. (The total dataset and zip package will be around 50 GB, make sure you have enough space on your disk.)

4. Run the classification script:

   ```bash
   python .\algorithms\Classification.py
   ```

   The script will preprocess the dataset and save the preprocessed dataset in the `HDD_dataset` directory, then it will train and test the classification models on the dataset.

## Core Parts of this Algorithm

1. **Feature Selection**: Currently we use the t-test for feature selection. We select the top 18 features based on the t-test scores.
2. **Dataset Unbalancing**: Currently we use SMOTE for data augmentation on the failed disk samples to balance the dataset, and use RandomUnderSampler for the majority class.
3. **Data Training**: Currently we use RandomForest, TCN, and LSTM for training the data, and use 'RMSE', 'MAE', 'FDR', 'FAR', 'F1', 'recall', and 'precision' metrics to evaluate the model, according to the result, the TCN model performs better than the other models.

## Future Work

1. Use the Genetic Algorithm (provided by DEAP) before the t-test for the statistical significance of the selected features.
2. Add multiple disk models for prediction. (Currently, we only have one disk model, ST4000DM000)
3. Use time-based SMOTE for data augmentation on failed disk samples to balance the dataset.
4. Use transfer learning to improve the model performance.
