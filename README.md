# tcn-hard-disk-failure-prediction

## Table of Contents

- [tcn-hard-disk-failure-prediction](#tcn-hard-disk-failure-prediction)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Code Structure](#code-structure)
  - [Description of each file](#description-of-each-file)
  - [Wiki](#wiki)

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
