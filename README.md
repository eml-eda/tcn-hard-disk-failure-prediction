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
  - [Articles](#articles)
  - [TODO](#todo)
  - [Future Work for Algorithm](#future-work-for-algorithm)

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

```shell
tcn-hard-disk-failure-prediction
│
├── algorithms
│   ├── app.py
│   ├── Classification.py
│   ├── Dataset_manipulation.py
│   ├── GeneticFeatureSelector.py
│   ├── json_param.py
│   ├── network_training.py
│   ├── Networks_pytorch.py
│   └── utils.py
│
├── datasets_creation
│   ├── files_to_failed.py
│   ├── find_failed.py
│   ├── get_dataset.py
│   ├── config.py
│   └── toList.py
│
├── inference
│   ├── app.py
│   ├── Dataset_processing.py
│   ├── Inference.py
│   └── Networks_inference.py
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
   python .\algorithms\app.py
   ```

   The script will preprocess the dataset from the `HDD_dataset` directory and save the preprocessed dataset as pkl file in the `output` folder, then it will train and test the classification models on the dataset.

5. Run the inference script:

   ```bash
   python .\inference\app.py
   ```

   The script will preprocess the inference data from uploaded csv file, then it will load the trained model and start the predictions on the parsed data.

## Core Parts of this Algorithm

1. **Feature Selection**: Currently we use the t-test for feature selection. We select the top 18 features based on the t-test scores.
2. **Dataset Unbalancing**: Currently we use SMOTE for upsampling on the failed disk samples to balance the dataset, and use RandomUnderSampler for downsampling the majority class.
3. **Hyperparameter Tuning**: Currently we use sklearn GridSearchCV and RandomSearchCV for hyperparameter tuning, and use 'RMSE', 'MAE', 'FDR', 'FAR', 'F1', 'recall', and 'precision' metrics to evaluate the model. (We use the 'F1' score as the main metric for hyperparameter tuning). For deep learning model, we use the ray tuning library for hyperparameter tuning.
4. **Data Training**: Currently we use RandomForest, TCN, and LSTM for training the data, and use 'RMSE', 'MAE', 'FDR', 'FAR', 'F1', 'recall', and 'precision' metrics to evaluate the model, according to the result, the TCN model performs better than the other models.

## Articles

1. [Predictive models of hard drive failures based on operational data](https://hal.archives-ouvertes.fr/hal-01703140/document), 2018

2. [Hard Drive Failure Prediction Using Classification and Regression Trees](https://www.researchgate.net/publication/286602543_Hard_Drive_Failure_Prediction_Using_Classification_and_Regression_Trees), 2014

3. [Random-forest-based failure prediction for hard disk drives](https://journals.sagepub.com/doi/full/10.1177/1550147718806480), 2018

4. [Proactive Prediction of Hard Disk Drive Failure](http://cs229.stanford.edu/proj2017/final-reports/5242080.pdf), 2017

5. [Hard Drive Failure Prediction for Large Scale Storage System](https://escholarship.org/uc/item/11x380ng), 2017

6. [Improving Storage System Reliability with Proactive Error Prediction](https://www.usenix.org/system/files/conference/atc17/atc17-mahdisoltani.pdf), 2017

7. [Predicting Disk Replacement towards Reliable Data Centers](https://www.kdd.org/kdd2016/papers/files/adf0849-botezatuA.pdf)

8. [Machine Learning Methods for Predicting Failures in Hard Drives: A Multiple-Instance Application](http://jmlr.csail.mit.edu/papers/volume6/murray05a/murray05a.pdf), 2005

9. [Anomaly detection using SMART indicators for hard disk drive failure prediction](https://www.etran.rs/common/pages/proceedings/IcETRAN2017/RTI/IcETRAN2017_paper_RTI1_6.pdf), 2017

10. [Failure Trends in a Large Disk Drive Population](https://static.googleusercontent.com/media/research.google.com/en//archive/disk_failures.pdf), 2007

11. [Improving Service Availability of Cloud Systems by Predicting Disk Error](https://www.usenix.org/system/files/conference/atc18/atc18-xu-yong.pdf), 2018

12. [Proactive error prediction to improve storage system reliability](https://www.usenix.org/system/files/conference/atc17/atc17-mahdisoltani.pdf), 2017

13. [A PROACTIVE DRIVE RELIABILITY MODEL TO PREDICT FAILURES IN THE HARD DISK DRIVES](http://www.iraj.in/journal/journal_file/journal_pdf/3-78-140957031862-68.pdf), 2014

14. [Predicting Hard Disk Failures in Data Centers Using Temporal Convolutional Neural Networks](https://www.researchgate.net/publication/350044713_Predicting_Hard_Disk_Failures_in_Data_Centers_Using_Temporal_Convolutional_Neural_Networks), 2021

15. [Transfer Learning based Failure Prediction for Minority Disks in Large Data Centers of Heterogeneous Disk Systems](https://dl.acm.org/doi/10.1145/3337821.3337881), 2019

## TODO

1. [ ] Thoroughly test the code and fix the bugs.
2. [ ] Add more comments to the code for better understanding, especially the parameters that need to be tuned.
3. [ ] Store the data in PostgreSQL database instead of CSV files.
4. [ ] Refactor the Python scripts in the `algorithm` folder using jupyter notebook, do not directly import the Python scripts in the `algorithm` folder.
5. [ ] Add a visual interface for `dataset_creation.ipynb` for adjusting the parameters.
6. [ ] Add multiple disk support for dataset input.
7. [ ] Add dataset re-train for transfer learning.

## Future Work for Algorithm

1. Use the Genetic Algorithm (provided by DEAP) before the t-test for the statistical significance of the selected features.
2. Add multiple disk models for prediction. (Currently, we only have one disk model, ST4000DM000)
3. Use time-based SMOTE for data augmentation on failed disk samples to balance the dataset.
4. Use transfer learning to improve the model performance.
