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
  - [Contact](#contact)

## Introduction

**A comprehensive machine learning project for predicting whether a hard disk will fail within a given time interval**

More than [2500 petabytes](https://www.domo.com/learn/data-never-sleeps-5?aid=ogsm072517_1&sf100871281=1) of data are generated every day by sources such as social media, IoT devices, etc., and every bit of it is valuable. That’s why modern storage systems need to be reliable, scalable, and efficient. To ensure that data is not lost or corrupted, many large-scale distributed storage systems, such as [Ceph](https://ceph.io) or [AWS](https://aws.amazon.com), use erasure-coded redundancy or mirroring. Although this provides reasonable fault tolerance, it can make it more difficult and expensive to scale up the storage cluster.

This project seeks to mitigate this problem using machine learning. Specifically, the goal of this project is to train a model that can predict if a given disk will fail within a predefined future time window. These predictions can then be used by Ceph (or other similar systems) to determine when to add or remove data replicas. In this way, the fault tolerance can be improved by up to an order of magnitude, since the probability of data loss is generally related to the probability of multiple, concurrent disk failures.

In addition to creating models, we also aim to catalyze community involvement in this domain by providing Jupyter notebooks to easily get started with and explore some publicly available datasets such as Backblaze Dataset and Ceph Telemetry. Ultimately, we want to provide a platform where data scientists and subject matter experts can collaborate and contribute to this ubiquitous problem of predicting disk failures.

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

- Folder: **algorithms**
  - `app.py`: This script is used to display the gradio interface and run the classification functions.
  - `Classification.py`: This script contains the code to train and test various classification models including RandomForest, TCN, and LSTM.
  - `Dataset_manipulation.py`: This script is used for manipulating the dataset. It could include tasks such as cleaning, preprocessing, feature extraction, etc.
  - `GeneticFeatureSelector.py`: This script is used to select the best features using the genetic algorithm.
  - `json_param.py`: This script is used to load the parameters from the JSON file or save the to the JSON file.
  - `logger.py`: This script is used to log the information during the training and testing process.
  - `network_training.py`: This script is used to train the deep learning networks or the classification models.
  - `Networks_pytorch.py`: This script contains the implementation of the deep learning networks using PyTorch.
  - `utils.py`: This script contains the utility functions used in the training and testing process.
- Folder: **datasets_creation**
  - `app.py`: This script is used to display the gradio interface and run the dataset creation functions.
  - `get_dataset.py`: This script is used to fetch and possibly preprocess the dataset for the hard disk failure prediction task.
  - `save_to_grouped_list.py`: This script is used to convert certain data structures to a list format, possibly for easier manipulation or usage in the project.
  - `save_to_list.py`: This script is used to find and mark failed instances in the dataset.
  - TODO: `save_to_mysql.py`: This script is used to save the dataset to a MySQL database.
  - `save_to_pkl.py`: This script is used to convert files to a failed state, as part of the dataset creation process.
- Folder: **inference**
  - `app.py`: This script is used to display the gradio interface and run the inference functions.
  - `Dataset_processing.py`: This script is used to preprocess the dataset for the inference task.
  - `Inference.py`: This script is used to run the inference on the preprocessed dataset.
  - `Networks_inference.py`: This script contains the implementation of the deep learning networks for the inference task.

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

3. Download the dataset via the `app.py` script:

   ```bash
   python .\datasets_creation\app.py
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

## Contact

This project is maintained by [Prognostika](https://prognostika.com/). If you have any questions, please feel free to contact us at [lrt2436559745@gmail.com](mailto:lrt2436559745@gmail.com)