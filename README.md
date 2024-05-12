# tcn-hard-disk-failure-prediction

## Table of Contents

- [tcn-hard-disk-failure-prediction](#tcn-hard-disk-failure-prediction)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Code Structure](#code-structure)
  - [Description of each file](#description-of-each-file)
  - [SMART Attributes Selection](#smart-attributes-selection)
    - [Metrics Used](#metrics-used)
    - [Description of Selected Metrics](#description-of-selected-metrics)
  - [Code Process](#code-process)
    - [Main Classification Process](#main-classification-process)
    - [Feature Selection Subflowchart](#feature-selection-subflowchart)
    - [Partition Dataset Subflowchart](#partition-dataset-subflowchart)
    - [Random Forest Classification Process](#random-forest-classification-process)
    - [TCN Classification Process](#tcn-classification-process)
      - [TCN Network Initialization Subflowchart](#tcn-network-initialization-subflowchart)
      - [Iterative Training Process](#iterative-training-process)
      - [TCN Training Subflowchart](#tcn-training-subflowchart)
      - [TCN Testing Subflowchart](#tcn-testing-subflowchart)
    - [LSTM Classification Process](#lstm-classification-process)
      - [Iterative Training Process](#iterative-training-process-1)
      - [LSTM Training Subflowchart](#lstm-training-subflowchart)
      - [LSTM Testing Subflowchart](#lstm-testing-subflowchart)

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
└── datasets_creation
    ├── files_to_failed.py
    ├── find_failed.py
    ├── get_dataset.py
    ├── README.txt
    └── toList.py
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

## SMART Attributes Selection

### Metrics Used

| SMART | Feature Selected | Attribute Name | Description |
| --- | --- | --- | --- |
| 1 | Yes | Raw read error rate | Rate of hardware read errors that occur when reading data from a disk. |
| 3 | No | Spin-up time | Average time (in milliseconds) of spindle spin up from zero RPM to fully operational. |
| 5 | Yes | Reallocated sectors count | Count of bad sectors that have been found and reallocated. A hard drive which has had a reallocation is very likely to fail in the immediate months. |
| 7 | Yes | Seek error rate | Rate of seek errors of the magnetic heads, due to partial failure in the mechanical positioning system. |
| 9 | Yes | Power-on hours | Count of hours in power-on state. |
| 12 | Yes | Power cycle count | Count of full hard drive power on/off cycles. |
| 183 | Yes | SATA downshift error count | Count of errors in communication between the drive and the host controller. |
| 184 | Yes | End-to-end error | Count of errors in communication between the drive and the host controller. |
| 187 | Yes | Reported uncorrectable errors | Count of errors that could not be recovered using hardware ECC (Error-Correcting Code), a type of memory used to correct data corruption errors. |
| 188 | No | Command timeout | Count of aborted operations due to hard drive timeout. |
| 189 | Yes | High fly writes | Count of times the recording head "flies" outside its normal operating range. |
| 190 | No | Temperature difference | Difference between current hard drive temperature and optimal temperature of 100°C. |
| 193 | Yes | Load cycle count | Count of load/unload cycles into head landing zone position. |
| 197 | Yes | Current pending sectors count | Count of bad sectors that have been found and waiting to be reallocated, because of unrecoverable read errors. |
| 198 | Yes | Offline uncorrectable sectors count | Total count of uncorrectable errors when reading/writing a sector, indicating defects of the disk surface or problems in the mechanical subsystem. |
| 199 | Yes | UltraDMA CRC error rate | Count of errors in data transfer via the interface cable as determined by the CRC (Cyclic Redundancy Check). |

### Description of Selected Metrics

1. **Raw Read Error Rate**
   - This metric quantifies the rate at which the hard drive encounters errors when reading data from the disk surface. It’s an indicator of the physical condition of the disk, where a higher value may suggest a deterioration in disk quality, potentially leading to data loss.

2. **Reallocated Sectors Count**
   - This count reflects the number of sectors on the disk that have been identified as defective and subsequently replaced with spare sectors from the reserve pool. Frequent reallocations can be a precursor to drive failure, as they indicate significant wear or damage to the disk surface.

3. **Seek Error Rate**
   - This rate measures the frequency of errors encountered by the drive's head when trying to reach a specific area of the disk. These errors are often due to mechanical failures within the drive’s moving components or external shocks and vibrations affecting the drive's precise operations.

4. **Power-On Hours**
   - This value records the cumulative number of hours the hard drive has been active. It is used to estimate the age of the drive and assess its wear level, helping predict end-of-life by comparing against typical lifespan estimates for the drive model.

5. **Power Cycle Count**
   - This count logs the number of times the hard drive has been powered up and down. Frequent power cycles can stress the mechanical components of the drive, especially the spindle motor and bearings, reducing the overall lifespan of the device.

6. **SATA Downshift Error Count**
   - This metric tracks the number of times the SATA interface has downshifted from a higher to a lower speed due to errors in data transmission. Persistent errors in downshifting can indicate issues with either the drive’s controller or the SATA cable quality.

7. **End-to-End Error**
   - This error count monitors the data integrity as data is transferred internally between the drive's buffer and its host. Errors here can imply issues with the drive's internal processing or hardware malfunctions that could compromise data integrity.

8. **Reported Uncorrectable Errors**
   - This feature logs the number of errors that could not be fixed using the hardware's error-correcting code. It’s a critical measure of a drive’s ability to maintain data integrity, with high values suggesting a risk of data loss.

9. **Load Cycle Count**
   - This count tracks how often the drive's heads are loaded into the read/write position. Excessive loading and unloading can accelerate wear on the head and the drive medium, potentially leading to drive failure.

10. **Current Pending Sectors Count**
    - This value reports the number of unstable sectors that have yet to be reallocated. Sectors awaiting reallocation can cause data read/write errors and may eventually be marked as bad, affecting data retrieval and overall system performance.

11. **Offline Uncorrectable Sectors Count**
    - This count reflects the number of sectors that failed during offline operations (such as during more intensive scans) and could not be corrected. It indicates problems with the disk surface or with the read/write heads.

12. **UltraDMA CRC Error Rate**
    - This measures the frequency of cyclic redundancy check (CRC) errors during Ultra DMA mode. These errors are usually due to problems with the drive interface or data cable issues and can significantly affect data transfer reliability and speed.

## Code Process

### Main Classification Process

```mermaid
graph TD
    A[Start]
    B{Load Dataset}
    C[Import Data]
    D[Filter Out Bad HDs]
    E[Define RUL Piecewise]
    F[Subflowchart: <br>Feature Selection]
    G[Subflowchart: <br>Partition Dataset]
    H{Classifier Type}
    I[RandomForest]
    J[Subflowchart: <br>TCN]
    K[Subflowchart: <br>LSTM]
    L[Feature Extraction]
    M[Reshape Data]
    N{Perform Classification}
    O[End]
    A --> B
    B -- Fail --> C
    C --> D
    D --> E
    E --> F
    F --> G
    B -- Success --> G
    G --> H
    H --> I
    H --> J
    H --> K
    I --> N
    J --> N
    K --> N
    N --> O
    L -- If perform_features_extraction is True --> M
    M --> N
```

> **TODO:**
> 
> Note 1: We can fix the unbalance of the dataset by using downsampling method to balance the dataset. Specifically, current code use SMOTE method to balance the dataset with 5 as resampler balance ratio. We can change this value to dynamically balance the dataset based on the number of samples in each class.
>
> In conclusion, whether GA or KNN will be better for feature selection depends on your specific use case. If you have a large number of features and you suspect that there may be complex interactions between features, GA might be a better choice. If you have a smaller number of features or computational efficiency is a concern, KNN might be a better choice. It's also worth noting that feature selection is often an iterative and experimental process, and it can be beneficial to try multiple methods and see which one works best for your specific dataset.
>
> Note 2: We can use the Genetic Evolution algorithm for feature selection. Paper reference: **Genetic Algorithm for feature selection in enhancing the hard disk failure prediction using artificial neural network**
>
> Note 3: 

### Feature Selection Subflowchart

```mermaid
graph TD
    A[Start] --> B[Define empty lists and dictionary]
    B --> C{For each feature in df.columns}
    C -->|If 'raw' in feature| D[Perform T-test]
    D --> E[Store p-value in dictionary]
    C -->|If 'smart' not in feature| F[Concatenate feature to features list]
    E --> G[Sort dictionary by p-value]
    G --> H[Convert dictionary to DataFrame and drop NaNs]
    H --> I[Select top 'num_features' features]
    I --> J[Update df to only include selected features]
    J --> K[End]
    F --> J
```

### Partition Dataset Subflowchart

```mermaid
graph TD
    A[Start: dataset_partitioning] --> B[Reset Index and Preprocess Data]
    B --> C{Check Windowing}
    C -- Yes --> D[Attempt to Load Pre-existing Windowed Dataset]
    D -- Success --> E[Loaded Existing Dataset]
    E --> F[Prepare Data for Modeling]
    D -- Failure --> G[Windowing Process]
    G --> F
    C -- No --> F
    F --> H{Technique Selection}
    H -- Random --> I[Random Partitioning]
    H -- HDD --> J[HDD Partitioning]
    H -- Other --> K[Other Technique]
    I --> L[Apply Sampling Techniques]
    J --> L
    K --> L
    L --> M[Final Dataset Creation]
    M --> N[Return Train and Test Sets]
```

### Random Forest Classification Process

This sequence is provided by the third-party library (sklearn), so the process is not detailed here.

### TCN Classification Process

#### TCN Network Initialization Subflowchart

```mermaid
graph TD
    A["Input Layer<br/>(num_inputs, history_signal)"] --> B0["Dilated Convolution Block 0"]
    B0 --> B1["Dilated Convolution Block 1"]
    B1 --> B2["Dilated Convolution Block 2"]
    B2 --> FC0["Fully Connected Layer 0"]
    FC0 --> FC1["Fully Connected Layer 1"]
    FC1 --> GFC["Final Output Layer<br/>(GwayFC)"]

    subgraph B0["Dilated Convolution Block 0"]
        b0_tcn0[("Conv1d: 32 outputs<br/>Kernel: 3, Dilation: 2, Padding: 2")]
        b0_bn0[("BatchNorm1d: 32 features")]
        b0_relu0[("ReLU")]
        b0_tcn1[("Conv1d: 64 outputs<br/>Kernel: 3, Dilation: 2, Padding: 2")]
        b0_pool[("AvgPool1d: Kernel: 3, Stride: 2, Padding: 1")]
        b0_bn1[("BatchNorm1d: 64 features")]
        b0_relu1[("ReLU")]

        b0_tcn0 --> b0_bn0 --> b0_relu0 --> b0_tcn1 --> b0_pool --> b0_bn1 --> b0_relu1
    end

    subgraph B1["Dilated Convolution Block 1"]
        b1_tcn0[("Conv1d: 64 outputs<br/>Kernel: 3, Dilation: 2, Padding: 2")]
        b1_bn0[("BatchNorm1d: 64 features")]
        b1_relu0[("ReLU")]
        b1_tcn1[("Conv1d: 128 outputs<br/>Kernel: 3, Dilation: 2, Padding: 2")]
        b1_pool[("AvgPool1d: Kernel: 3, Stride: 2, Padding: 1")]
        b1_bn1[("BatchNorm1d: 128 features")]
        b1_relu1[("ReLU")]

        b1_tcn0 --> b1_bn0 --> b1_relu0 --> b1_tcn1 --> b1_pool --> b1_bn1 --> b1_relu1
    end

    subgraph B2["Dilated Convolution Block 2"]
        b2_tcn0[("Conv1d: 128 outputs<br/>Kernel: 3, Dilation: 4, Padding: 4")]
        b2_bn0[("BatchNorm1d: 128 features")]
        b2_relu0[("ReLU")]
        b2_tcn1[("Conv1d: 128 outputs<br/>Kernel: 3, Dilation: 4, Padding: 4")]
        b2_pool[("AvgPool1d: Kernel: 3, Stride: 2, Padding: 1")]
        b2_bn1[("BatchNorm1d: 128 features")]
        b2_relu1[("ReLU")]

        b2_tcn0 --> b2_bn0 --> b2_relu0 --> b2_tcn1 --> b2_pool --> b2_bn1 --> b2_relu1
    end

    subgraph FC0["Fully Connected Layer 0"]
        fc0[("Linear: 256 units")]
        fc0_bn[("BatchNorm1d: 256 features")]
        fc0_relu[("ReLU")]
        fc0_drop[("Dropout: 50%")]

        fc0 --> fc0_bn --> fc0_relu --> fc0_drop
    end

    subgraph FC1["Fully Connected Layer 1"]
        fc1[("Linear: 64 units")]
        fc1_bn[("BatchNorm1d: 64 features")]
        fc1_relu[("ReLU")]
        fc1_drop[("Dropout: 50%")]

        fc1 --> fc1_bn --> fc1_relu --> fc1_drop
    end

    GFC[("GwayFC: 2 outputs")]
```

#### Iterative Training Process

```mermaid
graph TD
    A[Start]
    B[Initialize F1_list and counter i]
    C{Each epoch in 1 to epochs}
    D[Subflowchart: <br>Model Training]
    E[Store F1 score in F1_list]
    F[Increment counter i]
    G{Check if i equals 5}
    H[Reset counter i to 0, check F1_list]
    I{Check if epoch is a multiple of 20}
    J[Reduce learning rate by a factor of 10]
    K[End]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G -- Yes --> H
    H -- F1_list constant --> K
    H -- F1_list varies --> I
    I -- Yes --> J
    J --> C
    I -- No --> C
```

#### TCN Training Subflowchart

```mermaid
graph TD
    Z[Start Training]
    A[Initialize train_loss, correct, predictions]
    B[Shuffle Xtrain, ytrain]
    C{For each batch in Xtrain}
    D[Prepare data, target]
    E[Check and move to GPU if CUDA available]
    F[Set gradients to zero]
    G[Forward pass]
    H[Calculate batch loss]
    I[Backward pass]
    J[Optimization step]
    K[Update predictions, correct count]
    L[Update train_loss]
    M{Check if batch_idx is a multiple of 10}
    N[Print training progress, Reset train_loss]
    O[End of batch loop]
    P[Print epoch status, Calculate F1]
    Q[Flowchart: Model Testing]
    R[End of Training]
    Z --> A --> B --> C
    C --> D --> E --> F --> G --> H --> I --> J --> K --> L --> M
    M -- Yes --> N --> C
    M -- No --> C
    C -- End of loop --> O --> P --> Q --> R
```

#### TCN Testing Subflowchart

```mermaid
graph TD
    A[Start]
    B[Set model to evaluation mode]
    C[Initialize test_loss, correct, batchsize, <br>nbatches, predictions, and criterion]
    D{Start loop over batches}
    E[Prepare data and target]
    F[Check if CUDA is available]
    G[Move data and target to GPU]
    H[Pass data through model to get output]
    I[Calculate loss]
    J[Find predicted class]
    K[Update correct count]
    L[Store predictions]
    M[End of loop]
    N[Calculate average test_loss]
    O[Print 'T']
    P[Print test set average loss and accuracy]
    Q[Report metrics]
    R[End of Testing]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F -- Yes --> G
    G --> H
    F -- No --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> D
    D -- End of loop --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> R
```

### LSTM Classification Process

#### Iterative Training Process

```mermaid
graph TD
    A[Start]
    B[Create train dataset]
    C[Create test dataset]
    D[Initialize train loader]
    E[Initialize test loader]
    F[Begin training and validation process]
    G[Set up training environment]
    H[Initialize F1_list and counter]
    I{For each epoch in 1 to epochs}
    J[Subflowchart: <br>Call train_LSTM with current epoch]
    K[Update F1 score and training metrics]
    L[Subflowchart: <br>Call test_LSTM for validation]
    M[Store F1 score in F1_list]
    N[Increment counter i]
    O{Check if i equals 5}
    P{Check F1_list for early stopping criteria}
    Q[Adjust learning rate every 20 epochs]
    R[Reset counter i]
    S[Continue training]
    T[End training]
    U[End]
    A --> B
    A --> C
    B --> D
    C --> E
    D --> F
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O -- Yes --> P
    O -- No --> I
    P -- Continue --> Q
    P -- Stop --> T
    Q -- Adjusted --> R
    R --> I
    T --> U
    I -- Last Epoch --> T
```

#### LSTM Training Subflowchart

```mermaid
graph TD
    A[Start]
    B[Initialize train_loss, set model to train mode, correct, weights, <br>class_weights, predictions, ytrain, and criterion]
    C{Start loop over train_loader}
    D[Prepare sequences and labels]
    E[Set batchsize]
    F[Move sequences and labels to GPU]
    G[Set gradients of all model parameters to zero]
    H[Pass sequences through model to get output]
    I[Calculate loss]
    J[Backward pass: compute gradient of the loss <br>with respect to model parameters]
    K[Perform a single optimization step]
    L[Find predicted class]
    M[Store predictions and labels]
    N[Update correct count]
    O[Update train_loss]
    P{Check if i is a multiple of 10}
    Q[Print training progress]
    R[Reset train_loss]
    S[End of loop]
    T[Print 'T']
    U[Trim ytrain and predictions to match actual size]
    V[Calculate F1 score and other metrics]
    W[End]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    P -- Yes --> Q
    Q --> R
    R --> C
    P -- No --> C
    C -- End of loop --> S
    S --> T
    T --> U
    U --> V
    V --> W
```

#### LSTM Testing Subflowchart

```mermaid
graph TD
    A[Start]
    B[Set model to evaluation mode]
    C[Initialize test_loss, correct, and criterion]
    D{Start loop over test_loader}
    E[Prepare sequences and labels]
    F[Move sequences and labels to GPU]
    G[Pass sequences through model to get output]
    H[Find predicted class]
    I[Calculate loss]
    J[Update test_loss]
    K[Update correct count]
    L[Store predictions and labels]
    M[End of loop]
    N[Calculate average test_loss]
    O[Print 'T']
    P[Print test set average loss and accuracy]
    Q[Report metrics]
    R[End]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> D
    D -- End of loop --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> R
```