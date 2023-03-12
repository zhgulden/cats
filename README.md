# Сlassification of patients' condition using LSTM + Conv 1D 
Author: Yerkyn Yesbay

## Content
* [Introduction](#intro)
  * [Disease diagnosis](#seizure)
  * [EEG as a diagnostic tool](#eeg)
* [Problem](#problem)
* [Dataset](#data)
* [Data preprocessing](#prep)
* [Defining a Model](#model)
* [Dependencies](#dependencies)
* [Questions ans suggestions](#questions)



## <a name="intro"></a> Introduction

### <a name="seizure"></a> Disease diagnosis
Performing manual analysis of complex medical data for disease diagnosis is a time-consuming and error-prone task. Therefore, the development of machine learning enables researchers to propose new approaches for automating and partially alleviating some of the labor-intensive aspects of this task.


### <a name="eeg"></a> EEG as a diagnostic tool
EEG remains a key tool in diagnosing and managing patients with seizure disorders, along with a diverse array of other diagnostic methods developed in the past 30 years. This is because it is a convenient and cost-effective way to demonstrate the physiological state of abnormal cortical excitability that underlies epilepsy.

Abnormal electrical activity is frequently observed solely during seizures, while the brain activity is normal otherwise. The lack of an epileptic seizure during an EEG test simply indicates the absence of a seizure during the test, but it does not exclude the possibility of the patient having epilepsy.

Some people with epilepsy may have abnormal brain electrical activity even if they are not currently experiencing seizures. However, some people may have unusual EEG results not related to epilepsy, such as those caused by vision problems or brain injuries. Thus, the presence of unusual patterns of brain waves on an EEG does not always indicate the presence of epilepsy.

However, EEG has certain limitations. The electrical activity recorded by electrodes placed on the scalp or brain surface primarily reflects the summation of excitatory and inhibitory postsynaptic potentials in the apical dendrites of pyramidal neurons in the more superficial layers of the cortex. Quite large cortical areas - several square centimeters - must be activated synchronously to generate a sufficient potential for changes to be registered on electrodes placed on the scalp. The propagation of electrical activity through physiological pathways or through volume conduction in the extracellular space can give a false impression of the location of the electrical activity source. The cortical generators of many normal and abnormal cortical activities recorded on EEG are still largely unknown. Spatial sampling with standard scalp EEG is incomplete, as significant areas of the cortex, particularly in the basal and medial regions of the hemispheres, are not covered by the standard electrode placement. Temporal sampling is also limited, and the relatively short duration of routine EEG recordings in the interictal period is one reason why interictal epileptiform discharges may not be detected in patients with epilepsy during their initial EEG study.

##  <a name="problem"></a> Problem
In this study, our goal is to use EEG signals to classify patients' states and compare the performance of models with and without data preprocessing (wavelet transforms). The task at hand involves a set of sample pairs, expressed as:

$D = {(x_1,y_1), (x_2,y_2), … , (x_n,y_n)}$

Where $x_1, x_2, ..., x_n$ are observations and $y_1, y_2, ..., y_n$ are their corresponding class labels. The objective of this study is to find an accurate mapping between the feature space X and the class label space Y, i.e., $f: X \rightarrow Y$. The class space has a finite number of elements, i.e., $y \in {1, 2,..., K}$, where K=5 

##  <a name="data"></a> Dataset
The dataset used for the study consisted of EEG time series data from the University of Bonn, which had been restructured to contain 5 different target classes, 179 attributes, and 11500 samples. The original dataset contained 5 categories, each with 100 files representing a single subject. Each file recorded brain activity over 23.6 seconds, with the time series divided into 4097 data points representing EEG recordings at different times. Overall, there were 500 individuals in the dataset, each with 4097 data points over 23.5 seconds. The dataset was then divided and shuffled into 23 parts, each containing 178 data points per second, with the last column representing the label $y \in {1,2,3,4,5}$.

The response variable, y, was in column 179 and explanatory variables were X1, X2, ..., X178. The variable y indicated the category of a 178-dimensional input vector, with
- 1 - seizure activity
- 2 - EEG activity was recorded from the area of tumor localization
- 3 - a tumor was present in the brain, but the EEG activity was recorded in the healthy area of the brain
- 4 - eyes closed, meaning that the patient's eyes were closed during the EEG signal recording
- 5 - eyes open, meaning that the patient's eyes were open during the EEG brain signal recording

##  <a name="prep"></a> Data preprocessing
*** ***Implementation of this data preprocessing method can be found in the notebook***
The presented code is a `DataPreprocessor` class that is designed to preprocess EEG data for classification purposes. The class takes in a dataset consisting of EEG traces and 5 target patient conditions, and splits the data into training and testing sets using the `train_test_split` function from the `scikit-learn` library. The split data is then reshaped into the appropriate format for neural network input, and the target values are one-hot encoded using the `np.eye` function. The resulting preprocessed data is returned as `train_X`, `train_y_OH`, `test_X`, and `test_y_OH`.

This code is a useful tool for researchers and practitioners working in the field of EEG classification, as it simplifies the preprocessing of EEG data and makes it easier to train and test machine learning models on EEG data. The presented class can be integrated into a larger machine learning pipeline for EEG classification tasks. The use of this class can potentially lead to improved accuracy and performance of EEG classification models.



## <a name="model"></a>  Defining a Model 

*** ***Implementation of this models can be found in the notebook***

The current study introduces a model for automatic recognition of epileptic seizures through EEG signal analysis, employing a one-dimensional convolutional neural network with long-short-term memory (1D CNN-LSTM). Initially, the raw EEG signal data undergoes preprocessing and normalization. Subsequently, a one-dimensional convolutional neural network (CNN) is designed to efficiently extract features from the normalized EEG sequence data. The 1D CNN's convolutional filters and feature maps are exclusively one-dimensional, aligning with the raw EEG signal data's one-dimensional nature. As the CNN deepens, it extracts progressively higher-level features, enhancing its discriminative power in identifying epileptic seizures. The extracted features then undergo further processing through LSTM layers to extract temporal features. Finally, the output features traverse multiple fully connected layers for the ultimate recognition of epileptic seizures.


Experimental results indicate that the proposed method delivers high accuracy in recognizing epileptic seizures, even in tasks involving five-class data.



## <a name="dependencies"></a> Dependencies
#### pandas

```pip install pandas==1.4.1```

#### numpy

```pip install numpy==1.21.5```

#### sklearn

```pip install scikit-learn==1.0.2```

#### tensorflow

```pip install tensorflow==2.7.0```



## <a name="questions"></a> Questions and suggestions
If you have any questions or suggestions, write to the email: yesbay185@gmail.com
