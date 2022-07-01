# Real-Time Hand Gesture Recognition Based on Artificial Feed-Forward Neural Networks and EMG

## Overview
This work is the python implementation of [1] plus a validation methodology for hyperparameter's tuning. The problem addressed in [1] is the real-time hand gesture recognition based on superfictial electromyography (sEMG) signals. Given the sEMG data from the Myo-armband, the system is composed of four modules: preprocessing, feature extraction, classification, and post-processing.

## The System
### Preprocessing
The preprocessing module realizes the muscle activity detection through a rectification, low-pass filtering, short-time Fourier transform, and then segmentation of the input data based on a threshold.

### Feature Extraction/Selection
The feature extraction computes the dynamic time warping (DTW). First, we calculate the DTW of all signals in the training set in order to identify the gesture "center". Second, we apply the DTW between each signal and the centers, then we build the feature vector.

### Classification
The classification module is an artificial feed-forward neural network. There is only one hidden layer, which has the hyperbolic tangent as the activation function. The output layer has the softmax as activation. For training the neural network, we used the cross-entropy cost function and the conjugate gradient method.

### Post-processing
The post-processing is a time delay that eliminates consecutive repetitions of the same classification.

## Requirements
- Any OS.
- Packages in env.yml.

## Dataset
Available in this [link](https://drive.google.com/file/d/1DyHTOb_nNtDfwA8XT9vpxgLCRVIqA1QD/view?usp=sharing).

## Usage
- Download the [Dataset](https://drive.google.com/file/d/1DyHTOb_nNtDfwA8XT9vpxgLCRVIqA1QD/view?usp=sharing);
- Copy the dataset to the same folder of the source code;
- Execute main.py.

## References
[1] M. E. Benalcázar, C. E. Anchundia, J. A. Zea, P. Zambrano, A. G. Jaramillo, and M. Segura, “Real-time hand
gesture recognition based on artificial feed-forward neural networks and emg,” in 2018 26th European Signal
Processing Conference (EUSIPCO), 2018, pp. 1492–1496.
