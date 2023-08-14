# Indian Sign Language (ISL) Recognition using Convolutional Neural Networks (CNNs) for Multi-Class Classification

## Introduction

This repository is dedicated to the development of an advanced Convolutional Neural Network (CNN) model for the recognition of Indian Sign Language (ISL) gestures. The primary goal of this project is to create a sophisticated model that can accurately predict ISL letters from images, contributing to improved communication for the hearing-impaired community.

## CNN Architecture and Design

The CNN architecture is meticulously designed to leverage the spatial features present in ISL gestures. It consists of the following key components:

### Convolutional Layers

1. Initial Convolutional Layer:
   - Utilizes 64 filters of size 3x3 to extract lower-level features.
   - Employs Rectified Linear Unit (ReLU) activation for enhanced non-linearity.

2. Subsequent Convolutional Layer:
   - Comprises 32 filters of size 3x3, further capturing hierarchical features.
   - Continues to utilize ReLU activation.

### Pooling and Downsampling

- MaxPooling Layers:
  - Immediately follows each convolutional layer to perform feature downsampling.
  - Pool size of 2x2 is used to retain essential information while reducing spatial dimensions.

### Flattening and Fully Connected Layers

- Flattening Layer:
  - Converts the 2D feature maps into a 1D vector for subsequent fully connected layers.

- Dense Layers:
  - A dense layer with 128 units and ReLU activation is employed to capture high-level features.
  - Incorporates a Dropout layer with a rate of 0.5 for regularization.
  - The final dense layer has 36 units (representing the 36 possible ISL letters) and uses a softmax activation for multi-class classification.

## Dataset and Training

The dataset comprises a diverse collection of 42,000 ISL gesture images. The dataset is divided into training (33,600 images) and validation (8,400 images) sets. The model training is orchestrated over 20 epochs, utilizing the Adam optimizer and categorical cross-entropy loss.

### Training Results

- Training Accuracy: 99.92%
- Validation Accuracy: 100%

## Real-time Predictions and Performance

The model's efficacy is underscored by its real-time prediction capabilities. The model demonstrates high precision when predicting ISL letters, accurately identifying gestures such as 'a', 'l', '4', 'c', 'y', and more.

## Repository Structure

- `model.ipynb`: Comprehensive Jupyter Notebook containing the entire codebase, encompassing data preprocessing, model architecture, training, evaluation, and real-time predictions.
- `asl_model`: A dedicated directory housing the saved trained model weights.
- `sample_images`: A curated collection of sample ISL gesture images, facilitating testing and real-time prediction demonstrations.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/isl-recognition-cnn.git
