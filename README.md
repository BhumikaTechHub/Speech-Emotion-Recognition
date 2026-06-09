# Speech-Emotion-Recognition
The objective of this project is to classify emotions from speech audio recordings using engineered audio features and evaluate model performance using machine learning and deep learning approaches.

## Overview

Speech Emotion Recognition (SER) is the task of automatically identifying human emotions from speech signals. Understanding emotions from speech is a critical component of modern conversational AI systems, virtual assistants, voice-based educational platforms, and human-computer interaction systems.

This project develops an end-to-end Speech Emotion Recognition pipeline using audio feature engineering, machine learning, and deep learning techniques. The system processes raw speech recordings, extracts meaningful acoustic features, and classifies them into multiple emotional categories.

The project was implemented using Python, Librosa, Scikit-Learn, TensorFlow, and the RAVDESS emotional speech dataset.

---

#  Objectives

The primary objectives of this project are:

* Analyze speech signals and extract emotion-related acoustic characteristics.
* Perform audio preprocessing and feature engineering.
* Train machine learning and deep learning models for emotion classification.
* Compare model performance using multiple evaluation metrics.
* Understand the effectiveness of engineered speech features for emotion recognition tasks.

---

#   Real-World Applications

Speech Emotion Recognition has applications in:

* Conversational AI and Chatbots
* Virtual Assistants
* Child Speech Understanding Systems
* Mental Health Monitoring
* Customer Support Analytics
* Smart Education Platforms
* Human-Computer Interaction
* Voice-Based Recommendation Systems

---

#  Dataset

## RAVDESS Dataset

The project uses the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).

The dataset contains professionally recorded speech samples from multiple actors expressing different emotions.

### Emotion Classes

| Emotion Code | Emotion   |
| ------------ | --------- |
| 01           | Neutral   |
| 02           | Calm      |
| 03           | Happy     |
| 04           | Sad       |
| 05           | Angry     |
| 06           | Fearful   |
| 07           | Disgust   |
| 08           | Surprised |

### Dataset Statistics

* Total Speech Audio Files: 1440
* Audio Format: WAV
* Sample Rate: 22050 Hz
* Number of Emotion Classes: 8

---

#   Project Workflow

Raw Audio Files
     ↓
Audio Preprocessing
     ↓
Feature Engineering 
     ↓
Feature Matrix Creation
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Random Forest Classifier
     ↓
Neural Network Classifier
     ↓
Model Evaluation
     ↓
Performance Comparison

---

#  Data Preprocessing

Several preprocessing steps were performed before training the models.

## Audio Loading

Speech recordings were loaded using Librosa.

The audio signal was converted into a numerical representation suitable for feature extraction.

## Label Extraction

Emotion labels were extracted directly from the RAVDESS file naming convention.

Example:

03-01-05-01-01-01-01.wav

Emotion Code = 05

Emotion = Angry

## Dataset Creation

A structured dataframe was created containing:

* Audio File Path
* Emotion Label

## Train-Test Split

The dataset was divided into:

* Training Set: 80%
* Testing Set: 20%

This ensured unbiased evaluation on unseen samples.

## Feature Standardization

StandardScaler was applied to normalize feature values before training.

Benefits:

* Faster convergence
* Stable optimization
* Improved model performance

---

#  Feature Engineering

Feature engineering is the most important stage of this project.

Instead of using raw audio signals directly, several acoustic features were extracted.

---

## 1. MFCC (Mel Frequency Cepstral Coefficients)

MFCCs are among the most widely used features in speech processing.

They mimic human auditory perception and capture important speech characteristics.

Extracted Features:

* 40 MFCC Coefficients

Benefits:

* Captures speech timbre
* Reduces dimensionality
* Effective for speech classification tasks

---

## 2. Zero Crossing Rate (ZCR)

Measures the rate at which the audio signal changes sign.

Useful for distinguishing different speech patterns and emotional expressions.

Extracted Features:

* 1 Feature

---

## 3. RMS Energy

Root Mean Square Energy measures the loudness and intensity of speech.

Useful because emotional speech often varies in energy levels.

Extracted Features:

* 1 Feature

---

## 4. Spectral Centroid

Represents the center of mass of the frequency spectrum.

Helps capture frequency distribution differences between emotions.

Extracted Features:

* 1 Feature

---

## Final Feature Vector

| Feature Type      | Count |
| ----------------- | ----- |
| MFCC              | 40    |
| ZCR               | 1     |
| RMS Energy        | 1     |
| Spectral Centroid | 1     |

### Total Features = 43

---

#  Models Used

## 1. Random Forest Classifier

Random Forest was used as a baseline machine learning model.

### Advantages

* Handles nonlinear relationships
* Robust to overfitting
* Works well with engineered features
* Easy to interpret

### Parameters

* n_estimators = 200
* random_state = 42

---

## 2. Deep Neural Network

A fully connected neural network was developed using TensorFlow/Keras.

### Architecture

Input Layer (43 Features)
         ↓
Dense Layer (256 Neurons, ReLU)
         ↓
Dropout (0.3)
         ↓
Dense Layer (128 Neurons, ReLU)
         ↓
Dropout (0.3)
         ↓
Dense Layer (64 Neurons, ReLU)
         ↓
Output Layer (8 Neurons, Softmax)


### Advantages

* Learns complex feature relationships
* Better generalization
* Stronger classification capability

---

#  Model Evaluation

The models were evaluated using multiple metrics.

## Accuracy

Measures overall prediction correctness.

## Precision

Measures how many predicted emotion labels were correct.

## Recall

Measures how many actual emotions were correctly identified.

## F1 Score

Harmonic mean of Precision and Recall.

## Confusion Matrix

Provides a detailed breakdown of classification performance across all emotion classes.

---

#  Results

## Random Forest Performance

- Accuracy: 57%
- Precision: 0.58
- Recall: 0.57
- F1 Score: 0.57

---

## Neural Network Performance

- Accuracy: 64%
- Precision: 0.63
- Recall: 0.62
- F1 Score: 0.62

---

## Performance Comparison

| Model          | Accuracy |
| -------------- | -------- |
| Random Forest  | 57%      |
| Neural Network | 64%      |

---


#   Technologies Used

## Programming Language

* Python

## Libraries

* NumPy
* Pandas
* Librosa
* Matplotlib
* Seaborn
* Scikit-Learn
* TensorFlow
* Joblib

## Development Environment

* Google Colab
* Jupyter Notebook

---

#   Future Improvements

Potential enhancements include:

* CNN-based Spectrogram Classification
* LSTM-based Temporal Speech Modeling
* Transformer-based Speech Models (Wav2Vec2)
* Real-Time Microphone Emotion Detection
* Child Speech Emotion Recognition
* Speech-to-Text Integration
* Deployment using Streamlit or Flask
* Model Quantization for Edge Devices

---

# Author

**Bhumika Yadav**

B.Tech, Computer Science & Engineering

GitHub: https://github.com/BhumikaTechHub

LinkedIn: https://www.linkedin.com/in/bhumika-yadav-826255296/

---

## ⭐ If you found this project useful, consider giving the repository a star.
