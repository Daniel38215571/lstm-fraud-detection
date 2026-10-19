# LSTM Fraud Detection Engine

## Overview
An experimental sequence-based machine learning model for detecting fraudulent transactions based on user behavior over time. 

## Background
My existing [Fraud Detection Engine](https://github.com/Daniel38215571/fraud-detection-engine) uses a Random Forest classifier, which evaluates transactions independently. This project explores whether a Long Short-Term Memory (LSTM) neural network can detect fraud better by analyzing the sequence of a user's last 10 transactions.

## Methodology
- **Data:** Generated synthetic sequence data simulating 10 transactions per user with 5 features (amount, velocity, geo-velocity, time deviation, merchant novelty).
- **Architecture:** TensorFlow/Keras LSTM (32 units) with a Dense layer and a Sigmoid output for binary classification.
- **Evaluation:** Accuracy, Precision, and Recall (prioritizing Recall to catch as much fraud as possible, even at the cost of false positives).

## Status
The baseline LSTM pipeline is complete and training successfully. The next step is to connect this to the real `fraud-detection-engine` dataset and compare the LSTM's performance directly against the Random Forest baseline.

## Stack
- Python
- TensorFlow / Keras
- NumPy
