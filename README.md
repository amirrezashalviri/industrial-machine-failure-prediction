# Industrial Machine Failure Prediction

An end-to-end machine learning project for predicting industrial machine failures using operational and sensor data.

## Problem

The goal is to predict whether an industrial machine will experience a failure based on its operating conditions.

## Dataset

AI4I 2020 Predictive Maintenance Dataset.

- 10,000 samples
- 14 original columns
- Binary target: `Machine failure`
- Failure rate: 3.39%

The dataset is highly imbalanced, so accuracy alone is not considered sufficient for model evaluation.

## Features

The model uses:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

Identifier and failure-type columns were excluded from the predictive features to avoid unrealistic information leakage.

## Approach

1. Data understanding
2. Data quality analysis
3. Exploratory Data Analysis
4. Failure distribution analysis
5. Feature analysis
6. Train/test split with stratification
7. Logistic Regression baseline
8. Random Forest
9. Model evaluation
10. Feature importance analysis
11. False Negative analysis

## Models

### Logistic Regression

Used as a baseline model.

### Random Forest

Used as a nonlinear tree-based model capable of capturing interactions between machine operating conditions.

## Results

| Model | F1 | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.24 | 0.84 | 0.908 | 0.403 |
| Random Forest | 0.72 | 0.71 | 0.973 | 0.774 |

Random Forest performed substantially better than the Logistic Regression baseline.

## Feature Importance

The most important features according to the Random Forest model were:

1. Torque
2. Rotational speed
3. Tool wear
4. Air temperature
5. Process temperature

## Error Analysis

The Random Forest model detected 48 of 68 failure cases in the test set.

20 failure cases were classified as False Negatives.

These errors show that machine failures cannot be explained by a single operating variable and may result from combinations of machine conditions.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Pytest
- Git / GitHub

## Project Structure

```text
industrial-machine-failure-prediction/
├── data/
│   └── raw/
│       └── ai4i2020.csv
├── models/
├── results/
├── src/
│   └── main.py
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
