# Heart-attack-possibility-Heart-Disease-Prediction
# Heart Disease Prediction using Machine Learning

This project applies machine learning techniques to predict whether a patient is likely to have heart disease based on clinical health indicators. The workflow covers end-to-end data processing, feature scaling, model training, hyperparameter tuning, evaluation, and comparison across multiple algorithms.The study demonstrates how predictive modeling can assist in risk assessment by identifying important factors in cardiovascular health.

# Dataset Source
This project uses publicly available data from:
- UCI Machine Learning Repository – Heart Disease Dataset

- Kaggle – Heart Attack Possibility Dataset

- The working dataset file used in this repository is heart.csv

# Dataset Type

This is a structured, supervised binary classification dataset from the healthcare domain.It contains labeled patient records with a mix of numerical and categorical features used to predict the presence or absence of heart disease.

# Dataset Details

Each record represents a patient profile with diagnostic indicators and a target label.

# Features

- age – Age of the patient

- sex – Gender (1 = male, 0 = female)

- cp – Chest pain type

- trestbps – Resting blood pressure

- chol – Serum cholesterol

- fbs – Fasting blood sugar

- restecg – Resting ECG results

- thalach – Maximum heart rate

- exang – Exercise-induced angina

- oldpeak – ST depression

- slope – ST segment slope

- ca – Number of major vessels

- thal – Thalassemia type
  
- target – Disease status (1 = presence, 0 = absence)

# Requirements

- Python 3.x
- Jupyter Notebook 
- Required Python libraries installed

# Required Libraries

- pandas
- numpy
- scikit-learn
- matplotlib
- gc

# Key Features Used for Prediction

## Strong predictors identified during exploratory analysis include:

- Chest pain classification (cp)
- Maximum heart rate (thalach)
- ST depression (oldpeak)
- Age
- Vessel count (ca)

Thalassemia (thal)

Exercise-induced angina (exang)


target – Disease status (1 = presence, 0 = absence)
