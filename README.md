# Heartattack Prediction using Machine Learning

This project applies machine learning techniques to predict whether a patient is likely to have heart disease based on clinical health indicators. The workflow covers end-to-end data processing, feature scaling, model training, hyperparameter tuning, evaluation, and comparison across multiple algorithms.The study demonstrates how predictive modeling can assist in risk assessment by identifying important factors in cardiovascular health.

## Dataset Source
This project uses publicly available data from:
- UCI Machine Learning Repository – Heart Disease Dataset

- Kaggle – Heart Attack Possibility Dataset

- The working dataset file used in this repository is heart.csv

## Dataset Type

This is a structured, supervised binary classification dataset from the healthcare domain.It contains labeled patient records with a mix of numerical and categorical features used to predict the presence or absence of heart disease.

## Dataset Details

- Each record represents a patient profile with diagnostic indicators and a target label.

## Features

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

## Requirements

- Python 3.x
- Jupyter Notebook 
- Required Python libraries installed

## Required Libraries

- pandas
- numpy
- scikit-learn
- matplotlib
- gc

## Key Features Used for Prediction

#### Strong predictors identified during exploratory analysis include:

- Chest pain classification (cp)
- Maximum heart rate (thalach)
- ST depression (oldpeak)
- Age
- Vessel count (ca)
- Thalassemia (thal)
- Exercise-induced angina (exang)
- target – Disease status (1 = presence, 0 = absence)

## Data Preprocessing
#### Exploratory Data Analysis

- Summary statistics generated using describe()

- Feature distributions explored using histograms

- Correlation patterns visually inspected

## Feature Engineering

- Features were split into numeric and categorical groups
#### Numerical features**:
- age
- trestbps
- chol
- thalach

### Categorical features:

- sex, cp, fbs, restecg, slope, ca, thal

## Feature Scaling

- Numerical features were standardized using Z-score normalization
- Implemented using StandardScaler
- Resulting features have: Mean = 0, Standard deviation = 1
- Scaled numeric variables were recombined with categorical features

## Data Split

- Training set: 85%
- Test set: 15%
- Data split achieved using: train_test_split(test_size=0.15)
- Reproducibility ensured using fixed random seeds

## Models Used
- Logistic Regression
- Tuned using GridSearchCV
- Hyperparameters optimized: Regularization strength (C), Penalty (L1/L2), Solver selection

#### Support Vector Machine (SVM)

- Hyperparameters tuned: Kernel type, C, gamma
- Best kernel: RBF

#### Random Forest

- Hyperparameters tuned

## Model Training and Optimization

- Cross-validation applied to all models
- Logistic Regression and SVM used 10-fold validation
- Random Forest used Stratified K-Fold (5 splits)
- Best configurations selected via GridSearchCV
- Final models retrained using optimal parameters

## Model Evaluation Metrics

#### The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC Score
- Confusion Matrix
- Classification Report

## Results
- Logistic Regression (Best Performing Model):
- Accuracy: 80%
- Recall: 95%
- Precision: 70%
- F1 Score: 80%
- AUC: 0.898


## Support Vector Machine

- Accuracy: 78%
- Recall (Disease class): 90%
- Precision: 69%

## Random Forest

- Accuracy: 74%
- Recall (Disease class): 90%
- Precision: 64%


## Model Comparison


| Model                | Accuracy | Recall (Disease) | AUC   |
|---------------------|----------|------------------|-------|
| Logistic Regression | 80%      | 95%              | 0.898 |
| SVM                 | 78%      | 90%              | —     |
| Random Forest       | 74%      | 90%              | —     |

## ROC Curve

- ROC curve plotted for Logistic Regression model
- AUC close to 0.90 indicates strong classification performance
- Model demonstrates excellent separation between positive and negative classes

## Conclusion

#### Logistic Regression was selected as the final model due to:

- Highest accuracy among tested models

- Excellent recall

- Superior ROC-AUC performance

- Model interpretability

- Low computational complexity

- SVM and Random Forest performed well in recall but produced higher false-positive rates.


## Future Work

- Apply SHAP for feature interpretability
- Experiment with class balancing methods like SMOTE

  
