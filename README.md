# Sepsis_Prediction
Sepsis Prediction using Machine Learning
This project leverages machine learning to predict the likelihood of sepsis in patients based on specific clinical features. Sepsis is a severe, potentially life-threatening condition caused by the body's extreme response to an infection. Early prediction of sepsis can enable timely interventions, potentially reducing morbidity and mortality rates.

# Project Overview
This repository includes a web-based application for sepsis prediction, built with FastAPI for backend deployment and a machine learning model trained on clinical data. The model takes input features related to patient health metrics and outputs the probability of sepsis.

# Key Features
Machine Learning Model: A trained model based on multiple classifiers (e.g., Logistic Regression, Random Forest, XGBoost) that predicts the probability of sepsis. The model was chosen based on performance metrics such as F1 score.
FastAPI Backend: A fast, robust API backend that serves predictions to an interactive web interface.
Dynamic Web Interface: A simple, user-friendly HTML front end designed with a professional medical appearance, allowing users to enter patient data and view prediction results.
Scalable and Modular Code: Code structure is modular, making it easy to update the model or integrate additional features.

# Features Used for Prediction
The model considers the following clinical features to assess sepsis risk:

Plasma Glucose Level
Blood Work R1
Blood Pressure
Blood Work R3
BMI (Body Mass Index)
Blood Work R4
Patient Age
Installation and Setup

# Requirements
Python 3.7+
FastAPI
Uvicorn
Pandas, NumPy, Scikit-Learn
Pickle (for loading the trained model)

# Model Training
The model was trained using a dataset with multiple features, followed by hyperparameter tuning and k-fold cross-validation to ensure optimal performance. The selected model showed the best balance between sensitivity and specificity, ensuring reliable predictions for both positive and negative cases.
