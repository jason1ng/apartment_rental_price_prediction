# Apartment Rental Price Prediction
# Purpose/Objective:
- To collect and prepare the apartment rental dataset for modeling by handling missing values, treating outliers, encoding categorical features, and scaling numerical features, producing a clean and reliable dataset.
- To perform exploratory data analysis (EDA) and visualisation in order to identify the key relationships and patterns between property features and rental price
- To build and compare four regression models to predict apartment rental prices from property features: 
    **Linear Regression
    K-Nearest Neighbours Regressor
    Random Forest Regressor
    Gradient Boosting Regressor**
- To evaluate and compare the performance of the four models using RMSE, MAE, and R², and to identify the most accurate and reliable model for rental price prediction.
- To deploy the best-performing model as a functional prototype that allows users to estimate apartment rental prices interactively.

Dataset Source: https://www.kaggle.com/datasets/shashanks1202/apartment-rent-data

# Problem Statement:
- Which features of the property have the greatest impact on the price of an apartment.
- How effective are machine learning regression models in prediction of the rental price from these features.
- Which of the models gives the best predictions in terms of accuracy and reliability (based on the results of **RMSE, MAE, and R²**).

# Method:
Cross Industry Standard Process for Data Mining (CRISP-DM) framework with Python & Streamlit
Business Understanding -> Data Understanding -> Data Preparation -> Modelling -> Evaluation -> Deployment

# Code:
- [Business Understanding & Data Understanding] Includes visualization for business & data understanding of the dataset
- [Data Preparation] Includes data cleaning, data transformation and data outlier detection coding before modelling
- [Modelling] Includes all models training setup and enhancements
- [Evaluation] Includes all suitable evaluation shared by all models to do comparison
- [Deployment] Streamlit visualization and prediction
