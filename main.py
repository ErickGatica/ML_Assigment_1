"""
This is the script to run the main function of the project.
"""

# Importing the required libraries
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import tracemalloc 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


# Importing the required functions from the src folder
from src.chf_prediction import linear_regression_model, gda_model, naive_bayes_model, plot_results, plot_results_features, logistic_regression_model, ridge_regression_model, plot_mse, plot_execution_time, plot_memory_usage  

# Some global variables
results = {} # Dictionary to store the results of the models    

# Star tracking of memory usage
tracemalloc.start()

# 1. We load the data set
#  https://www.nrc.gov/reading-rm/doc-collections/nuregs/knowledge/km0011/index.html#pub-info
#  https://www.kaggle.com/datasets/kaustubhdhote/critical-heat-flux-dataset
data = pd.read_csv('data/CHF_data_ML.csv')


# 2. Defining the features and the target variable
X = data[['D', 'L', 'P', 'G', 'X_chf', 'Dh_in', 'T_in']] # Features. D: Diameter, L: Length, P: Pressure, G: Mass flux, X_chf : Quality, Dh_in: Hydraulic diameter, T_in: Inlet temperature
y = data['CHF'] # Target variable. CHF: Critical heat flux 

# 3. Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 4. Running the models
lin_model, results['Linear Regression'] = linear_regression_model(X_train, X_test, y_train, y_test) # Linear Regression Model
#gda_model_r , results['Gaussian Discriminant Analysis'] = gda_model(X_train, X_test, y_train, y_test) # Gaussian Discriminant Analysis Model
nb_model, results['Naive Bayes'] = naive_bayes_model(X_train, X_test, y_train, y_test) # Naive Bayes Model
#log_reg_model , results['Logistic Regression'] = logistic_regression_model(X_train, X_test, y_train, y_test) # Logistic Regression Model  ----> This model is so slow, we have to increase number of iterations
ridge_model_1 , results['Ridge Regression 0.1'] = ridge_regression_model(X_train, X_test, y_train, y_test, 0.1) # Ridge Regression Model with alpha = 0.1
ridge_model_8, results['Ridge Regression 0.8'] = ridge_regression_model(X_train, X_test, y_train, y_test, 0.8) # Ridge Regression Model with alpha = 0.8


# 5. Plotting the results for each model
for model_name in results.keys():
    print(model_name)
    plot_results(model_name, y_test, results[model_name]['y_pred'])
    
    

# 6. Plotting the data in function of the features for linear regression model
plot_results_features(lin_model, X_test, y_test, 'D') # Diameter m 
plot_results_features(lin_model, X_test, y_test, 'L') # Length m 
plot_results_features(lin_model, X_test, y_test, 'P') # Pressure kPa
plot_results_features(lin_model, X_test, y_test, 'G') # Mass flux kg/m2s
plot_results_features(lin_model, X_test, y_test, 'X_chf') # Quality 
plot_results_features(lin_model, X_test, y_test, 'Dh_in') # Hydraulic diameter m
plot_results_features(lin_model, X_test, y_test, 'T_in') # Input temperature C


# Plotting the mean squared error, execution time and memory usage for each model
plot_mse(results)
plot_execution_time(results)
plot_memory_usage(results)

# Stop tracking memory usage
tracemalloc.stop()