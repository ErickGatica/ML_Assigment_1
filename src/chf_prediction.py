"""
This is the script to predict the Critical heat flux using different machine learning models with supervised learning.
"""
# Importing the required libraries
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import tracemalloc 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

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

# Function to evaluate the model performance

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    This function evaluates the model performance and returns the mean squared error.
    """
    start_time = time.time()
    # Tranining the model
    model.fit(X_train, y_train)

    # Predicting the target variable
    y_pred = model.predict(X_test)

    # Calculating the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    end_time = time.time()
    # Calculating the execution time in seconds
    execution_time = end_time - start_time

    #Get the current memory usage
    current, peak = tracemalloc.get_traced_memory()

    return y_pred, mse, execution_time, peak/10**6 # Returning predictions, MSE, execution time and memory usage in MB

# Linear Regression Model  

def linear_regression_model(X_train, X_test, y_train, y_test):
    """
    This function creates a linear regression model and evaluates the model performance.

    Input:
    X_train: Training data
    X_test: Testing data
    y_train: Training target variable
    y_test: Testing target variable

    Output:
    lin_reg: Linear regression model

    """
    lin_reg= LinearRegression()
    # Evaluating the model
    y_pred_lin, mse_lin, execution_time_lin, memory_lin = evaluate_model(lin_reg, X_train, X_test, y_train, y_test)  
    # Saving the results
    results['Linear Regression'] = {'y_pred': y_pred_lin , 'MSE': mse_lin, 'Execution Time': execution_time_lin, 'Memory Usage': memory_lin}
    print('Linear Regression Model')
    print('Mean Squared Error:', mse_lin)
    print('Execution Time:', execution_time_lin)
    print('Memory Usage:', memory_lin)
    print('-----------------------------------')
    return lin_reg, results['Linear Regression']

# Linear Regression Model (Regularized Linear Regression)

def ridge_regression_model(X_train, X_test, y_train, y_test, alpha):
    """
    This function creates a regularized linear regression model and evaluates the model performance.

    Input:
    X_train: Training data
    X_test: Testing data
    y_train: Training target variable
    y_test: Testing target variable

    Output:
    ridge_reg: Ridge regression model

    """

    ridge_reg = Ridge(alpha = alpha)
    # Evaluating the model
    y_pred_ridge, mse_ridge, execution_time_ridge, memory_ridge = evaluate_model(ridge_reg, X_train, X_test, y_train, y_test)
    # Saving the results
    results['Ridge Regression '+str(alpha)] = {'y_pred': y_pred_ridge , 'MSE': mse_ridge, 'Execution Time': execution_time_ridge, 'Memory Usage': memory_ridge}
    print('Ridge Regression Model with alpha = ', alpha)
    print('Mean Squared Error:', mse_ridge)
    print('Execution Time:', execution_time_ridge)
    print('Memory Usage:', memory_ridge)
    print('-----------------------------------')
    return ridge_reg, results['Ridge Regression '+str(alpha)]

# Logistic Regression Model

def logistic_regression_model(X_train, X_test, y_train, y_test):
    """
    This function creates a logistic regression model and evaluates the model performance.

    Input:
    X_train: Training data
    X_test: Testing data
    y_train: Training target variable
    y_test: Testing target variable

    Output:
    log_reg: Logistic regression model

    """

    log_reg = LogisticRegression()
    # Evaluating the model
    y_pred_log, mse_log, execution_time_log, memory_log = evaluate_model(log_reg, X_train, X_test, y_train, y_test)
    # Saving the results
    results['Logistic Regression'] = {'y_pred': y_pred_log , 'MSE': mse_log, 'Execution Time': execution_time_log, 'Memory Usage': memory_log}
    print('Logistic Regression Model')
    print('Mean Squared Error:', mse_log)
    print('Execution Time:', execution_time_log)
    print('Memory Usage:', memory_log)
    print('-----------------------------------')
    return log_reg, results['Logistic Regression']

# GDA Model

def gda_model(X_train, X_test, y_train, y_test):
    """
    This function creates a Gaussian Discriminant Analysis model and evaluates the model performance.

    Input:
    X_train: Training data
    X_test: Testing data
    y_train: Training target variable
    y_test: Testing target variable

    Output:
    gda: Gaussian Discriminant Analysis model

    """
    gda = QDA()
    # Evaluating the model
    y_pred_gda, mse_gda, execution_time_gda, memory_gda = evaluate_model(gda, X_train, X_test, y_train, y_test)
    # Saving the results
    results['Gaussian Discriminant Analysis'] = {'y_pred': y_pred_gda , 'MSE': mse_gda, 'Execution Time': execution_time_gda, 'Memory Usage': memory_gda}
    print('Gaussian Discriminant Analysis Model')
    print('Mean Squared Error:', mse_gda)
    print('Execution Time:', execution_time_gda)
    print('Memory Usage:', memory_gda)
    print('-----------------------------------')
    return gda, results['Gaussian Discriminant Analysis']

# Naive Bayes Model

def naive_bayes_model(X_train, X_test, y_train, y_test):
    """
    This function creates a Naive Bayes model and evaluates the model performance.

    Input:
    X_train: Training data
    X_test: Testing data
    y_train: Training target variable
    y_test: Testing target variable
    
    Output:
    nb: Naive Bayes model

    """
    nb = GaussianNB()
    # Evaluating the model
    y_pred_nb, mse_nb, execution_time_nb, memory_nb = evaluate_model(nb, X_train, X_test, y_train, y_test)
    # Saving the results
    results['Naive Bayes'] = {'y_pred': y_pred_nb , 'MSE': mse_nb, 'Execution Time': execution_time_nb, 'Memory Usage': memory_nb}
    print('Naive Bayes Model')
    print('Mean Squared Error:', mse_nb)
    print('Execution Time:', execution_time_nb)
    print('Memory Usage:', memory_nb)
    print('-----------------------------------')
    return nb, results['Naive Bayes']

# Function to plot the results from the models and the actual values

def plot_results(model_name, y_test, y_pred):
    """
    This function plots the results from the models and the actual values.
    
    Input:
    model_name: Name of the model
    y_test: Testing target variable
    y_pred: Predicted target variable

    Output:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, label = 'Data') # Scatter plot of the predicted values
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = 'red', linewidth = 4 , label = 'Ideal line') #  ideal line
    plt.title(f'{model_name} Model - Predicted vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    # Saving figure in the folder figures as png file 
    plt.tight_layout()
    plt.savefig(f'figures/{model_name}_results.png')
    # Show the plot
    plt.show()

    return plt

# Function to plot regression line (for linear models)

def plot_results_features(lin_model, X_test, y_test, feature):
    """
    This function is to plot the regression line whenyou use linear models

    Input:
    lin_model: Linear model
    X_test: Testing data
    y_test: Testing target variable
    feature: Feature to plot the regression line

    Output:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[feature], y_test, color = 'blue', alpha = 0.6, label = 'Actual')
    plt.scatter(X_test[feature], lin_model.predict(X_test), color = 'red', alpha= 0.6 , label = 'Predicted')
    plt.title('Regression Line')
    plt.xlabel(feature)
    plt.ylabel('CHF')
    plt.legend()
    # Saving figure in the folder figures as png file 
    plt.tight_layout()
    plt.savefig(f'figures/{lin_model}_{feature}_results.png')
    # Show the plot
    plt.show()
    return plt


# Function to plot the mean squared error for each model
def plot_mse(results):
    """
    This function plots the mean squared error for each model.

    Input:
    results: Dictionary with the results of the models

    Output:
    None
    """
    # Checking the models and creating a list with the names as strings
    models = list(results.keys())
    print(models)
    mse = [results[model]['MSE'] for model in models]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, mse)
    # Adding labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    plt.title('Mean Squared Error for each model')
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error')
    # Saving figure in the folder figures as png file 
    plt.tight_layout()
    plt.savefig(f'figures/MSE_results.png')
    # Show the plot
    plt.show()
    return plt

# Function to plot the execution time for each model
def plot_execution_time(results):
    """
    This function plots the execution time for each model.

    Input:
    results: Dictionary with the results of the models

    Output:
    None
    """
    models = list(results.keys())
    execution_time = [results[model]['Execution Time'] for model in models]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, execution_time)
    # Adding labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    plt.title('Execution Time for each model')
    plt.xlabel('Models')
    plt.ylabel('Execution Time (s)')
    # Saving figure in the folder figures as png file 
    plt.tight_layout()
    plt.savefig(f'figures/Execution_time_results.png')
    # Show the plot
    plt.show()
    return plt

# Function to plot the memory usage for each model
def plot_memory_usage(results):
    """
    This function plots the memory usage for each model.

    Input:
    results: Dictionary with the results of the models

    Output:
    None
    """
    models = list(results.keys())
    memory_usage = [results[model]['Memory Usage'] for model in models]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, memory_usage)
    # Adding labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    plt.title('Memory Usage for each model')
    plt.xlabel('Models')
    plt.ylabel('Memory Usage (MB)')
    # Saving figure in the folder figures as png file 
    plt.tight_layout()
    plt.savefig(f'figures/Memory_usage_results.png')
    # Show the plot
    plt.show()
    return plt

"""
# 4. Running the models
lin_model = linear_regression_model(X_train, X_test, y_train, y_test) # Linear Regression Model
#gda_model_r = gda_model(X_train, X_test, y_train, y_test) # Gaussian Discriminant Analysis Model
nb_model = naive_bayes_model(X_train, X_test, y_train, y_test) # Naive Bayes Model
#log_reg_model = logistic_regression_model(X_train, X_test, y_train, y_test) # Logistic Regression Model  ----> This model is so slow, we have to increase number of iterations
ridge_model_1 = ridge_regression_model(X_train, X_test, y_train, y_test, 0.1) # Ridge Regression Model with alpha = 0.1
ridge_model_8 = ridge_regression_model(X_train, X_test, y_train, y_test, 0.8) # Ridge Regression Model with alpha = 0.8


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

"""