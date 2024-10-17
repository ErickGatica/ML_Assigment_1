# Report 

## Abstract



## Introduction

Over the past 70 years, numerous methods have been developed for predicting critical heat flux (CHF). Early approaches were mostly empirical, relying on basic correlations that lacked a solid physical foundation and had limited applicability. As time progressed, a variety of phenomenological equations and physical models were introduced, many of which were incorporated into reactor safety analysis codes. However, these models depend on the specific mechanisms that govern CHF, which vary with different flow regimes. As a result, multiple models, equations, or correlations must be employed to accurately predict CHF in typical reactor conditions. Given the need for such diversity and the existence of more than 500 CHF correlations for water-cooled tubes alone, a more comprehensive and universal CHF prediction method became necessary [1].

CHF refers to the maximum heat flux (amount of heat per unit area) that can be transferred from a surface to a boiling liquid before the surface temperature dramatically increases. This occurs when a large portion of the heating surface is covered by vapor rather than liquid, which reduces heat transfer.

When CHF is exceeded, it leads to a condition known as boiling crisis or departure from nucleate boiling (DNB). Beyond this point, the surface can overheat, which can be dangerous, especially in reactors or other heat-sensitive systems.




## Method

To analyze if it is possible to build Machine learning based models to predict the CHF in different conditions the data from the Groeneveld was used. 

The data used to derive the 2006 Groeneveld Critical Heat Flux (CHF) Lookup Table comprises over 25,000 experimental CHF data points. These were collected from water-cooled tubes and represent a compilation of 62 datasets gathered over the past 60 years. The datasets vary widely in their conditions, covering different pressures, mass fluxes, diameters, and inlet temperatures. These parameters include tube diameter (D), heated length (L), pressure (P), mass flux (G), critical quality (X_chf), inlet enthalpy (Dh_in), and inlet temperature (T_in), all of which are critical for modeling CHF. This extensive database was used to create a reliable and comprehensive lookup table that can predict CHF across a wide range of conditions​

The data was splitted in a training set and in a test set, this last one being the 20% of the original data set.

The Sklearn library from phyton was used for this project.

The selected model to try were:
- Linear regression model
- Naive Bayes model
- Ridge Linear Regression model with alpha 0.1
- Ridge Linear Regression model with alpha 0.8


The Linear Regression model is a fundamental approach to predictive modeling. It attempts to model the relationship between a dependent variable (in this case, CHF) and one or more independent variables (the input features like D, L, P, G, etc.) by fitting a straight line to the data. The general form of the equation is:

y=β0+β1x1​+β2x2​+…+βn​xn
​
Key Concept: Linear regression finds the coefficients 𝛽 that minimize the sum of the squared differences between the observed values and the predicted values (minimizing the mean squared error). This model assumes that the relationship between inputs and outputs is linear, which is its main limitation when dealing with more complex or non-linear relationships.

Although Naive Bayes is typically used for classification tasks, it can be adapted for regression in some cases (such as Gaussian Naive Bayes). The model assumes that the input features are independent of each other (hence "naive") and follows a specific probability distribution, often Gaussian in regression.

The Gaussian Naive Bayes model calculates the likelihood of each feature given a certain output, using probability distributions to estimate the target variable (CHF) based on these likelihoods.

Key Concept: Naive Bayes is powerful when the independence assumption holds true but can be less accurate when the input features are strongly correlated. Unlike linear regression, which is purely deterministic, Naive Bayes incorporates probabilistic reasoning

Ridge Regression is a variant of linear regression that includes a regularization term to address overfitting, especially when there is multicollinearity (i.e., highly correlated features) in the dataset. The Ridge Regression model adds a penalty to the size of the coefficients, which is controlled by the regularization parameter 𝛼.

For Ridge Regression with α = 0.1, the regularization is relatively small, which means the model is closer to standard linear regression. It tries to strike a balance between fitting the data and limiting overfitting by penalizing large coefficient values.

Key Concept: This model is well-suited for situations where the input features are moderately correlated, and the goal is to improve the model’s generalization ability by shrinking the coefficients.

In Ridge Regression with α = 0.8, the regularization is stronger compared to α = 0.1. This means that the model places more emphasis on minimizing the size of the coefficients (shrinkage), which helps in preventing overfitting even more than with α = 0.1.

Key Concept: A larger α results in greater bias but lower variance, making the model more robust to overfitting at the cost of potentially underfitting the data. It is most useful when the data exhibits a high level of noise or when certain features may not be particularly important for prediction.


Each of these models were evaluated on the dataset, and their performance compared in terms of accuracy (mean square error), processing time, and memory usage, giving insights into which approach best balances complexity and predictive power for this particular task.


## Results




## Bibliography 
[1] D.C. Groeneveld, Critical Heat Flux Data Used to Generate the 2006 Groeneveld Lookup Tables (NUREG/KM-0011), (2019). 
https://www.nrc.gov/reading-rm/doc-collections/nuregs/knowledge/km0011/index.html#pub-info

