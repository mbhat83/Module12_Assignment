# Module 12 - Credit Risk Classification

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this Challenge, you’ll use various techniques to train and evaluate models with imbalanced classes. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.


### Instructions & Technology used

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The purpose of the analysis is to build a model that can predict the credit risk  and identify the creditworthiness of borrowers. With the oversampled data by making the risky loans data equal to that of the healthy loan data . currently this classification is imbalanced because of healthy loan has outnumbers risky loans 
    
* Explain what financial information the data was on, and what you needed to predict.
The financial information that loan_status data has healthy loan and high-risk loan as target('y') data which we want to predict and rest of information loan size, interest rate, income, debt_income_ratio, total debt as a features('X') will be use to make predictions.
 

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
                         Original imbalanced data  : oversampled data after increasing the data to  high-risk loans 
0(healthy loans)    -                    75036     :    56271
1(high-risk loans)  -                     2500     :    56271


* Describe the stages of the machine learning process you went through as part of this analysis.
- read the data from csv files stored in the folder.
- created a variables X (taking historically information of the lending data)and y(classified data as 2 outcomes healthy loan and high-risk loan) 
- spliting the data into train and test sets
- Creating the logistic regression model  
- Fitting the model into logistic regression with training data sets 
- Predicting the testing features data with model 
- evaluating the model performance  


* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
Usage -  
'`LogisticRegression` method - This method is used to predict on the dicrete outcomes and uses multiple variables. The logistic regression model analyzes the available data. When presented with a new sample of data, the model mathematically determines the probability of the sample belonging to a class. If the probability is greater than a certain cutoff point, the model assigns the sample to that class. If the probability is less than or equal to the cutoff point, the model assigns the sample to the other class

'`RandomOverSampler` method - This method is used where there is imbalanced label data by increasing the size of the minority labeled data. 


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
- Model 1 Accuracy score = 95%
- Model 1 precision = 1.00 for healthy loan and 0.85 for high-risk loan 
- Model 1 recall = 0.99 for healthy loan and 0.91 for high-risk loan

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
- Model 2 Accuracy score = 99%
- Model 2 precision = 1.00 for healthy loan and 0.84 for high-risk loan 
- Model 2 recall = 0.99 for healthy loan and 0.99 for high-risk loan

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best? 
The model 2 performed peform well with oversampled data in evaluating the credit risk 

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
It is important to predict the 1 as to evaluate the credit risk in issuing the high risk loans based on the available data of the borrowers. 

If you do not recommend any of the models, please justify your reasoning.

 
