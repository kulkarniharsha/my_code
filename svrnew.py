#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 19:29:28 2018

@author: harsha
"""

# Pandas is used for data manipulation
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import random
# Read in data and display data from "final.xlsx"
features = pd.read_csv('GH_Dataset.csv')
features.head(1751)
print('The shape of our features is:', features.shape)
#The shape of our features is: (348, 9)
# Descriptive statistics for each column
print('Statistic of each column:',features.describe())
# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(features['OverallGasHoldup'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('OverallGasHoldup', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
features_new=np.delete(features,np.s_[16:1685],axis=1)
features_New=np.delete(features_new,[1,15,16],axis=1)
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features_New, labels, test_size = 0.35, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#####################################################
## The baseline predictions are the historical averages #error to check
#baseline_preds = test_features[:, feature_list.index('')]
#
## Baseline errors, and display average baseline error
#baseline_errors = abs(baseline_preds - test_labels)
#
#print('Average baseline error: ', round(np.mean(baseline_errors), 2))

######################################################

# Import the model we are using
#from sklearn.ensemble import RandomForestRegressor
import sklearn, sklearn.svm
"""Now lets get the SVM """

clf = sklearn.svm.SVR(C=10000.0,gamma =0.0001,epsilon=0.0002, kernel='rbf') #make our SVM object. Call it whatever you want
"""Lets train it on the training set """
# Train the model on training data
clf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = clf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# Import tools needed for visualization

############################################
#from sklearn.tree import export_graphviz
#import pydot
#
## Pull out one tree from the forest
#tree = rf.estimators_[5]
#
## Export the image to a dot file
#export_graphviz(tree, out_file = 'tree.dot', feature_names = Value, rounded = True, precision = 1)
#
## Use dot file to create a graph
#(graph, ) = pydot.graph_from_dot_file('tree.dot')
#
## Write graph to a png file
#graph.write_png('tree.png')
#################################################

clf.fit(test_features.shape, test_labels.shape)

"""And now lets see how well it fits."""

Yvar_pred = clf.predict(test_features.shape)
plt.plot(test_labels.shape, Yvar_pred, 'ro')
plt.plot([0, max(test_labels.shape)],[0, max(test_labels.shape)], 'b')
plt.show()

"""Let's get a correlation:"""
r, p = scipy.stats.pearsonr(test_labels.shape, Yvar_pred)
"""Percent correlation is:"""
r_sqr=r**2*100
print (r)
print(r_sqr)


#plt.figure(2)
#plt.plot(test_labels, predictions, 'ro')
#plt.plot([0, max(predictions)],[0, max(predictions)], 'b')
#plt.show()

#"""Let's get a correlation:"""
#import scipy
#r, p = scipy.stats.pearsonr(test_labels, predictions)
#"""Percent correlation is:"""
#r_sqr=r**2*100
#print (r)
#print(r_sqr)
