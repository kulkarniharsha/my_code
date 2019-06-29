#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 19:29:28 2018

@author: harsha
"""

# Pandas is used for data manipulation
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 5000, random_state = 55)

# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

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

# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
###########################################
#print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances;
###########################################

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Extract the two most important features
important_indices = [feature_list.index('Vg'), feature_list.index('Vl')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]

# Train the random forest
rf_most_important.fit(train_important, train_labels)

# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)

errors = abs(predictions - test_labels)

# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape

print('Accuracy:', round(accuracy, 2), '%.')
# Import matplotlib for plotting and use magic command for Jupyter Notebooks
#import matplotlib.pyplot as plt

#% matplotlib inline figure plotting
plt.figure(1)
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

plt.figure(2)
plt.plot(test_labels, predictions, 'ro')
plt.plot([0, max(predictions)],[0, max(predictions)], 'b')
plt.show()

"""Let's get a correlation:"""
import scipy
r, p = scipy.stats.pearsonr(test_labels, predictions)
"""Percent correlation is:"""
r_sqr=r**2*100
print (r)
print(r_sqr)
