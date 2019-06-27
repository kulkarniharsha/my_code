# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

"""Playing with Harshvardhan's SVM"""
""" Lets see what we have."""

import scipy
import pandas as pd
import matplotlib.pyplot as plt
import sklearn, sklearn.svm
import random

"""We will import data from "final.xlsx" into the Pandas dataframe df."""

df = pd.read_excel("final.xlsx", sheetname="Sheet1", skiprows=1)

"""We want to use "Temperature", "pH", "CrContent", "DO" and "FluidVelocity" 
to predict "MassTransferCoefficient".The former 
we club in a list called Xvars and the latter is Yvar."""

Xvars = ["temperature","pH","CrContent","DO","FluidVelocity"]
Yvar = "MassTransferCoefficient"

"""We will have to divide the dataset into training and validation sets.  
The training set is 70% of the data and the validation is the rest.  
We will sample randomly from the data for the training set."""

list_indices = scipy.array(list(df.index))

frac_train = 0.9
n_train = int(len(df)*frac_train)

training_indices = random.sample(list(list_indices), n_train) #Sample n_train items from list_indices without replacement
validation_indices = scipy.array(list(set(list_indices) - set(training_indices)))  #Subtracted two sets
len(validation_indices), len(training_indices)

"""Now that we have the indices, lets split the dataset into training and validation sections.
   Fit the SVM on the training section and test the fit on the validation section."""
   
df_training = df.iloc[training_indices]
df_validation = df.iloc[validation_indices]

Xvars_training = df_training[Xvars].as_matrix()
Yvar_training = df_training[Yvar].as_matrix()

Xvars_validation = df_validation[Xvars].as_matrix()
Yvar_validation = df_validation[Yvar].as_matrix()

"""Now lets get the SVM """

clf = sklearn.svm.SVR(C = 100.0, gamma = 0.0009, epsilon = 0.00089, kernel='rbf') #make our SVM object. Call it whatever you want
"""Lets train it on the training set """

clf.fit(Xvars_training, Yvar_training)

"""And now lets see how well it fits."""

Yvar_pred = clf.predict(Xvars_validation)
plt.plot(Yvar_validation, Yvar_pred, 'ro')
plt.plot([0, max(Yvar_validation)],[0, max(Yvar_validation)], 'b')
plt.show()

"""Let's get a correlation:"""
r, p = scipy.stats.pearsonr(Yvar_validation, Yvar_pred)
"""Percent correlation is:"""
r_sqr=r**2*100
print (r)
print(r_sqr)

