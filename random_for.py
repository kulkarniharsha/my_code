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
from sklearn.ensemble import RandomForestRegressor 


"""We will import data from "final.xlsx" into the Pandas dataframe df."""

df = pd.read_excel("GH_Dataset.xlsx", sheetname="GH", skiprows=1)

"""We want to use "Vg", "Vl", "Molwt_gas", "Rho_gas","sparger","Sparger_hole_diameter",
"Number_of_sparger_holes", "Rho_liq"and,"Tens_liq_",
"Ionic_strength_of_the_electrolyte_solution_","Temp_","Press","H"and"D" 
to predict "OverallGasHoldup".The former 
we club in a list called Xvars and the latter is Yvar."""

Xvars = ["Vg","Vl","Molwt_gas","Rho_gas","Sparger","Sparger_hole_dia","Number_of_sparger_holes","Rho_liq","Tens_liq_","Ionic_strength_of_the_electrolyte_solution_","Temp_","Press","H","D"]
Yvar = "OverallGasHoldup"
regr = RandomForestRegressor(random_state=0,n_estimators=100)
regr.fit(Xvars, Yvar)
