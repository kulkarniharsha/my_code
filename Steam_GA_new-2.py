#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 12:21:28 2018

@author: harsha
"""
# min 	A1*(X(2)-X(1)^2)^2 + (A2-x(1))^2
#    s.t.:	X(1)^2 + X(2)^2 - A2 <= 0
#            -1.0 <= xi <= 1.0,  i = 1,2
from pyOpt import Optimization
from pyOpt import NSGA2

# Make the objective function
def objfunc(x):
    f=3.32*(1/x[0]-1.21/(x[0]*x[1])**2.12)#objective function
#    ff = 2*[0.0]
#    ff[0] = (x - 0.0)**2 + (y - 0.0)**2
#    ff[1] = (x - 1.0)**2 + (y - 1.0)**2
    g = [] # constraned equation
    fail=False
    
    return f, g, fail


# Setup the optimisation problem
opt_prob = Optimization('Steam_ejector', objfunc)
#opt_prob.addVar('x', 'c', value=10, lower=-10, upper=10)
opt_prob.addVar('x1', 'c', value=3.0, lower=0.0, upper=6.0)
opt_prob.addVar('x2', 'c', value=6.0, lower=2.4, upper=10.0)
opt_prob.addObj('f')
print(opt_prob)

options = {
    'PopSize':300, #	Population Size (a Multiple of 4)
    'maxGen' :1000, # Maximum Number of Generations
    'pCross_real':0.3, #Probability of Crossover of Real Variable (0.6-1.0)
    'pMut_real':0.2, #Probablity of Mutation of Real Variables (1/nreal)
    'eta_c':10.0, 	#Distribution Index for Crossover (5-20) must be > 0
    'eta_m':20.0, 	#Distribution Index for Mutation (5-50) must be > 0
    'pCross_bin':0.0, #Probability of Crossover of Binary Variable (0.6-1.0)
    'pMut_bin':0.0, 	#Probability of Mutation of Binary Variables (1/nbits)
    'PrintOut' :2,   #Flag to Turn On Output to files (0-None, 1-Subset, 2-All)
    'seed':0.0	    #Random Number Seed (0 - Auto-Seed based on time clock)
     }
nsg2 = NSGA2()
nsg2.setOption('PrintOut',2)
nsg2(opt_prob,options)
print(opt_prob.solution(0))
