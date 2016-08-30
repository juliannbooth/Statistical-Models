# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:53:48 2015

@author: jbooth
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

#Use "header = None" to prevent the first row being used as the column names
data = pd.read_csv('http://faculty.tarleton.edu/crawford/documents/math505/Lab2Data.txt', header = None)
data.head()
np.shape(data)

#Make four vectors:

    
Y = data.iloc[:,0]
X1 = data.iloc[:,1:4]

X = sm.add_constant(X1)
est = sm.OLS(Y, X).fit()
est.summary()

#Problem 1: Write a function to find Beta hat given Y and X
def beta(X,Y):
    X = np.matrix(X)
    Y = np.matrix(Y)
    first = np.linalg.inv(X.T*X)
    second = first*X.T
    beta = second*Y.T
    return beta

def beta1(X,Y):
    X = np.matrix(X)
    Y = np.matrix(Y)
    beta = (np.linalg.inv(X.T*X))*X.T*Y.T
    return beta

beta2 = beta1(X,Y)
beta2
#Problem 2: the vector of residuals

def residual(X,Y, beta):
    X = np.matrix(X)
    Y = np.matrix(Y)
    e = Y.T - X*beta
    return e
    
residuals = residual(X,Y, beta2)
residuals

#Problem 3: estimate sigma hat
e = []
for i in range(0,99):
    e.append(1)
    
import math

def sigmahat(emptyvect, residuals, n, p):
    for i in range (0,99):
        emptyvect[i] = residuals[i]**2
    ressq = sum(emptyvect)
    sigmaest = math.sqrt(ressq / (n-p))
    return sigmaest
    
sigmahatest = sigmahat(e,residuals, 100, 3)

#Problem 4
def covhat(sigma2, X):
    X = np.matrix(X)
    cov = (sigma2**2) *(np.linalg.inv(X.T*X))
    return cov
    
estcov = covhat(sigmahatest, X)
