# -*- coding: utf-8 -*-
"""
Created on Mon Sep 07 15:32:38 2015

@author: jbooth
"""

import numpy as np
import random 
import matplotlib.pyplot as plt

#Problem 1
A1 = [[1,2,3],[4,5,6]]
B1 = [[-2,1],[0,4],[2,6]]
C1 = [[2,7],[-1,6]]
E1 = [[0,1,1],[1,0,1],[1,1,0]]
F1 = [[2,-1,0,0,0],[-1,2,-1,0,0],[0,-1,2,-1,0],[0,0,-1,2,-1],[0,0,0,-1,2]]

A = np.array(A1)
B = np.array(B1)
C = np.array(C1)
E = np.array(E1)
F = np.array(F1)

#(a)
#add = A+B
#(b)
AB = np.dot(A,B)
#(c)
detC = np.linalg.det(C)
traceC = np.trace(C)
#(d)
Cinv = np.linalg.inv(C)
#(e)
EigE = np.linalg.eig(E)
#(f) Show that F is positive definite
#Write F as RDR^T. Then if diag of D are positive, then F is positive definite
def positivedefinite(matrix):
    eigenvalues = np.linalg.eig(matrix)[0]
    if eigenvalues.all > 0:
        print "This matrix is positive definite."
    elif eigenvalues.all >= 0:
        print "This matrix is non-negative definite."
    else:
        print "This matrix is not positive definite."

#(g)
#note, this only works for a matrix with 5 eigenvalues! 
def decompose(matrix):
    eigvals = np.linalg.eig(matrix)[0]
    D1 = [[eigvals[0],0,0,0,0],[0,eigvals[1],0,0,0],[0,0,eigvals[2],0,0],[0,0,0,eigvals[3],0],[0,0,0,0,eigvals[4]]]
    D = np.array(D1)
    R = np.linalg.eig(matrix)[1]
    firstcheck = np.dot(D, R.T)
    secondcheck = np.dot(R, firstcheck)
    return D, R, R.T, secondcheck

#_______________________________________________
#_______________________________________________
#2. (a)
#HISTOGRAMS
import matplotlib.mlab as mlab

position = 0
steps = 100
walks = 100
p = .65

s= np.random.binomial(walks,p,steps)
mu,sigma = np.mean(s), np.std(s)
walks,bins,patches = plt.hist(s,bins = 20, normed = True)
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins,y, 'r--', linewidth = 2)    

    