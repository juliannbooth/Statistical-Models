# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:56:36 2015

@author: jbooth

Lab Chapter 4 - Statistical Models
"""

#1(a).
x = [20,16,19.8,18.4,17.1,15.5,14.7,17.1,15.4,16.2,15,17.2,16,17,14.4]
y = [88.6,71.6,93.3,84.3,80.6,75.2,69.7,82,69.4,83.3,79.6,82.6,80.6,83.5,76.3]


#Using previous knowledge to double check the values of b0 and b1 :)

def square(list):
    return [i**2 for i in list]
    

def multiplytwo(list1, list2):
    store = []
    i = 0
    while i < len(list1):
        store.append(list1[i] * list2[i])
        i += 1
    return store
    
sumx = sum(x)
sumy = sum(y)   
xsq = sum(square(x))
ysq = sum(square(y))
xandy = sum(multiplytwo(x,y))

n = len(x)
SSxx = xsq - (sumx**2/n)
SSxy = xandy - (sumx*sumy/n)
SSyy = ysq - (sumy**2/n)

b1 = SSxy/SSxx
b0 = (sumy/n) - b1*(sumx/n)

#OLS Estimator:
Betahat = [b1,b0] #[25.2323, 3.29109]
#___________________________________________________________________
#using book / multiple regression way:
import numpy as np

x1 = []
for i in x:
    x1.append(1)

Yt = np.matrix([y])
Y = Yt.T
Xt = np.matrix([x1,x]) #note this is 2x15 (the transpose)
X = Xt.T
XtXinv = np.linalg.inv(Xt*X)
XtXinvXt = XtXinv * Xt
#OLS Estimator. Same as above :)
B = XtXinvXt * Y
#--------------------------------------------------------------------

#(b)
import matplotlib.pyplot as plt

regressx = [14, 15,16,17,18,19,20]
     
#a function to give the predicted y values to plot
def regressyfunc(b0,b1,x):
    regressy = []
    i = 0
    while i < len(x):
        entry = b0 + b1*x[i]
        regressy.append(entry)
        i+= 1
    return regressy


regressy = regressyfunc(b0,b1,regressx)
test = regressyfunc(b0,b1,x)


plt.plot(x,y, 'ro')
plt.plot(regressx, regressy)
plt.plot()
plt.ylabel('Temperature')
plt.xlabel('Chirps per Second')
plt.show()

#-----------------------------------------------------------
#(c) What is the estimate for cov(B|X)?
#calculate residuals which are actual y values - our predicted y values
e = Y - X*B
n=15
p=2
#note, these need to be on one line but I put them on multiple lines to print
esqsum = e[0]**2 + e[1]**2 + e[2]**2 + e[3]**2 + e[4]**2 + e[5]**2 + 
 e[6]**2 + e[7]**2 + e[8]**2+ e[9]**2 + e[10]**2 + e[11]**2 +
  e[12]**2 + e[13]**2 + e[14]**2
esqsum = 190.54734318
sigmasqhat = esqsum / (n-p)   
Covhat = sigmasqhat*XtXinv

#--------------------------------------------------------
#(d) residuals on vertical axis and x on horizontal axis
lx = [14,15,16,18,20]
ly = [0,0,0,0,0]
plt.plot(x,e, 'ro')
plt.plot(lx,ly) #horizontal line at y=0
plt.ylabel('residuals')
plt.xlabel('X')
plt.show()

#-------------------------------------------------------
#(e) create a lag plot for the residuals
lx = [-8, -4, 0, 2,6]
ly = [0,0,0,0,0]
ei = e[0:14,]
ei1 = e[1:15,]
plt.plot(ei,ei1, 'ro')
plt.plot(lx,ly) #horizontal line at y = 0
plt.ylabel('e_(i+1)')
plt.xlabel('e_i')
plt.show()

#-----------------------------------------------------
#(f) Compute R^2
R2 = (np.var(X*B))/(np.var(Y))
#0.697465