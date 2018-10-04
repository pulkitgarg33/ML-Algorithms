# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 16:34:06 2018

@author: pulki
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean(values):
    return (sum(values) / float(len(values)))

def variance(values):
    return sum([ (x-mean(values))**2  for x in values ])

def covariance(x ,y, mean_x , mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x)*(y[i] - mean_y)
    return covar

        
#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

#calculating the coefficients
m = covariance(x,y,mean(x),mean(y)) / variance(x)
c = mean(y) - m*mean(x)

plt.plot(x,(m*x + c) ,'red')
plt.scatter(x  , y)
plt.show()

