# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:50:39 2018

@author: pulki
"""

import numpy as np
import pandas as pd 

def covariance_matrix(X):
    mean_X = np.mean(X , axis = 0)
    return ( (X - mean_X).T @ (X - mean_X) )/len(X[0] - 1)

        
def pca(data):
    diff = data - np.mean(data , axis = 0)
    cov_mat = covariance_matrix(data)
    eigen_val , eigen_vec = np.linalg.eig(cov_mat)
    final_data = eigen_vec @ diff.T
    final_data = final_data.T
    return eigen_val , final_data
        

dataset = pd.read_csv("50_Startups.csv")
dataset = dataset.drop("State" , axis = 1)
dataset = np.array(dataset)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data = sc.fit_transform(dataset)

eig_val , pca_data = pca(data)


