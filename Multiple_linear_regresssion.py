import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))
                  

#gradient descent
def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    return theta,cost

def predict(x_test , weight):
    return (weight @ x_test.T).T


dataset = pd.read_csv("50_Startups.csv")
dataset = dataset.drop('State' , axis = 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset = sc.fit_transform(dataset)
X = dataset[:45 , :3]
Y = dataset[:45 , -1:]
X_test = dataset[45: , :3]
Y_test = dataset[45: , -1:]

weight = np.random.rand(1,3)
weight,cost = gradientDescent(X ,Y,weight,4000,0.001)

y_pred = predict(X_test , weight)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(Y_test, y_pred))

