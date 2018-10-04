# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 18:04:15 2018

@author: pulki
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
def bucketize(dataset , col_index , cat = 10):
    feat = dataset.iloc[:,col_index].values
    low =min(feat)
    gap = int((max(feat) - low)/cat)
    for i in range(len(feat)):
        if(((feat[i]-low)/gap)+1 > cat):
            feat[i] = cat
        else:
            feat[i] = ((feat[i]-low)/gap)+1
    dataset.iloc[:,col_index] = feat
    return dataset

def label(dataset , a):
    for col in a:
        cat = []
        for i in range(len(dataset)):
            element = dataset.iloc[i , col]
            if element not in cat:
                cat.append(element)
            dataset.iloc[i , col] = cat.index(element) + 1
    return dataset

def entropy_calc(attribute):
    val,freq = np.unique(attribute , return_counts = True)
    rel_freq = freq/len(attribute)
    return -rel_freq.dot(np.log2(rel_freq))

def info_gain_calc(attribute , dv):
    val,freq = np.unique(attribute , return_counts = True)
    feat = dict(zip(val, freq))
    info_gain = float(0)
    for at_val,at_freq in feat.items():
        info_gain += at_freq * entropy_calc(dv[attribute == at_val]) 
    info_gain = entropy_calc(dv) - (info_gain/len(attribute))
    return info_gain 

    
def decision_tree(data,originaldata,features,dv,parent_node_class = None):
    #first we check the three termisnation criteria
    #1. if all labels in the dataset are of same value, then make a leaf node with that value
    if len(np.unique(data[dv])) <= 1:
        return np.unique(data[dv])[0]
    #2. If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data)==0:
        return np.unique(originaldata[dv])[np.argmax(np.unique(originaldata[dv],return_counts=True)[1])]
    #3. if the attribute_list is empty then return to the parent node
    elif len(features) ==0:
        return parent_node_class
    #If none of the terminating criteria holds true than grow the tree further
    else:
        parent_node_class = np.unique(data[dv])[np.argmax(np.unique(data[dv],return_counts=True)[1])]
        #Select the feature which best splits the dataset
        item_values = [info_gain_calc(data[feature] , data[dv]) for feature in features] 
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature:{}}
        features = [i for i in features if i != best_feature]
        #now growing the children nodes
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = decision_tree(sub_data,dataset,features,dv,parent_node_class)
            tree[best_feature][value] = subtree
        return(tree)   
    
def predict(query,tree,default = 1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
            

def prediction(data,tree , dv):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    return predicted

#importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
dataset = dataset.drop("User ID" , 1)

#preprocessing the data
dataset = bucketize(dataset , 1)
dataset = bucketize(dataset , 2)
dataset = label(dataset ,[0] )
test = dataset.iloc[320: , :]
dataset = dataset.iloc[:320 , :]

#calculating entropy for out dv
Entropy = entropy_calc(dataset.iloc[:,-1].values)

#info_gain for different attributes
g1 = info_gain_calc(dataset.iloc[:,2].values, dataset.iloc[:,-1].values )
g2 = info_gain_calc(dataset.iloc[:,1].values, dataset.iloc[:,-1].values )
g3 = info_gain_calc(dataset.iloc[:,0].values, dataset.iloc[:,-1].values )

tree = decision_tree(dataset , dataset , dataset.columns[:-1] , 'Purchased')
pprint(tree)
y_pred = prediction(test , tree , "Purchased")
y_pred = np.array(y_pred).astype(int)
y = test.iloc[: ,-1].values

#checking the performance of our model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)
correct =0
for i in range(len(y_pred)):
    if(y[i] == y_pred[i]):
        correct+=1
print('The prediction accuracy is: ',correct/len(y_pred),'%')
        
    