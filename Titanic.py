#   
#   Titanic Passenger Survival Classification
#   Dataset: https://www.kaggle.com/c/titanic/data
#   Author: Mario Dellaviola
#   Date: July 24, 2018
##################################################

import os
import logging, sys
import numpy as np
import re
from sklearn.naive_bayes import GaussianNB

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logging.debug('A debug message!')

os.chdir('.\Titanic')

def LoadData():
    trainFile = input('Input Training Data Filename: ') or 'train.csv'
    testFile = input('Input Testing Data Filename: ') or 'test.csv'
    train = open(trainFile)
    test = open(testFile)
    return train, test

def ProcessData(train, test):
    featureLabel = np.array(train.readline().rstrip().split(','))                  #featureLabel contains name of features
    trainArray = []           
    for x in train:
        trainArray.append(x.rstrip().split(','))
        #trainResponse_arr = np.array(trainResponse)
        #trainPredict_arr = np.array(trainPredict) 
    catPredictors = [3,4,5,9,11,12]
    conPredictors = [0,2,6,7,8,10]
    train_Arr = np.array(trainArray)
    trainResponse = np.array([train_Arr[:,1]])
    trainPredict_cat = np.array([train_Arr[:,i] for i in catPredictors])
    trainPredict_con = np.array([train_Arr[:,j] for j in conPredictors])
    np.insert(featureLabel, 2, 'surname')
    return np.transpose(trainResponse), np.transpose(trainPredict_cat), np.transpose(trainPredict_con), featureLabel  


trainData, testData = LoadData()
trainClass, trainPredict_cat,trainPredict_con, featureLabel = ProcessData(trainData, testData)


gnb_cat = GaussianNB()
gnb_con = GaussianNB()
gnb_cat.fit(trainPredict_cat[:,0].reshape(-1,1).astype(np.string),np.squeeze(trainClass))
    

#Test Outputs
print(featureLabel.shape, trainClass.shape, trainPredict_cat.shape)
print(featureLabel[0])
print(trainClass[0])