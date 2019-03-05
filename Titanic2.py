#   
#   Titanic Passenger Survival Classification
#   Dataset: https://www.kaggle.com/c/titanic/data
#   Author: Mario Dellaviola
#   Date: July 30, 2018
##################################################
import logging, sys, re, os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logging.debug('A debug message!')
os.chdir('.\Titanic')

class Data(object):

    def  __init__(self):
        trainFile = input('Input Training Data Filename: ') or 'train.csv'
        testFile = input('Input Testing Data Filename: ') or 'test.csv'
        labelFile = input('Input Label Filename: ') or 'gender_submission.csv'
        self._train = pd.read_csv(trainFile)
        self._test = pd.read_csv(testFile)
        self._label = pd.read_csv(labelFile)
        self.train_all = self._train
        self.train_con = self._train
        self.train_cat = self._train
        self.test_all = self._test
        self.test_con = self._test
        self.test_cat = self._test

        self.respInd = input('Response Column: ') or 'Survived'


    def dataMethod(self, argument):                                                                 #selects which data processing method to use given a string
        method = getattr(self, argument, lambda: "Invalid_Method")
        return method()

#try method of removing all features with 
    def removeFeat(self):                                                                           #'removeFeat' removes features with incomplete data in either set
        print('Selected remove method:')
        self.train_all = self._train[self._train.Ticket != 'LINE']                                  #toss observations with inconsistent entries
        self.train_all['Ticket'] = self.train_all['Ticket'].str.split(' ').str[-1].astype(float)
        self.test_all['Ticket'] = self._test['Ticket'].str.split(' ').str[-1].astype(float)         #keep only numeric (2nd part) of ticket number                                           
        trueInd = self._train.notnull().all() & self._test.notnull().all()                          #keep only complete features
        print('Using features: ',  )

        self.response = self.train_all.ix[:,self.respInd]
        self.train_cat = self.train_all.loc[:, trueInd].select_dtypes(include=['object'])
        self.train_con = self.train_all.loc[:, trueInd].select_dtypes(exclude=['object'])
    
    def getResponse(self):
        return self.response

    def getPredictors(self, partition):
        return getattr(self,partition + '_cat'), getattr(self,partition + '_con')
    
    def DatasetMetrics(self):                                 #data information function
 #   print('Observations in test data: ', test.shape[0])
 #   print('Observations in training data: ', train.shape[0])
        print('% Null in test data:\n', self._test.isnull().sum()/self._test.shape[0]*100)
        print('% Null in training data:\n',  self._train.isnull().sum()/self._train.shape[0]*100)
        print(self._train.info())
        print(self._test.info())


#train two gaussian naive bayesian classifer for each categorical and continuous data
#cat: categorical training data
#con: continous training data
#repsonse: class labels, using titanic this is 1 or 0 for survived/notsurvived
#method: several methods are implemented for comparison
#           'remove' removes incomplete features from training data
#           'awdawd'
#           'adwd'
def TrainClassifier(d):   
    ComputeBNBcat = BernoulliNB()
    ComputeGNBcon = GaussianNB()
    ComputeBNBcat.fit(d.getPredictors('train')[0],d.getResponse())
    ComputeGNBcon.fit(d.getPredictors('train')[1],d.getResponse())
    return ComputeGNBcat, ComputeGNBcon

def Classification(TrainingModel_cat,TrainingModel_con,d):
    insampleLabel_cat = TrainingModel_cat.predict(d.getPredictors('train')[0])
    catProb = TrainingModel_cat.predict_proba(d.getPredictors('train')[0])

    insampleLabel_con = TrainingModel_con.predict(d.getPredictors('train')[1])
    conProb = TrainingModel_con.predict_proba(d.getPredictors('train')[1])

    return(catProb, conProb)

d = Data()
d.dataMethod('removeFeat')
d.DatasetMetrics()
a,b = TrainClassifier(d)
c1,c2 = Classification(a,b,d)
