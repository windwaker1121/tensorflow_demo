#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:31:30 2017

@author: work
"""
import sklearn as sk
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data #input
y = iris.target #output

from sklearn.cross_validation import train_test_split #匯入分資料函式
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

#import 分類器
#myCLF = sk.neighbors.KNeighborsClassifier() #asign classifier 
myCLF = sk.tree.DecisionTreeClassifier()

myCLF.fit(x_train, y_train) #trainning

predictions = myCLF.predict(x_test) #prediction
print(predictions[:50])
print(y_test[:50])
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))






