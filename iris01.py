#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:32:23 2017

@author: work
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris=load_iris()
print(iris.feature_names)#印出所有特徵
print(iris.target_names)#印出總共的結果
print(iris.data[0])#印出第一筆資料
print(iris.target[0])#印出第一種結果
test_idx = [0,50,100]
#把0,50,100拿掉剩下的資料拿來做訓練
train_data = np.delete(iris.data,test_idx,axis=0)
train_target = np.delete(iris.target,test_idx)
#把拿掉的那三個拿來做驗證
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

clf = tree.DecisionTreeClassifier().fit(train_data,train_target)
print(test_target)
print(clf.predict(test_data))
