# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]]#trainning data
lables = [0,0,1,1]#the superviser answer
#clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier().fit(features,lables)
#clf = clf.fit(features,lables) #trainning with features and lables
if clf.predict([[150,0]]) == [1]:
    print("orange")
else:
    print('apple')
#print(clf.predict([[150,0]]))


