#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 01:04:17 2019

@author: mac
"""

import pandas as pd
from utils import df_to_supervised_classification

filepath = './MHEALTHDATASET/mHealth_subject1.log'

data = pd.read_csv(filepath, sep='\t', skipinitialspace=True, header=None) #parse_dates=[0]
data.columns = ['acc_chest_x', 'acc_chest_y', 'acc_chest_z', 
                'elec_sig_1', 'elec_sig_2', 
                'acc_left_ank_x', 'acc_left_ank_y', 'acc_left_ank_z',
                'gyro_left_ank_x', 'gyro_left_ank_y', 'gyro_left_ank_z',
                'magneto_left_ank_x', 'magneto_left_ank_y', 'magneto_left_ank_z',
                'acc_rig_arm_x', 'acc_rig_arm_y', 'acc_rig_arm_z',
                'gyro_rig_arm_x', 'gyro_rig_arm_y', 'gyro_rig_arm_z',
                'magneto_rig_arm_x', 'magneto_rig_arm_y', 'magneto_rig_arm_z',
                'label']


from sklearn.model_selection import train_test_split

df = df_to_supervised_classification(data[:], 'label', 2, 1)

print(df.shape)

X = df[:].drop('label',axis=1)
Y = df['label']
# print(X.head())


X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.9,
#    test_size=0.33, 
    random_state=333, stratify=Y)
print(X_train.shape)
#print(Y.shape)

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3) #5
clf.fit(X_train, y_train.ravel()) 

pred = clf.predict(X_train)

print("Accuracy(train): %.2f" % accuracy_score(y_train, pred))

pred = neigh.predict(X_test)

# print("Accuracy(test): %.2f" % accuracy_score(encoder.transform(y_test.ravel()), pred))
print("Accuracy(test): %.2f" % accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
