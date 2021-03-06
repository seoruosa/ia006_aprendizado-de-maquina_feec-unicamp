# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette, JointGrid
from utils import df_to_supervised_classification
from sklearn.model_selection import train_test_split
import time


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



df = df_to_supervised_classification(data[:], 'label', 2, 1)

print(df.shape)

X = df[:].drop('label',axis=1)
Y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.3,
#    test_size=0.33, 
    random_state=333, stratify=Y)
#print(X_train.shape)
#print(Y.shape)


#from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# # Encode values to 0 and 1
# encoder = LabelEncoder()
# encoder.fit(y_train.ravel())
# y_train = encoder.transform(y_train.ravel())


# SVM
clf = SVC(gamma='auto', 
#          kernel='linear', 
          kernel='rbf',
          max_iter=200000,
          verbose=True)


start = time.time()
clf.fit(X_train, y_train.ravel())
end = time.time()
print("Training period: %.6f"%(end-start))


pred = clf.predict(X_train)

print("Accuracy(train): %.2f" % accuracy_score(y_train, pred))

start = time.time()
pred = clf.predict(X_test)
end = time.time()
print("Prediction period: %.6f"%(end-start))

# print("Accuracy(test): %.2f" % accuracy_score(encoder.transform(y_test.ravel()), pred))
print("Accuracy(test): %.2f" % accuracy_score(y_test, pred))

