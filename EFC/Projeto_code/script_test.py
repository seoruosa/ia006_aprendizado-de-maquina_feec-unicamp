#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 01:43:38 2019

@author: mac
"""

filepath = lambda x: "./MHEALTHDATASET/mHealth_subject%d.log"%x
clf_name = 'SVM_20p_rbf'

#if 'testes' not in vars() or 'testes' in globals():

testes = []
for i in range(2,11):
    data = pd.read_csv(filepath(i), sep='\t', skipinitialspace=True, header=None) #parse_dates=[0]
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
    
    X = df[:].drop('label',axis=1)
    Y = df['label']
    
    X = scaler.transform(X)
    
    start = time.time()
#    pred = clf.predict(X)
    pred = clf.predict(X)
    end = time.time()
    testes.append({'clf':clf_name,
#            'clf':'DecisionTree',
                   'dataset':i,
                   'acc':accuracy_score(Y, pred),
                   'predict_time':(end-start),
                   'CM':confusion_matrix(Y,pred),
                   'f1_score':f1_score(Y,pred)})
    print("Accuracy(%d): %.2f   -   time: %.6f" % (i,accuracy_score(Y, pred),(end-start)))

#print(confusion_matrix(Y, pred))
