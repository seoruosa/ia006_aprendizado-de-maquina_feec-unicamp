#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 00:02:54 2019

@author: mac
"""


import time
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from utils import *


folderpath = '/Users/mac/Documents/code/ia006_aprendizado-de-maquina_feec-unicamp/EFC/Projeto_code'

filepath_ = '../MHEALTHDATASET/mHealth_subject1.log'
columns_name = ['acc_chest_x', 'acc_chest_y', 'acc_chest_z', 
                'elec_sig_1', 'elec_sig_2', 
                'acc_left_ank_x', 'acc_left_ank_y', 'acc_left_ank_z',
                'gyro_left_ank_x', 'gyro_left_ank_y', 'gyro_left_ank_z',
                'magneto_left_ank_x', 'magneto_left_ank_y', 'magneto_left_ank_z',
                'acc_rig_arm_x', 'acc_rig_arm_y', 'acc_rig_arm_z',
                'gyro_rig_arm_x', 'gyro_rig_arm_y', 'gyro_rig_arm_z',
                'magneto_rig_arm_x', 'magneto_rig_arm_y', 'magneto_rig_arm_z',
                'label']


# Train SVM with 30% of the data using RBF kernel
clf_name='SVM_30p_rbf'
n_in = 3

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.3)

train_results, scaler, clf = SVM_model_train_test(X_train, X_test, y_train, y_test, 200000, 
                                                  kernel_='rbf', verbose_=True)
save_tests_model(clf_name, clf,train_results)


# Train SVM with 30% of the data using Linear kernel
clf_name='SVM_30p_linear'
n_in = 3

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.3)

train_results, scaler, clf = SVM_model_train_test(X_train, X_test, y_train, y_test, 200000, 
                                                  kernel_='linear', verbose_=True)
save_tests_model(clf_name, clf,train_results)


# Train SVM with 30% of the data using RBF kernel
clf_name='SVM_50p_rbf_5_lags'
n_in = 5

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.5)

train_results, scaler, clf = SVM_model_train_test(X_train, X_test, y_train, y_test, 100000, 
                                                  kernel_='rbf', verbose_=True)
save_tests_model(clf_name, clf,train_results)


# Train SVM with 30% of the data using Linear kernel
clf_name='SVM_50p_linear_5_lags'
n_in = 5

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.5)

train_results, scaler, clf = SVM_model_train_test(X_train, X_test, y_train, y_test, 100000, 
                                                  kernel_='linear', verbose_=True)

save_tests_model(clf_name, clf,train_results)


################################## Train with less data with more iterations


# Train SVM with 30% of the data using RBF kernel
clf_name='SVM_90p_rbf_5_lags'
n_in = 5

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.9)

train_results, scaler, clf = SVM_model_train_test(X_train, X_test, y_train, y_test, 300000, 
                                                  kernel_='rbf', verbose_=True)
save_tests_model(clf_name, clf,train_results)


# Train SVM with 30% of the data using Linear kernel
clf_name='SVM_90p_linear_5_lags'
n_in = 5

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.9)

train_results, scaler, clf = SVM_model_train_test(X_train, X_test, y_train, y_test, 300000, 
                                                  kernel_='linear', verbose_=True)
save_tests_model(clf_name, clf,train_results)


################
#  DECISION TREE
################

clf_name='decisiontree_50p_3_lags_md_3'
n_in = 3
max_depth=3

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.5)

train_results, scaler, clf = decision_tree_model_train_test(X_train, X_test, y_train, y_test, max_depth)
save_tests_model(clf_name, clf,train_results)

###########

clf_name='decisiontree_50p_5_lags_md_3'
n_in = 5
max_depth=3

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.75)

train_results, scaler, clf = decision_tree_model_train_test(X_train, X_test, y_train, y_test, max_depth)
save_tests_model(clf_name, clf,train_results)

#################

clf_name='decisiontree_50p_3_lags_md_5'
n_in = 3
max_depth=5

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.5)

train_results, scaler, clf = decision_tree_model_train_test(X_train, X_test, y_train, y_test, max_depth)
save_tests_model(clf_name, clf,train_results)

############

clf_name='decisiontree_50p_5_lags_md_5'
n_in = 5
max_depth=5

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.75)

train_results, scaler, clf = decision_tree_model_train_test(X_train, X_test, y_train, y_test, max_depth)
save_tests_model(clf_name, clf,train_results)


#####
#  KNN
######

clf_name='KNN_70p_3_lags_neigh_3'
n_in = 3
n_neighbors=3

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.7)

train_results, scaler, clf = KNN_model_train_test(X_train, X_test, y_train, y_test, n_neighbors)
save_tests_model(clf_name, clf,train_results)

######################3

clf_name='KNN_70p_5_lags_neigh_1'
n_in = 3
n_neighbors=1

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.7)

train_results, scaler, clf = KNN_model_train_test(X_train, X_test, y_train, y_test, n_neighbors)
save_tests_model(clf_name, clf,train_results)


#########

clf_name='KNN_95p_3_lags_neigh_1'
n_in = 3
n_neighbors=1

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.95)

train_results, scaler, clf = KNN_model_train_test(X_train, X_test, y_train, y_test, n_neighbors)
save_tests_model(clf_name, clf,train_results)

#########

clf_name='KNN_99p_3_lags_neigh_1'
n_in = 3
n_neighbors=1

X_train, X_test, y_train, y_test = train_test_sup_classification(filepath_, 
                                                                 columns_name, 
                                                                 n_in, 
                                                                 0.99)

train_results, scaler, clf = KNN_model_train_test(X_train, X_test, y_train, y_test, n_neighbors)
save_tests_model(clf_name, clf,train_results)