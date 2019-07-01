from pandas import DataFrame
from pandas import concat
 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if (type(data) is list or len(data.shape)==1) else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np

def df_to_supervised_df(df_input, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as Pandas Dataframe.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    columnsOriginal = df_input.columns[:].values
    data = df_input.values
    n_vars = 1 if (len(data.shape)==1) else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (columnsOriginal[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (columnsOriginal[j])) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (columnsOriginal[j], i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def df_to_supervised_classification(data, nameClassColumn, n_in_=1, n_out_=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset for a classification problem
    Arguments:
        data: Sequence of observations as Pandas Dataframe.
        nameClassColumn: name of the column with the labels of timeseries 
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    
    X = data[:].drop(nameClassColumn,axis=1)
    Y = data[nameClassColumn]
    
    X = df_to_supervised_df(X, n_in_, n_out_, False)
    X[nameClassColumn] = Y
    
    if dropnan:
        X.dropna(inplace=True)
    return X

from sklearn.model_selection import train_test_split

def train_test_sup_classification(filepath, columns_name, n_in, test_size_):
    """
    Arguments:
        n_in: Number of lag observations as input (X).
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    
    data = pd.read_csv(filepath, sep='\t', skipinitialspace=True, header=None) #parse_dates=[0]
    data.columns = columns_name
    
    df = df_to_supervised_classification(data[:], 'label', n_in, 1)
    
    print(df.shape)
    
    X = df[:].drop('label',axis=1)
    Y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=test_size_,
        random_state=333, 
        stratify=Y)
    return (X_train, X_test, y_train, y_test)

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import time

def SVM_model_train_test(X_train, X_test, y_train, y_test, max_iter_, kernel_='rbf', verbose_=True):
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    # SVM
    clf = SVC(gamma='auto',  
              kernel=kernel_,
              max_iter=max_iter_,
              verbose=verbose_)
    
    
    start = time.time()
    clf.fit(X_train, y_train.ravel())
    end = time.time()
    print("Training period: %.6f"%(end-start))
    
    
    pred = clf.predict(X_train)
    
    print("Accuracy(train): %.2f" % accuracy_score(y_train, pred))
    
    start = time.time()
    pred = clf.predict(X_test)
    end = time.time()
    period_train = end-start
    print("Prediction period: %.6f"%period_train)
    
    # print("Accuracy(test): %.2f" % accuracy_score(encoder.transform(y_test.ravel()), pred))
    print("Accuracy(test): %.2f" % accuracy_score(y_test, pred))
    train_results = {
                       'acc':accuracy_score(y_test, pred),
                       'predict_time':period_train,
                       'CM':confusion_matrix(y_test,pred)}
    return (train_results, scaler, clf)

from sklearn.neighbors import KNeighborsClassifier
def KNN_model_train_test(X_train, X_test, y_train, y_test, n_neighbors_):
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = KNeighborsClassifier(n_neighbors=n_neighbors_)
    
    
    start = time.time()
    clf.fit(X_train, y_train.ravel())
    end = time.time()
    print("Training period: %.6f"%(end-start))
    
    
    pred = clf.predict(X_train)
    
    print("Accuracy(train): %.2f" % accuracy_score(y_train, pred))
    
    start = time.time()
    pred = clf.predict(X_test)
    end = time.time()
    period_train = end-start
    print("Prediction period: %.6f"%period_train)
    
    # print("Accuracy(test): %.2f" % accuracy_score(encoder.transform(y_test.ravel()), pred))
    print("Accuracy(test): %.2f" % accuracy_score(y_test, pred))
    train_results = {
                       'acc':accuracy_score(y_test, pred),
                       'predict_time':period_train,
                       'CM':confusion_matrix(y_test,pred)}
    return (train_results, scaler, clf)

from sklearn.tree import DecisionTreeClassifier
def decision_tree_model_train_test(X_train, X_test, y_train, y_test, max_depth_):
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    
    clf = DecisionTreeClassifier(max_depth=max_depth_)
    
    
    start = time.time()
    clf.fit(X_train, y_train.ravel())
    end = time.time()
    print("Training period: %.6f"%(end-start))
    
    
    pred = clf.predict(X_train)
    
    print("Accuracy(train): %.2f" % accuracy_score(y_train, pred))
    
    start = time.time()
    pred = clf.predict(X_test)
    end = time.time()
    period_train = end-start
    print("Prediction period: %.6f"%period_train)
    
    # print("Accuracy(test): %.2f" % accuracy_score(encoder.transform(y_test.ravel()), pred))
    print("Accuracy(test): %.2f" % accuracy_score(y_test, pred))
    train_results = {'acc':accuracy_score(y_test, pred),
                       'predict_time':period_train,
                       'CM':confusion_matrix(y_test,pred)}
    return (train_results, scaler, clf)


def test_over_volunteers(clf_name, columns_name, n_in, scaler, clf):
    print("Tests over volunteers using model %s"%clf_name)
    filepath = lambda x: "./MHEALTHDATASET/mHealth_subject%d.log"%x
    
    #if 'testes' not in vars() or 'testes' in globals():
    
    testes = []
    for i in range(2,11):
#        print(filepath(i))
        data = pd.read_csv(filepath(i), sep='\t', skipinitialspace=True, header=None) #parse_dates=[0]
        data.columns = columns_name
        
        
        df = df_to_supervised_classification(data[:], 'label', n_in, 1)
        
        X = df[:].drop('label',axis=1)
        Y = df['label']
        
        X = scaler.transform(X)
        
        start = time.time()
    
        pred = clf.predict(X)
        end = time.time()
        testes.append({'clf':clf_name,
                       'dataset':i,
                       'acc':accuracy_score(Y, pred),
                       'predict_time':(end-start),
                       'CM':confusion_matrix(Y,pred)})
        print("Accuracy(%d): %.2f   -   time: %.6f" % (i,accuracy_score(Y, pred),(end-start)))
    return(testes)

from sklearn.externals.joblib import dump, load
def save_tests_model(clf_name, clf,train_results=None, testes=[]):
    a = {'clf_name':clf_name,
         'clf':clf,
         'testes':testes,
         'train_results':train_results
         }
    dump(a, clf_name+'.joblib')