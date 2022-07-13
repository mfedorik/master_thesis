## PACKAGE IMPORTS 
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import pickle
#!pip install pyod ## https://pyod.readthedocs.io/en/latest/ ##

from pyod.models.ocsvm import OCSVM
from sklearn.metrics import (f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix)

### IMPORT DATASET
def import_dataset(path):
    df_train = pd.read_csv(path + 'df_train.csv', index_col = 0, sep=';')
    X_train = df_train.drop("purchase" ,axis= 1)
    y_train = df_train['purchase']
    del df_train
    
    df_valid = pd.read_csv(path + 'df_valid.csv', index_col = 0, sep=';')
    X_valid = df_valid.drop("purchase" ,axis= 1)
    y_valid = df_valid['purchase']
    del df_valid
    
    df_fulltest = pd.read_csv(path + 'df_fulltest.csv', index_col = 0, sep=';')
    X_fulltest = df_fulltest.drop("purchase" ,axis= 1)
    y_fulltest = df_fulltest['purchase']
    del df_fulltest
    
    return X_train, y_train, X_fulltest, y_fulltest, X_valid, y_valid
  
path = '' # custom path
X_train, y_train, X_fulltest, y_fulltest, X_valid, y_valid = import_dataset(path + '/DATA/')


# EARLY PREDICTION + FULL MODEL ESTIMATION

def earlyprediction(X_train, X_valid, y_valid, X_fulltest, y_fulltest, leng):
  result = {}
  for i in leng:
    X_train = X_train[list(X_train.columns)[:5*i]]
    X_valid = X_valid[list(X_valid.columns)[:5*i]]
    X_fulltest = X_fulltest[list(X_fulltest.columns)[:5*i]]
    print("Prediction with data in shape:", X_train.shape)
    model = OCSVM(kernel = 'rbf', coef0 = 0.5, contamination = 0.026) 
    model.fit(X_train)
    final_result = {}
    final_result['parameters'] = {'length = {}, kernel = '"rbf"', coef0 = 0.5, contamination = 0.026'.format(i)}
    final_result['threshold'] = model.threshold_
    final_result['labels'] = model.labels_
    final_result['descision_scores'] = model.decision_scores_
    
    prediction = model.predict(X_valid)
    final_result['prediction'] = prediction

    final_result['confusion matrix'] = confusion_matrix(y_valid,prediction)        
    final_result['precision'] = precision_score(y_valid,prediction)
    final_result['accuracy'] = accuracy_score(y_valid,prediction)
    final_result['f1_score'] = f1_score(y_valid,prediction)
    print('f1 =', final_result['f1_score'])
    final_result['roc_auc'] = roc_auc_score(y_valid,prediction)
    print('roc_auc =', final_result['roc_auc'])
    final_result['recall'] = recall_score(y_valid,prediction)

    full_prediction = model.predict(X_fulltest)
    final_result['full_prediction'] = full_prediction

    final_result['full_confusion matrix'] = confusion_matrix(y_fulltest,full_prediction)        
    final_result['full_precision'] = precision_score(y_fulltest,full_prediction)
    final_result['full_accuracy'] = accuracy_score(y_fulltest,full_prediction)
    final_result['full_f1_score'] = f1_score(y_fulltest,full_prediction)
    print('FULL-f1 =', final_result['full_f1_score'])
    final_result['full_roc_auc'] = roc_auc_score(y_fulltest,full_prediction)
    print('FULL-roc_auc =', final_result['full_roc_auc'])
    final_result['full_recall'] = recall_score(y_fulltest,full_prediction)

    result['length_{0}'.format(i)] = final_result
    del model, prediction, full_prediction, final_result
  df = pd.DataFrame.from_dict(result)
  df.to_csv(path + "OC-SVM_EARLY_RESULTS.csv", index = True, sep = ';')

earlyprediction(X_train, X_valid, y_valid, X_fulltest, y_fulltest, [60,55,50,45,40,35,30,25,20,17,15,12,10,9,8,7,6,5])
