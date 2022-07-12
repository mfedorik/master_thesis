## PACKAGE IMPORTS 
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import pickle
#!pip install pyod ## https://pyod.readthedocs.io/en/latest/ ##

from pyod.models.auto_encoder import AutoEncoder
from tensorflow.keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from sklearn.metrics import (recall_score,  roc_auc_score, f1_score, precision_score, accuracy_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV

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

### GridSearch for parameter selection
parameters = {'hidden_neurons':[[128, 64, 32, 16, 16, 32, 64, 128],[128, 64, 32, 32, 64, 128],
                                [128, 64, 16, 16, 64, 128],[128, 64, 32, 32, 64, 128],
                                [128, 32, 16, 16, 32, 128],[64, 32, 16, 16, 32, 64],
                                [128, 64, 64, 128],[128, 32, 32, 128],[64, 32, 32, 64],
                                [64, 16, 16, 64],[32, 16, 16, 32],
                                [128, 64, 32, 16, 32, 64, 128],[128, 64, 32, 64, 128],
                                [128, 64, 16, 64, 128],[128, 32, 16, 32, 128],[128,64,128],
                                [128,32,128],[64,32,64],[64,16,64],[32,16,32]], 
              'loss':['binary_crossentropy'],
              'contamination': [0.026],
              'l2_regularizer': [0.01],
              'optimizer' : [Adam(0.001), Adam(0.005), Adam(0.01)],
              'batch_size' : [16,32,64,128]}
##### Contamination in the dataset was on level ~2.6% of purchase sessions from all sessions

ae_model = AutoEncoder(epochs = 20)
model = GridSearchCV(ae_model, parameters, scoring = ['f1', 'roc_auc', 'recall'], refit = 'f1', verbose = 2, n_jobs = None, cv= 3)
model.fit(X_train, y_train)

### Exporting the results in .csv
results = pd.DataFrame.from_dict(model.cv_results_)
results.to_csv(path + "/AE_RESULTS.csv", index = True, sep = ';')

### Re-estimation of the model with the best performing parameters
model1 = AutoEncoder(batch_size=32, contamination=0.026, epochs =20,
      hidden_activation='relu', hidden_neurons=[128, 32, 128],
      l2_regularizer=0.01, loss='binary_crossentropy',
      optimizer=Adam(0.001), validation_size=0.1, verbose=1)
model1.fit(X_train)


final_result = {}
final_result['parameters'] = {'(batch_size=32, contamination=0.026, epochs =20, hidden_activation=''relu'', hidden_neurons=[128, 32, 128], l2_regularizer=0.01, loss=''binary_crossentropy'', optimizer=Adam(0.001), validation_size=0.1, verbose=1))'}
final_result['threshold'] = model1.threshold_
final_result['labels'] = model1.labels_
final_result['descision_scores'] = model1.decision_scores_
    
prediction = model1.predict(X_valid)
final_result['prediction'] = prediction

final_result['confusion matrix'] = confusion_matrix(y_valid,prediction)        
final_result['precision'] = precision_score(y_valid,prediction)
final_result['accuracy'] = accuracy_score(y_valid,prediction)
final_result['f1_score'] = f1_score(y_valid,prediction)
final_result['roc_auc'] = roc_auc_score(y_valid,prediction)
final_result['recall'] = recall_score(y_valid,prediction)

full_prediction = model1.predict(X_fulltest)
final_result['full_prediction'] = full_prediction

final_result['full_confusion matrix'] = confusion_matrix(y_fulltest,full_prediction)        
final_result['full_precision'] = precision_score(y_fulltest,full_prediction)
final_result['full_accuracy'] = accuracy_score(y_fulltest,full_prediction)
final_result['full_f1_score'] = f1_score(y_fulltest,full_prediction)
final_result['full_roc_auc'] = roc_auc_score(y_fulltest,full_prediction)
final_result['full_recall'] = recall_score(y_fulltest,full_prediction)

file1 = open(path + '/AE_FINAL_RESULTS.pkl', 'wb')
pickle.dump(final_result, file1)
file1.close()

# EARLY PREDICTION

##### [60,55,50,45,40,35,30,25,20,17,15,12,10,9,8,7,6,5]
i = 55 

X_train = X_train[list(X_train.columns)[:5*i]]
X_valid = X_valid[list(X_valid.columns)[:5*i]]
X_fulltest = X_fulltest[list(X_fulltest.columns)[:5*i]]

model1.fit(X_train)

final_result = {}
final_result['length'] = i
final_result['shape'] = X_train.shape
final_result['parameters'] = {'(batch_size=32, contamination=0.026, epochs =20, hidden_activation=''relu'', hidden_neurons=[128, 32, 128], l2_regularizer=0.01, loss=''binary_crossentropy'', optimizer=Adam(0.001), validation_size=0.1, verbose=1))'}
final_result['threshold'] = model1.threshold_
final_result['labels'] = model1.labels_
final_result['descision_scores'] = model1.decision_scores_
    
prediction = model1.predict(X_valid)
final_result['prediction'] = prediction

final_result['confusion matrix'] = confusion_matrix(y_valid,prediction)        
final_result['precision'] = precision_score(y_valid,prediction)
final_result['accuracy'] = accuracy_score(y_valid,prediction)
final_result['f1_score'] = f1_score(y_valid,prediction)
final_result['roc_auc'] = roc_auc_score(y_valid,prediction)
final_result['recall'] = recall_score(y_valid,prediction)

full_prediction = model1.predict(X_fulltest)
final_result['full_prediction'] = full_prediction

final_result['full_confusion matrix'] = confusion_matrix(y_fulltest,full_prediction)        
final_result['full_precision'] = precision_score(y_fulltest,full_prediction)
final_result['full_accuracy'] = accuracy_score(y_fulltest,full_prediction)
final_result['full_f1_score'] = f1_score(y_fulltest,full_prediction)
final_result['full_roc_auc'] = roc_auc_score(y_fulltest,full_prediction)
final_result['full_recall'] = recall_score(y_fulltest,full_prediction)

file1 = open(path + '/AE_EARLY_RESULTS_{}.pkl'.format(i), 'wb')
pickle.dump(final_result, file1)
file1.close()
