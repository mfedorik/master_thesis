## PACKAGE IMPORTS 
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import pickle
#!pip install pyod ## https://pyod.readthedocs.io/en/latest/ ##

from pyod.models.vae import VAE
from tensorflow.keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from sklearn.metrics import (recall_score,  roc_auc_score, f1_score, precision_score, accuracy_score, confusion_matrix)

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

### Model estimation
#### Hyperparameter tuning was performed manually - GridSearchCV did not satisfy the requirements needed for VAE and loss function converged to the infinity.
##### Contamination in the dataset was on level ~2.6% of purchase sessions from all sessions

###### batch_size - [128, 64, 32]
###### encoder size - usually larger or equal to decoder with the number of nodes from 8 to 128 in encoder and from 4 to 128 in decoder
####### Sample:
model = VAE(encoder_neurons= [64,32,16], decoder_neurons= [8,16], 
            verbose = 1, epochs = 10, contamination= 0.026, loss= binary_crossentropy,
            l2_regularizer = 0.00001, optimizer = Adam(0.00001),
            batch_size = 128, random_state= 22)

model.fit(X_train)

final_result = {}
final_result['parameters'] = {'(encoder_neurons= [64,32,16], decoder_neurons= [8,16], verbose = 1, epochs = 10, contamination= 0.026, loss= binary_crossentropy, l2_regularizer = 0.00001, optimizer = Adam(0.00001),batch_size = 128, random_state= 22'}
final_result['threshold'] = model.threshold_
final_result['labels'] = model.labels_
final_result['descision_scores'] = model.decision_scores_
    
prediction = model.predict(X_valid)
final_result['prediction'] = prediction

final_result['confusion matrix'] = confusion_matrix(y_valid,prediction)        
final_result['precision'] = precision_score(y_valid,prediction)
final_result['accuracy'] = accuracy_score(y_valid,prediction)
final_result['f1_score'] = f1_score(y_valid,prediction)
final_result['roc_auc'] = roc_auc_score(y_valid,prediction)
final_result['recall'] = recall_score(y_valid,prediction)

full_prediction = model.predict(X_fulltest)
final_result['full_prediction'] = full_prediction

final_result['full_confusion matrix'] = confusion_matrix(y_fulltest,full_prediction)        
final_result['full_precision'] = precision_score(y_fulltest,full_prediction)
final_result['full_accuracy'] = accuracy_score(y_fulltest,full_prediction)
final_result['full_f1_score'] = f1_score(y_fulltest,full_prediction)
final_result['full_roc_auc'] = roc_auc_score(y_fulltest,full_prediction)
final_result['full_recall'] = recall_score(y_fulltest,full_prediction)

file1 = open(path + '/VAE_FINAL_RESULTS.pkl', 'wb')
pickle.dump(final_result, file1)
file1.close()

# EARLY PREDICTION - performed manually

##### [60,55,50,45,40,35,30,25,20,17,15,12,10,9,8,7,6,5]
i = 55 

X_train = X_train[list(X_train.columns)[:5*i]]
X_valid = X_valid[list(X_valid.columns)[:5*i]]
X_fulltest = X_fulltest[list(X_fulltest.columns)[:5*i]]

###### Sample model 
model1 = VAE(encoder_neurons= [16,8], decoder_neurons= [8], 
            verbose = 1, epochs = 10, contamination= 0.026, loss= binary_crossentropy,
            l2_regularizer = 0.00001, optimizer = Adam(0.00001),
            batch_size = 128, random_state= 22)

model1.fit(X_train)

final_result = {}
final_result['length'] = i
final_result['shape'] = X_train.shape
final_result['parameters'] = {'(encoder_neurons= [16,8], decoder_neurons= [8],  verbose = 1, epochs = 10, contamination= 0.026, loss= binary_crossentropy, l2_regularizer = 0.00001, optimizer = Adam(0.00001), batch_size = 128, random_state= 22)'}
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

file1 = open(path + '/VAE_EARLY_RESULTS_{}.pkl'.format(i), 'wb')
pickle.dump(final_result, file1)
file1.close()
