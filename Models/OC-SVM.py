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

#### Computing the contamination level
c = round(sum(y_train == 1)/len(y_train),3)

##### Contamination in the training dataset was on level ~2.4% of purchase sessions from all sessions - c=0.024
model = OCSVM(kernel = 'rbf', coef0 = 0.5, contamination = c)         
model.fit(X_train)

##### Model export
file = open(path + "/OCSVM/OCSVM_FINAL_MODEL.pkl", "wb")
pickle.dump(model, file)
file.close()

##### Prediction on validation data, prediction on test data, evaluation
final_result = {}
final_result['parameters'] = {str(model)}
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

test_prediction = model.predict(X_test)
final_result['test_prediction'] = test_prediction

final_result['test_confusion matrix'] = confusion_matrix(y_test,test_prediction)        
final_result['test_precision'] = precision_score(y_test,test_prediction)
final_result['test_accuracy'] = accuracy_score(y_test,test_prediction)
final_result['test_f1_score'] = f1_score(y_test,test_prediction)
final_result['test_roc_auc'] = roc_auc_score(y_test,test_prediction)
final_result['test_recall'] = recall_score(y_test,test_prediction)
result = {}
result['model'] = final_result
file = open(path + "/OCSVM/OCSVM_FINAL_RESULT.pkl", "wb")
pickle.dump(final_result, file)
file.close()

del file, final_result, model, prediction, result, test_prediction, X_test, X_train, X_valid, y_test, y_train, y_valid

#### Early prediction function
def earlyprediction(leng):
    result = {}
    for i in leng:
        ### IMPORT DATASET
        def import_dataset(path):
            df_train = pd.read_csv(path + 'df{}_train.csv'.format(i), index_col = 0, sep=';')
            X_train = df_train.drop("purchase" ,axis= 1)
            y_train = df_train['purchase']
            del df_train

            df_valid = pd.read_csv(path + 'df{}_valid.csv'.format(i), index_col = 0, sep=';')
            X_valid = df_valid.drop("purchase" ,axis= 1)
            y_valid = df_valid['purchase']
            del df_valid

            df_test = pd.read_csv(path + 'df{}_test.csv'.format(i), index_col = 0, sep=';')
            X_test = df_test.drop("purchase" ,axis= 1)
            y_test = df_test['purchase']
            del df_test

            return X_train, y_train, X_test, y_test, X_valid, y_valid

        path = '' # custom path
        X_train, y_train, X_fulltest, y_fulltest, X_valid, y_valid = import_dataset(path + '/DATA/')

        #### Computing the contamination level
        c = round(sum(y_train == 1)/len(y_train),3)

        ##### Contamination in the training dataset was adjusted by contamination of data subset
        model = OCSVM(kernel = 'rbf', coef0 = 0.5, contamination = c)  
        model.fit(X_train)
        
        ##### Model export
        file = open(path + "/OCSVM/OCSVM_FINAL_MODEL_LEN_{}.pkl".format(i), "wb")
        pickle.dump(model, file)
        file.close()
        
        final_result = {}
        final_result['parameters'] = {str(model)}
        final_result['threshold'] = model.threshold_
        final_result['labels'] = model.labels_
        final_result['descision_scores'] = model.decision_scores_
        final_result['length'] = i
        final_result['train_shape'] = X_train.shape
        
        ##### Prediction on validation data, prediction on test data, evaluation
        prediction = model.predict(X_valid)
        final_result['prediction'] = prediction

        final_result['confusion matrix'] = confusion_matrix(y_valid,prediction)        
        final_result['precision'] = precision_score(y_valid,prediction)
        final_result['accuracy'] = accuracy_score(y_valid,prediction)
        final_result['f1_score'] = f1_score(y_valid,prediction)
        final_result['roc_auc'] = roc_auc_score(y_valid,prediction)
        final_result['recall'] = recall_score(y_valid,prediction)

        test_prediction = model.predict(X_test)
        final_result['test_prediction'] = test_prediction

        final_result['test_confusion matrix'] = confusion_matrix(y_test,test_prediction)        
        final_result['test_precision'] = precision_score(y_test,test_prediction)
        final_result['test_accuracy'] = accuracy_score(y_test,test_prediction)
        final_result['test_f1_score'] = f1_score(y_test,test_prediction)
        final_result['test_roc_auc'] = roc_auc_score(y_test,test_prediction)
        final_result['test_recall'] = recall_score(y_test,test_prediction)
        result['model_with_length_{}'.format(i)] = final_result
        file = open(path + "/OCSVM/OCSVM_FINAL_RESULT_LEN_{}.pkl".format(i), "wb")
        pickle.dump(final_result, file)
        file.close()
        del X_train, y_train, X_test, y_test, X_valid, y_valid, c, model, prediction, test_prediction, final_result, file

    ### Exporting the results in .csv
    result = pd.DataFrame.from_dict(result)
    result.to_csv(path + "/OCSVM/OCSVM_RESULTS_EARLY.csv", index = True, sep = ';')

earlyprediction([5,10,15,20,25,30])
