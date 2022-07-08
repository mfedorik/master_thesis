### import of packages 
import random
import datetime
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime


### data upload
path = "" # custom path
df = pd.read_csv(path + 'browsing_train.csv', header = 0, sep = ',')

### creating subssed of data with 20% of original dataset 
random.seed(22)
index20 = random.sample(set(df['session_id_hash']), round(len(set(df['session_id_hash']))*0.2))
df = df.loc[df['session_id_hash'].isin(index20)]
del index20

### EDA - information about original dataset
print('event types: {}'.format(set(df['event_type']))) # event types, event_product are product related clicks (detail, add, purchase, remove), pageview are general views.
print('product actions: {}'.format(set(df['product_action']))) # product actions
print('unique session IDs: {}'.format(len(set(df['session_id_hash'])))) # session IDs - FULL 4 934 699 sessions, SUBSET 986 940 
print('unique product SKUs: {}'.format(len(set(df['product_sku_hash'])))) # products - FULL 57 484 products, SUBSET 35 611
print('shape of original data: {}'.format(df.shape)) # shape -  SUBSET (7 215 820, 6)

##### preprocessing and eda #####
df['product_action'] = df['product_action'].fillna('view') ## filling the NaN for event_type pageview with 'view'
df.groupby('event_type')['event_type'].describe()
df.groupby(['event_type','product_action'])[['product_action']].describe()

### grouping sessions by session_id_hash ### 
df = df.groupby('session_id_hash')['product_action'].agg(list).reset_index()

### adding new column for labelling purchase sessions, counting the sessions 
df['purchase'] = np.where(df.product_action.map(set(['purchase']).issubset), 1, 0)
print(df['purchase'].value_counts()) ### non-purchase: 4881490, purchase: 53209;   976 237/ 10 703

### counting the length of sessions 
df['len'] = df.product_action.map(len)
print(min(df['len']), max(df['len'])) ### - length of original sessions is between 1 and 206

### identify first purchase, cut sessions after first purchase 
df['purchase_index'] = df['product_action'].map(lambda x : x.index('purchase') if 'purchase' in x else len(x))
df['help_col'] = list(zip(df['product_action'], df['purchase_index']))
df['session'] = df['help_col'].map(lambda x: x[0][0:x[1]+1])
df = df.drop(labels = ['product_action', 'purchase_index', 'help_col'], axis = 1)

### descriptive statistics
df['len'] = df.session.map(len)

### dropping sessions shorter than 5 clicks and longer than 60 clicks ### 
df.drop(df[df.len < 5].index, inplace = True)
df.drop(df[df.len > 60].index, inplace = True)
print(min(df['len']), max(df['len']))

### descriptive analysis of sessions lengths 
print(len(df)) # all sesisons - 367 558
print(len(df[df.purchase == 1])) # purchase sessions -  8761
print(df[df.purchase == 1]['len'].mean()) # purchase sessions - 22.77
print(df[df.purchase == 1]['len'].describe()) # purchase sessions - 25% 12 clicks, 50% 19 clicks, 75% 31 clicks
print(len(df[df.purchase == 0])) # non-purchase sessions - 358 797
print(df[df.purchase == 0]['len'].mean()) # purchase sessions - 13.61
print(df[df.purchase == 0]['len'].describe()) # purchase sessions - 25% 6 clicks, 50% 10 clicks, 75% 17 clicks

### plot of lengths of sessions 
g = sns.catplot(x = 'len', y= 'purchase', data = df, kind ='violin', orient= 'h', palette=["#F8CECC", "#D5E8D4"])
g.set_axis_labels("The length of session (in clicks)", "The density distribution for each class")
plt.show()

### Counter - count various types of clicks, Symbols - giving labels to specific types of clicks ### f
counts = Counter([item for session in df['session'] for item in session])
print(counts)
# Counter({'view': 3 649 616, 'detail': 1 336 424, 'add': 49833, 'remove': 36623, 'purchase': 8761})
symbols = {symbol: idx for idx, symbol in enumerate(sorted(counts, key=counts.get, reverse=True), 1)}
print(symbols)

### Symbolising sessions, creating new column of symbolised sessions, drop 3 columns  
symbolised_sessions = []
for id, session in enumerate(df['session']):
    symbolised_session = [symbols[s] for s in session]
    symbolised_sessions.append(symbolised_session)
df['ssessions'] = symbolised_sessions
df = df.drop(labels = ['session_id_hash', 'session', 'len'], axis = 1)

### Sessions padding 
padded_sessions = []
for ind in df.index:
  leng = 60 - len(df['ssessions'][ind])
  padding = leng * [0]
  padded_sessions.append(df['ssessions'][ind] + padding)
df['session'] = padded_sessions
del ind, leng, padding, padded_sessions 
df = df.drop(labels = ['ssessions'], axis = 1)

### Splitting the data into training, validation, test and final test set, partition 0.7/0.15/0.15
df_index = list(df.index)
df_index.sort()
random.seed(22)
training_index = random.sample(df_index, round(len(df)* 0.7))
df_train = df.loc[training_index]
df_fulltest = df.loc[~df.index.isin(training_index)]
random.seed(22)
validation_index = random.sample(list(df_fulltest.index), round(len(df_fulltest)* 0.5))
df_valid = df_fulltest.loc[validation_index]
df_test = df_fulltest.loc[~df_fulltest.index.isin(validation_index)]

del df
### Creating 6x155 columns for shallow models
for i in range(0,60):
    helplist = []
    for ind in df_train.index:
        helplist.append(df_train['session'][ind][i])
    list0 = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    for j in helplist:
        if j == 0:
            list0.append(1)
            list1.append(0)
            list2.append(0)
            list3.append(0)
            list4.append(0)
            list5.append(0)
        elif j == 1:
            list0.append(0)
            list1.append(1)
            list2.append(0)
            list3.append(0)
            list4.append(0)
            list5.append(0)
        elif j == 2:
            list0.append(0)
            list1.append(0)
            list2.append(1)
            list3.append(0)
            list4.append(0)
            list5.append(0)
        elif j == 3:
            list0.append(0)
            list1.append(0)
            list2.append(0)
            list3.append(1)
            list4.append(0)
            list5.append(0)
        elif j == 4:
            list0.append(0)
            list1.append(0)
            list2.append(0)
            list3.append(0)
            list4.append(1)
            list5.append(0)
        else:
            list0.append(0)
            list1.append(0)
            list2.append(0)
            list3.append(0)
            list4.append(0)
            list5.append(1)
    df_train['{}'.format(tuple([i+1, 0]))] = list0
    df_train['{}'.format(tuple([i+1, 1]))] = list1
    df_train['{}'.format(tuple([i+1, 2]))] = list2
    df_train['{}'.format(tuple([i+1, 3]))] = list3
    df_train['{}'.format(tuple([i+1, 4]))] = list4
    df_train['{}'.format(tuple([i+1, 5]))] = list5
    
index_key = {'training index' : training_index}
df_train = df_train.drop(labels = ['session'], axis = 1)
df_train.to_csv(path + 'df_train.csv', header = df_train.columns, sep = ';')
del df_train

for i in range(0,60):
    helplist = []
    for ind in df_fulltest.index:
        helplist.append(df_fulltest['session'][ind][i])
    list0 = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    for j in helplist:
        if j == 0:
            list0.append(1)
            list1.append(0)
            list2.append(0)
            list3.append(0)
            list4.append(0)
            list5.append(0)
        elif j == 1:
            list0.append(0)
            list1.append(1)
            list2.append(0)
            list3.append(0)
            list4.append(0)
            list5.append(0)
        elif j == 2:
            list0.append(0)
            list1.append(0)
            list2.append(1)
            list3.append(0)
            list4.append(0)
            list5.append(0)
        elif j == 3:
            list0.append(0)
            list1.append(0)
            list2.append(0)
            list3.append(1)
            list4.append(0)
            list5.append(0)
        elif j == 4:
            list0.append(0)
            list1.append(0)
            list2.append(0)
            list3.append(0)
            list4.append(1)
            list5.append(0)
        else:
            list0.append(0)
            list1.append(0)
            list2.append(0)
            list3.append(0)
            list4.append(0)
            list5.append(1)
    df_fulltest['{}'.format(tuple([i+1, 0]))] = list0
    df_fulltest['{}'.format(tuple([i+1, 1]))] = list1
    df_fulltest['{}'.format(tuple([i+1, 2]))] = list2
    df_fulltest['{}'.format(tuple([i+1, 3]))] = list3
    df_fulltest['{}'.format(tuple([i+1, 4]))] = list4
    df_fulltest['{}'.format(tuple([i+1, 5]))] = list5
    
df_fulltest = df_fulltest.drop(labels = ['session'], axis = 1)
df_fulltest.to_csv(path + 'df_fulltest.csv', header = df_fulltest.columns, sep = ';')
del df_fulltest

for i in range(0,60):
    helplist = []
    for ind in df_valid.index:
        helplist.append(df_valid['session'][ind][i])
    list0 = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    for j in helplist:
        if j == 0:
            list0.append(1)
            list1.append(0)
            list2.append(0)
            list3.append(0)
            list4.append(0)
            list5.append(0)
        elif j == 1:
            list0.append(0)
            list1.append(1)
            list2.append(0)
            list3.append(0)
            list4.append(0)
            list5.append(0)
        elif j == 2:
            list0.append(0)
            list1.append(0)
            list2.append(1)
            list3.append(0)
            list4.append(0)
            list5.append(0)
        elif j == 3:
            list0.append(0)
            list1.append(0)
            list2.append(0)
            list3.append(1)
            list4.append(0)
            list5.append(0)
        elif j == 4:
            list0.append(0)
            list1.append(0)
            list2.append(0)
            list3.append(0)
            list4.append(1)
            list5.append(0)
        else:
            list0.append(0)
            list1.append(0)
            list2.append(0)
            list3.append(0)
            list4.append(0)
            list5.append(1)
    df_valid['{}'.format(tuple([i+1, 0]))] = list0
    df_valid['{}'.format(tuple([i+1, 1]))] = list1
    df_valid['{}'.format(tuple([i+1, 2]))] = list2
    df_valid['{}'.format(tuple([i+1, 3]))] = list3
    df_valid['{}'.format(tuple([i+1, 4]))] = list4
    df_valid['{}'.format(tuple([i+1, 5]))] = list5
    
df_valid = df_valid.drop(labels = ['session'], axis = 1)
df_valid.to_csv(path + 'df_valid.csv', header = df_valid.columns, sep = ';')
del df_valid
    
for i in range(0,60):
    helplist = []
    for ind in df_test.index:
        helplist.append(df_test['session'][ind][i])
    list0 = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    for j in helplist:
        if j == 0:
            list0.append(1)
            list1.append(0)
            list2.append(0)
            list3.append(0)
            list4.append(0)
            list5.append(0)
        elif j == 1:
            list0.append(0)
            list1.append(1)
            list2.append(0)
            list3.append(0)
            list4.append(0)
            list5.append(0)
        elif j == 2:
            list0.append(0)
            list1.append(0)
            list2.append(1)
            list3.append(0)
            list4.append(0)
            list5.append(0)
        elif j == 3:
            list0.append(0)
            list1.append(0)
            list2.append(0)
            list3.append(1)
            list4.append(0)
            list5.append(0)
        elif j == 4:
            list0.append(0)
            list1.append(0)
            list2.append(0)
            list3.append(0)
            list4.append(1)
            list5.append(0)
        else:
            list0.append(0)
            list1.append(0)
            list2.append(0)
            list3.append(0)
            list4.append(0)
            list5.append(1)
    df_test['{}'.format(tuple([i+1, 0]))] = list0
    df_test['{}'.format(tuple([i+1, 1]))] = list1
    df_test['{}'.format(tuple([i+1, 2]))] = list2
    df_test['{}'.format(tuple([i+1, 3]))] = list3
    df_test['{}'.format(tuple([i+1, 4]))] = list4
    df_test['{}'.format(tuple([i+1, 5]))] = list5

df_test = df_test.drop(labels = ['session'], axis = 1)
df_test.to_csv(path + 'df_test.csv', header = df_test.columns, sep = ';')
del df_test  

del i, j, ind, list0, list1, list2, list3, list4, list5, helplist

### Back-up for index
index_key = {'training index' : training_index, 'validation index' : validation_index, 'df_index' : df_index }
file = open(path + "index_key.pkl", "wb")
pickle.dump(index_key, file)
file.close()
del training_index, validation_index, file, index_key, df_index 