# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:55:07 2020

@author: Krist
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
#%%
path = 'C:/Users/Krist/OneDrive/desktop/bbg/pj/'
trade = pd.read_csv(path+'trades_narrow_BTC-USD_2018.delim', sep = '|')
trade_seg = trade.loc[(trade['received_utc_nanoseconds'] >= 1522627200*1e9) & (trade['received_utc_nanoseconds'] < (1522627200 + 5*24*3600)*1e9)].copy()

for t in [2,5,10,30,60]:
    data_2min = pd.read_csv(path+'produce/'+str(t)+'min.csv')
    data_2min = data_2min.rename(columns = {'Unnamed: 0': 'Index'})
    data_2min = data_2min.set_index('Index')
    trade_seg = trade_seg.join(data_2min)
#%% 
trade_seg['ma_2_5'] = trade_seg['ewmp_2min'] - trade_seg['ewmp_5min']
trade_seg['ma_5_10'] = trade_seg['ewmp_5min'] - trade_seg['ewmp_10min'] 
trade_seg['ma_10_30'] = trade_seg['ewmp_10min'] - trade_seg['ewmp_30min']
trade_seg['ma_30_60'] = trade_seg['ewmp_30min'] - trade_seg['ewmp_60min'] 

trade_seg['exe'] = np.where(trade_seg['Side'] == -1, 1, 0) 

#%%
data = trade_seg.dropna()
data_train = data.loc[(data['received_utc_nanoseconds'] >= 1522627200*1e9) & (data['received_utc_nanoseconds'] < (1522627200 + 4*24*3600)*1e9)]
data_new = data.loc[(data['received_utc_nanoseconds'] >= (1522627200 + 4*24*3600)*1e9) & (data['received_utc_nanoseconds'] < (1522627200 + 5*24*3600)*1e9)]
X = data_train[['spread_2min','spread_5min','spread_10min','spread_30min','spread_60min',\
               'Sratio_2min','Sratio_5min','Sratio_10min','Sratio_30min','Sratio_60min',\
               'ma_2_5','ma_5_10','ma_10_30','ma_30_60',\
               'vol_2min','vol_5min','vol_10min','vol_30min','vol_60min']].values
y = data_train['exe'].values
#%%
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
'''
for i in range(10):
    plt.figure()
    plt.plot(pca.components_[i])
    '''
#%%
#cv generator

def gen_fold(n, size):
    for i in range(n-1):
        idx_train = [i for i in range((i+1)*size)]
        idx_test = [i for i in range((i+1)*size, (i+2)*size)]
        yield (np.array(idx_train), np.array(idx_test))


#%%
#function for day forward nested cross validation

def day_forward_cv(space, split, model):
    size = len(X)//split
    
    if model == 'logit':
        lr = LogisticRegression(random_state = 0,max_iter = 1e4, class_weight = 'balanced', solver = 'lbfgs', penalty = 'l2')
        gsc = GridSearchCV(lr, space, scoring='precision', cv = gen_fold(split, size), verbose = 2)
        gsc.fit(X, y)
        best_params = gsc.best_params_
        return best_params
    if model == 'rf':
        rf = RandomForestClassifier(criterion = 'entropy', class_weight = 'balanced', random_state = 0)
        rsc = RandomizedSearchCV(rf, space, scoring='precision', n_iter = 10, cv = gen_fold(split,size), verbose=2, random_state=0, n_jobs = -1)
        rsc.fit(X, y)
        best_params = rsc.best_params_
        return best_params
    if model == 'gb':
        gb = GradientBoostingClassifier(random_state = 0)
        rsc = RandomizedSearchCV(gb, space, scoring='precision', n_iter = 10, cv = gen_fold(split,size), verbose=2, random_state=0, n_jobs = -1)
        rsc.fit(X, y)
        best_params = rsc.best_params_
        return best_params
#%%
space = {'C':[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}      
best_c = day_forward_cv(space, 5, 'logit')
space = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'log2'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [5,20,50,100]}
best_rf = day_forward_cv(space, 5, 'rf')
space = {'learning_rate': [0.05,0.1,0.2],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'log2'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [5,20,50,100]}
best_gb = day_forward_cv(space, 5, 'gb')
#%%
# data for prediction
sc = StandardScaler()
X = sc.fit_transform(X)
pca = PCA(n_components = 10)
X = pca.fit_transform(X)
X_test = data_new[['spread_2min','spread_5min','spread_10min','spread_30min','spread_60min',\
               'Sratio_2min','Sratio_5min','Sratio_10min','Sratio_30min','Sratio_60min',\
               'ma_2_5','ma_5_10','ma_10_30','ma_30_60',\
               'vol_2min','vol_5min','vol_10min','vol_30min','vol_60min']]
y_test = data_new['exe']
X_test = sc.transform(X_test)
X_test = pca.transform(X_test)
#%% 
# logit
classifier = LogisticRegression(random_state = 0,max_iter = 1e4, class_weight = 'balanced', solver = 'lbfgs', penalty = 'l2', C = 1e-5)
classifier.fit(X, y)
y_pred = classifier.predict(X_test)

print(classifier.coef_)

# evaluation
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm).to_csv(path+'lr.csv')
ra_score = roc_auc_score(y_test, classifier.predict_proba(X_test)[:,1])
print(ra_score)
fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
#%%
# rf
classifier = RandomForestClassifier(criterion = 'entropy', class_weight = 'balanced', random_state = 0,\
                                    bootstrap = best_rf['bootstrap'], max_depth = best_rf['max_depth'],\
                                    max_features = best_rf['max_features'], min_samples_leaf = best_rf['min_samples_leaf'],\
                                    min_samples_split = best_rf['min_samples_split'], n_estimators = best_rf['n_estimators'])
classifier.fit(X, y)
y_pred = classifier.predict(X_test)

#print(classifier.coef_)

# evaluation
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm).to_csv(path+'rf.csv')
ra_score = roc_auc_score(y_test, classifier.predict_proba(X_test)[:,1])
print(ra_score)
fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, marker='.', label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
#%%
# gb
classifier = GradientBoostingClassifier(random_state = 0,\
                                    learning_rate = best_gb['learning_rate'], max_depth = best_gb['max_depth'],\
                                    max_features = best_gb['max_features'], min_samples_leaf = best_gb['min_samples_leaf'],\
                                    min_samples_split = best_gb['min_samples_split'], n_estimators = best_gb['n_estimators'])
classifier.fit(X, y)
y_pred = classifier.predict(X_test)

#print(classifier.coef_)

# evaluation
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm).to_csv(path+'gb.csv')
ra_score = roc_auc_score(y_test, classifier.predict_proba(X_test)[:,1])
print(ra_score)
fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, marker='.', label='Gradient Boosting')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

