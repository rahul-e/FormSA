#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
import gp_emulator

# In[3]:


# Design points

df0 = pd.read_excel('result1.xlsx',sheet_name='DP',header=None)
df0.head()
df0.tail()


# In[28]:


# Convert design points into a numpy array with variables as columns

df = df0.dropna(axis=1).T
df.columns=df0[0]
df=df.iloc[:,1:50]
df.head()


# In[33]:


X = df.iloc[:,0:50].to_numpy()
print(np.shape(X))


# In[35]:


df = pd.read_excel('result.xlsx',sheet_name='28',header=None)
df=df.dropna(how='all')
y=df[28].dropna().to_list()
print(np.shape(y))


# In[36]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# In[37]:

print(np.shape(X_train),np.shape(Y_train))

scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train))
X_test_scaled = pd.DataFrame(scaler.transform(X_test))

# In[41]:
#------------------------
# Random forest emulator |
#------------------------
estimator = RandomForestRegressor(n_estimators=1001,
                               criterion='mse', min_samples_split=4,
                               random_state=1, oob_score='True',
                               n_jobs=-1)
estimator.fit(X_train, Y_train)
Y_train_pred = estimator.predict(X_train)
Y_test_pred = estimator.predict(X_test)

# In[42]:

print('MSE train Avgleft: %.3f, test: %.3f' % (
        mean_squared_error(Y_train, Y_train_pred),
        mean_squared_error(Y_test, Y_test_pred)))
print('R^2 train Avgleft: %.3f, test: %.3f' % (
        r2_score(Y_train, Y_train_pred),
        r2_score(Y_test, Y_test_pred)))


# In[43]:
print(Y_test, Y_test_pred)

# In[44]:

estimator = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
estimator.fit(X_train, Y_train)
Y_train_pred = estimator.predict(X_train)
Y_test_pred = estimator.predict(X_test)
print('MSE train Avgleft: %.3f, test: %.3f' % (
        mean_squared_error(Y_train, Y_train_pred),
        mean_squared_error(Y_test, Y_test_pred)))
print('R^2 train Avgleft: %.3f, test: %.3f' % (
        r2_score(Y_train, Y_train_pred),
        r2_score(Y_test, Y_test_pred)))
print(Y_test, Y_test_pred)
# In[ ]:


estimator = MLPRegressor(hidden_layer_sizes=(1000,), random_state=1, max_iter=500).fit(X_train, Y_train)
Y_test_pred = estimator.predict(X_test)
print('MSE train Avgleft: %.3f, test: %.3f' % (
        mean_squared_error(Y_train, Y_train_pred),
        mean_squared_error(Y_test, Y_test_pred)))
print('R^2 train Avgleft: %.3f, test: %.3f' % (
        r2_score(Y_train, Y_train_pred),
        r2_score(Y_test, Y_test_pred)))
print(Y_test, Y_test_pred)
