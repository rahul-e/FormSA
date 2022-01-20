#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import os
#from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import xgboost as xgb

#import gp_emulator

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
#df.head()
df=df.drop(['Ps2', 'Ft2', 'C2', 'eta02', 'ap2', 'bp2',
            'Ft1', 'ap1'], axis=1)

# In[33]:

X = df.iloc[:,0:50].to_numpy()

# In[35]:


df = pd.read_csv('MaxStrain.txt',header=None,delimiter='\t')
df=df.dropna(how='all')
Sxx=df[0].dropna().to_numpy()
Syy=df[1].dropna().to_numpy()
Sxy=df[2].dropna().to_numpy()


print('Number of variables are %.0d and evaluations are %.0d' \
        %(np.shape(X)[1], len(Sxx)))

fig, big_axes = plt.subplots(figsize=(10.0, 15.0), nrows=2, ncols=1, sharey=False)
# In[36]:

for i, big_ax in enumerate(big_axes, start=1):
    # big_ax.set_title("Damping factor = %s" % DF[i], fontsize=16)
    # Turn off axis lines and ticks of the big subplot
    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax._frameon = False
#
# PLot histogram

ax1 = fig.add_subplot(2,1,1)
ax1.hist(Sxy)

ax2 = fig.add_subplot(2,1,2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Sxy, test_size=0.1, random_state=1)

# In[37]:

print('Dimension of training variable set is %.0d X %.0d and evaluation vector %.0d' \
        %(np.shape(X_train)[0], np.shape(X_train)[1], np.shape(Y_train)[0]))

# Scale the value of input parameters
scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train))
X_test_scaled = pd.DataFrame(scaler.transform(X_test))


#print(np.shape(X_train_scaled),np.shape(X_test_scaled))
# In[41]:
print('------------------------')
print(' Random forest emulator ')
print('------------------------')
estimator = RandomForestRegressor(n_estimators=1001,
                               criterion='squared_error', # “absolute_error”,“poisson”
                               min_samples_split=10,
                               max_samples=0.5,
                               random_state=123,
                               # min_samples_leaf=3,
                               # ccp_alpha=0.005,
                               # oob_score='True',
                               n_jobs=-1)

estimator.fit(X_train_scaled, Y_train)
Y_train_pred_rf= estimator.predict(X_train_scaled)
Y_test_pred_rf = estimator.predict(X_test_scaled)

# In[42]:

print('MSE train Avgleft: %.3f, test: %.3f' % (
        mean_squared_error(Y_train, Y_train_pred_rf),
        mean_squared_error(Y_test, Y_test_pred_rf)))
print('R^2 train Avgleft: %.3f, test: %.3f' % (
        r2_score(Y_train, Y_train_pred_rf),
        r2_score(Y_test, Y_test_pred_rf)))


# In[43]:
ax2.plot(Y_test, (Y_test_pred_rf-Y_test),'^',label='RF')
#print(X_test, Y_test_pred_rf)
print('------------------------')
print('     XG Booster          ')
print('------------------------')
# In[44]:
#estimator = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.3,
#                max_depth = 5, alpha = 10, n_estimators = 101, subsample=0.9)
estimator = xgb.XGBRegressor(colsample_bytree=0.4, # [0.4,0.6,0.8]
                 gamma=0, # [0,0.03,0.1,0.3]
                 learning_rate=0.07, # [0.1,0.07]
                 max_depth=5, # [3,5]
                 min_child_weight=1.5, # [1.5,6,10],
                 n_estimators=10000,
                 reg_alpha=0.75, # [1e-5, 1e-2,  0.75]
                 reg_lambda=0.45, # [1e-5, 1e-2, 0.45]
                 subsample=0.6, # [0.6,0.95]
                 seed=42)

estimator.fit(X_train, Y_train)
Y_train_pred_xgb = estimator.predict(X_train)
Y_test_pred_xgb = estimator.predict(X_test)
print('MSE train Avgleft: %.3f, test: %.3f' % (
        mean_squared_error(Y_train, Y_train_pred_xgb),
        mean_squared_error(Y_test, Y_test_pred_xgb)))
print('R^2 train Avgleft: %.3f, test: %.3f' % (
        r2_score(Y_train, Y_train_pred_xgb),
        r2_score(Y_test, Y_test_pred_xgb)))
#print(Y_test, Y_test_pred_xgb)
ax2.plot(Y_test, (Y_test_pred_xgb-Y_test),'s',label='XGB')
# In[ ]:

print('------------------------')
print('     MLP Regressor      ')
print('------------------------')

estimator = MLPRegressor(hidden_layer_sizes=(100,), random_state=1, max_iter=500).fit(X_train_scaled, Y_train)
Y_train_pred_mlp = estimator.predict(X_train_scaled)
Y_test_pred_mlp = estimator.predict(X_test_scaled)
print('MSE train Avgleft: %.3f, test: %.3f' % (
        mean_squared_error(Y_train, Y_train_pred_mlp),
        mean_squared_error(Y_test, Y_test_pred_mlp)))
print('R^2 train Avgleft: %.3f, test: %.3f' % (
        r2_score(Y_train, Y_train_pred_mlp),
        r2_score(Y_test, Y_test_pred_mlp)))
#print(Y_test, Y_test_pred_mlp)
#plt.plot(Y_test, (Y_test_pred_mlp-Y_test),'o',label='MLP')


print('------------------------')
print('     SVR Regressor      ')
print('------------------------')


#estimator = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
#estimator = SVR(kernel="linear", C=100, gamma="auto")
estimator = SVR(kernel="linear",
                # degree=6,
                C=100,
                gamma=0.1,
                epsilon=0.1,)

estimator.fit(X_train_scaled, Y_train)
Y_train_pred_svr = estimator.predict(X_train_scaled)
Y_test_pred_svr = estimator.predict(X_test_scaled)
print('MSE train Avgleft: %.3f, test: %.3f' % (
        mean_squared_error(Y_train, Y_train_pred_svr),
        mean_squared_error(Y_test, Y_test_pred_svr)))
print('R^2 train Avgleft: %.3f, test: %.3f' % (
        r2_score(Y_train, Y_train_pred_svr),
        r2_score(Y_test, Y_test_pred_svr)))
#print(Y_test, '\n', Y_test_pred_svr)
ax2.plot(Y_test, (Y_test_pred_svr-Y_test),'*',label='SVR')


print('------------------------')
print('     Linear Regression  ')
print('------------------------')

estimator = linear_model.LinearRegression()
estimator.fit(X_train_scaled, Y_train)
Y_train_pred_lr = estimator.predict(X_train_scaled)
Y_test_pred_lr = estimator.predict(X_test_scaled)
print('MSE train Avgleft: %.3f, test: %.3f' % (
        mean_squared_error(Y_train, Y_train_pred_lr),
        mean_squared_error(Y_test, Y_test_pred_lr)))
print('R^2 train Avgleft: %.3f, test: %.3f' % (
        r2_score(Y_train, Y_train_pred_lr),
        r2_score(Y_test, Y_test_pred_lr)))
#print(Y_test, '\n', Y_test_pred_lr)
ax2.plot(Y_test, (Y_test_pred_lr-Y_test),'*',label='Linear Reg')

#

print('------------------------')
print('   TheilSen  ')
print('------------------------')

estimator = linear_model.TheilSenRegressor(#loss='squared_error',
                            fit_intercept=True,
                            #penalty='elasticnet',
                            #alpha=0.0001,
                            #learning_rate='constant',
                            #positive=True,
                            #solver='lsqr',
                            #max_iter=11000,
                            random_state=123)
estimator.fit(X_train_scaled, Y_train)
Y_train_pred_lasso = estimator.predict(X_train_scaled)
Y_test_pred_lasso = estimator.predict(X_test_scaled)
print('MSE train Avgleft: %.3f, test: %.3f' % (
        mean_squared_error(Y_train, Y_train_pred_lasso),
        mean_squared_error(Y_test, Y_test_pred_lasso)))
print('R^2 train Avgleft: %.3f, test: %.3f' % (
        r2_score(Y_train, Y_train_pred_lasso),
        r2_score(Y_test, Y_test_pred_lasso)))
#print(Y_test, '\n', Y_test_pred_lr)
ax2.plot(Y_test, (Y_test_pred_lasso-Y_test),'*',label='Lasso Reg')

ax2.hlines(0,min(Y_test)-0.1,max(Y_test)+0.1,'r')
ax2.grid(True, linestyle='-',which ='major',alpha=0.5)
ax2.grid(True, linestyle=':',which ='minor')
ax2.grid(True, linestyle='-',which ='major',alpha=0.5)
ax2.grid(True, linestyle=':',which ='minor')
# ax2.grid(which='both')
ax2.minorticks_on()

plt.legend()
plt.show()
