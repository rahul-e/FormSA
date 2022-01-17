#!/usr/bin/env python
# coding: utf-8

# In[43]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import chaospy as cp
import numpy as np


# In[15]:


df = pd.read_excel('result.xlsx',sheet_name='28',header=None)
df=df.dropna(how='all')
df.head()
#df.tail()

# In[16]:
x = df[28].dropna().to_list()
x = np.asarray(x)
print('Length of the vector of evaluations', len(x))
#plt.hist(x)


# In[27]:


#df[28].hist() #Epsilon XY


# In[28]:


#df[27].hist() #Epsilon YY


# In[29]:


#df[26].hist() #Epsilon XX


# In[4]:


percent = 0.1
#dist_Pc1 = (1+cp.Uniform(-1*percent,1*percent))*20
#dist_Pc2  = (1+cp.Uniform(-1*percent,1*percent))*20

#dist_Ft1 = (1+cp.Uniform(-1*percent,1*percent))*0.1
#dist_Ft2 = (1+cp.Uniform(-1*percent,1*percent))*0.1

dist_C1 = (1+cp.Uniform(-1*percent,1*percent))*4.6252
dist_C2 = (1+cp.Uniform(-1*percent,1*percent))*4.6251

dist_eta01 = (1+cp.Uniform(-1*percent,1*percent))*4.95739
dist_eta02 = (1+cp.Uniform(-1*percent,1*percent))*4.36586

dist_ap1 = (1+cp.Uniform(-1*percent,1*percent))*1.0
dist_ap2 = (1+cp.Uniform(-1*percent,1*percent))*1.0

dist_bp1 = (1+cp.Uniform(-1*percent,1*percent))*1.0668
dist_bp2 = (1+cp.Uniform(-1*percent,1*percent))*0.75456



dist_Pc1 = cp.Uniform(18.0002,22.0001)
dist_Pc2 = cp.Uniform(18.001,22.0000)
dist_Ft1 = cp.Uniform(0.09000,0.11001)
dist_Ft2 = cp.Uniform(0.09001,0.11000)
dist_C1 = cp.Uniform(4.16268,5.08772)
dist_C2 = cp.Uniform(4.16259,5.08761)
#
dist_eta01 = cp.Uniform(4.461651,5.453129)
dist_eta02 = cp.Uniform(3.9292,4.8024)
#
dist_ap1 = cp.Uniform(0.9000,1.1000)
dist_ap2 = cp.Uniform(0.9001,1.1001)
dist_bp1 = cp.Uniform(0.96012,1.17348)
dist_bp2 = cp.Uniform(0.6791,0.8300)


dist = cp.J(dist_Pc1, dist_Pc2, dist_Ft1, dist_Ft2, dist_C1, dist_C2, dist_eta01, dist_eta02, dist_ap1, dist_ap2, dist_bp1, dist_bp2)#,dist_Ft1,dist_Ft2,dist_C1,dist_C2,dist_eta01,dist_eta02,dist_ap1,dist_ap2,dist_bp1,dist_bp2)
#print(dist_Pc.sample(2,'R'), dist_P.sample(2,'R'))
# In[5]:

degree=2
orths=cp.orth_ttr(degree, dist, normed=True)
print('Length of terms in PCE', len(orths))
#orths=cp.orth_ttr(degree, (1+dist_Pc)*20, normed=True)
#print(orths)
# In[10]:

# In[9]:

df0 = pd.read_excel('result1.xlsx',sheet_name='DP',header=None)
df0.head()


# In[23]:
l = []
for i in range(1,51):
    l.append(df0[i].drop(0,axis=0).to_list())

#samples = dist.sample(50,'R')
#np.shape(samples)
l = np.asarray(l)
l=l.T
print('Shape of the design point array',np.shape(l))
#
labels = df0[0].drop(0,axis=0).to_list()
pce = cp.fit_regression(orths, l, x)
T_sens = cp.Sens_t(pce, dist)
print(T_sens)
varnum = [n+1 for n in range(0,len(T_sens))]
plt.bar(varnum, T_sens)
plt.xticks(varnum, labels)
plt.show()
