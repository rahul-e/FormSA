#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import chaospy as cp
import numpy as np


# In[89]:


df = pd.read_csv('DispZ_NumNode_maxdisp.txt',header=None,delimiter='\t')
df=df.dropna(how='all')
df.columns=['Number_of_nodes', 'Max'  ]
print(df.head())

NumN = df['Number_of_nodes']
MaxDisp = df['Max']

var = NumN

print('Length of the vector of evaluations', len(var))
percent = 0.1

Ps1 = 10  # 1 - corresponding to tool-ply interface
Ps2 = 10  # 2 - corresponding to ply-ply interface
Ft1 = 0.1
Ft2 = 0.1
C1 = 0.737
C2 = 0.343
eta01 = 0.097
eta02 = 0.036
ap1 = 1.0
ap2 = 1.0
bp1 = 0.56
bp2 = 0.53

dist_Pc1 = cp.Uniform(Ps1*(1-percent),Ps1*(1+percent))
dist_Pc2  = cp.Uniform(Ps2*(1-percent),Ps2*(1+percent))

dist_Ft1 = cp.Uniform(Ft1*(1-percent),Ft1*(1+percent))
dist_Ft2 = cp.Uniform(Ft2*(1-percent),Ft2*(1+percent))

dist_C1 = cp.Uniform(C1*(1-percent),C1*(1+percent))
dist_C2 = cp.Uniform(C2*(1-percent),C2*(1+percent))

dist_eta01 = cp.Uniform(eta01*(1-percent),eta01*(1+percent))
dist_eta02 = cp.Uniform(eta02*(1-percent),eta02*(1+percent))

dist_ap1 = cp.Uniform(ap1*(1-percent),ap1*(1+percent))
dist_ap2 = cp.Uniform(ap2*(1-percent),ap2*(1+percent))

dist_bp1 = cp.Uniform(bp1*(1-percent),bp1*(1+percent))
dist_bp2 = cp.Uniform(bp2*(1-percent),bp2*(1+percent))

labels = [ 'Ps (Tool-ply)', 'Ft (Tool-ply)', 'C (Tool-ply)', 'Eta0 (Tool-ply)', 'Ap (Tool-ply)', 'Bp (Tool-ply)', 'Ps (Ply-ply)',         'Ft (Ply-ply)', 'C (Ply-ply)', 'Eta0 (Ply-ply)', 'Ap (Ply-ply)', 'Bp (Ply-ply)']

dist = cp.J(dist_Pc1, dist_Ft1, dist_C1, dist_eta01, dist_ap1, dist_bp1, dist_Pc2, dist_Ft2, dist_C2, dist_eta02, dist_ap2, dist_bp2)

degree = 2
orths=cp.expansion.stieltjes(degree, dist, normed=True)
print('Length of terms in PCE', len(orths))


# In[90]:


df0 = pd.read_excel('result1.xlsx',sheet_name='DP180',header=None)
#print(df0.head(),'\n',df0.tail())

l = []
for i in range(1,180+1):
    l.append(df0[i].drop(0,axis=0).to_list())

l = np.asarray(l)

test_train_ratio = 0.98

n = int(test_train_ratio*len(l))

train = l[0:n,:]
test = l[n+1:len(l),:]

train=train.T
test=test.T

print('Shape of the training set is {} and test set is {}'.format(np.shape(train),np.shape(test)))


# In[91]:


eval_train = var[0:n]

pce = cp.fit_regression(orths, train, eval_train)

err_train = []
err_train.append(eval_train-pce(*train))
mse = sum(err_train[0]*err_train[0])/len(err_train[0])
print('Coefficient of determination (R-squared) for training set: {}'.format(1-mse/eval_train.var(0)))

eval_test = var[n+1:len(var)]
err_test = []
err_test.append(eval_test-pce(*test))
mse = sum(err_test[0]*err_test[0])/len(err_test[0])
print('Coefficient of determination (R-squared) for test set: {}'.format(1-mse/eval_test.var(0)))


# In[92]:


T_sens = cp.Sens_t(pce, dist)
print(T_sens, sum(T_sens))
varnum = [n+1 for n in range(0,len(T_sens))]


# In[103]:


#fig, big_axes = plt.subplots( figsize=(15.0, 16.0) , nrows=2, ncols=2, sharey=False)
#fig, axd = plt.subplot_mosaic([['left', 'right'],['bottom', 'bottom']],
#                              constrained_layout=True)
fig, axd = plt.subplot_mosaic([['top'],['bottom']],
                              constrained_layout=True)

#for i, big_ax in enumerate(big_axes, start=1):
    # Turn off axis lines and ticks of the big subplot
#    big_ax.set_xticks([])
#    big_ax.set_yticks([])
#    big_ax._frameon = False

x_train = [i for i in range(1, len(eval_train)+1)]

#ax1 = fig.add_subplot(2,2,1)
axd['top'].plot(x_train, eval_train, 'o', label='Aniform evaluations',fillstyle='none',markersize=6)
axd['top'].plot(x_train, pce(*train), '*', label='Surrogate model', fillstyle='none',markersize=6)
#ax1.set_ylabel('Displacement (mm)')
axd['top'].set_ylabel('Number of nodes', fontsize=10)
axd['top'].set_xlabel('Design point number', fontsize=10)
axd['top'].tick_params(axis='both', which='both',direction='out',labelsize=10)
#ax1.set_title('Variation in maximum value of Z-axis displacement',fontsize=10)
axd['top'].set_title('Number of nodes in laminate exceeding a critical Z-axis displacement', fontsize=10)
axd['top'].xaxis.grid(True, linestyle='-',which ='major',alpha=1.0)
axd['top'].xaxis.grid(True, linestyle=':',which ='minor')
axd['top'].yaxis.grid(True, linestyle='-',which ='major',alpha=1.0)
axd['top'].yaxis.grid(True, linestyle=':',which ='minor')
axd['top'].minorticks_on()
axd['top'].legend(ncol=2, loc='upper center')

#x_test = [i for i in range(1, len(eval_test)+1)]

#ax1 = fig.add_subplot(2,2,2)
#axd['top'].plot(x_test, eval_test, 'o', x_test, pce(*test), '*')
#ax1.set_ylabel('Displacement (mm)')
#axd['top'].set_ylabel('Number of nodes', fontsize=10)
#axd['top'].set_xlabel('Design point number', fontsize=10)
#axd['top'].tick_params(axis='both', which='both',direction='out',labelsize=10)
#ax1.set_title('Variation in maximum value of Z-axis displacement',fontsize=14)
#axd['top'].set_title('Number of nodes in laminate exceeding a critical Z-axis displacement', fontsize=14)
#axd['top'].xaxis.grid(True, linestyle='-',which ='major',alpha=1.0)
#axd['top'].xaxis.grid(True, linestyle=':',which ='minor')
#axd['top'].yaxis.grid(True, linestyle='-',which ='major',alpha=1.0)
#axd['top'].yaxis.grid(True, linestyle=':',which ='minor')
#axd['top'].minorticks_on()

#ax4 = fig.add_subplot(2,2,3)
axd['bottom'].bar(varnum, T_sens)
axd['bottom'].set_xlabel('Penalty polymer friction parameters',fontsize=10)
axd['bottom'].set_ylabel('Sensitivity Index',fontsize=10)
axd['bottom'].set_xticks(varnum, minor=False)
axd['bottom'].set_xticklabels( labels, rotation = 90, ha="right", rotation_mode="anchor")
axd['bottom'].tick_params(axis='both', which='both',direction='out',labelsize=10)
axd['bottom'].xaxis.grid(True, linestyle='-',which ='major',alpha=1.0)
axd['bottom'].xaxis.grid(True, linestyle=':',which ='minor')
axd['bottom'].yaxis.grid(True, linestyle='-',which ='major',alpha=1.0)
axd['bottom'].yaxis.grid(True, linestyle=':',which ='minor')
plt.savefig('{}_order_{}_DPs_{}_Sobol.png'.format('Node-number', degree, len(var)),dpi=500,pad_inches=0.01)
plt.show()


# In[ ]:
