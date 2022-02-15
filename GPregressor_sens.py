#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import GPy
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity

np.random.seed(10)
# In[56]:


df0 = pd.read_excel('result1.xlsx',sheet_name='DP180',header=None)
#print(df0.head(),'\n',df0.tail())

l = []
for i in range(1,180+1):
    l.append(df0[i].drop(0,axis=0).to_list())

l = np.asarray(l)

test_train_ratio = 0.9

n = int(test_train_ratio*len(l))

train = l[0:n,:]
test = l[n:len(l),:]

df = pd.read_csv('DispZ_NumNode_maxdisp.txt',header=None,delimiter='\t')
df=df.dropna(how='all')
df.columns=['Number_of_nodes', 'Max'  ]
print(df.head())

NumN_train = df['Number_of_nodes'][0:n]
NumN_test = df['Number_of_nodes'][-(df.shape[0]-n):]

x_plot_train=df.index[0:n]
x_plot_test=df.index[-(df.shape[0]-n):]

MaxDisp = df['Max']
#MaxDisp=np.array([MaxDisp])
#NumN=np.array([NumN])


print(train.shape, MaxDisp.shape)
model_gpy = GPRegression(train,np.array([NumN_train]).T,GPy.kern.RBF(12, lengthscale=0.08, variance=20), noise_var=1e-10)
model_emukit = GPyModelWrapper(model_gpy)


# In[58]:


from emukit.test_functions.sensitivity import Ishigami ### change this one for the one in the library
from emukit.core import ContinuousParameter, ParameterSpace

#target_simulator = ishigami.fidelity1
#variable_domain = (-np.pi,np.pi)
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


space = ParameterSpace([ContinuousParameter('Ps1', Ps1*0.9, Ps1*1.1),
                        ContinuousParameter('Ft1', Ft1*0.9, Ft1*1.1),
                        ContinuousParameter('C1', C1*0.9, C1*1.1),
                        ContinuousParameter('eta01', eta01*0.9, eta01*1.1),
                        ContinuousParameter('ap1', ap1*0.9, ap1*1.1),
                        ContinuousParameter('bp1', bp1*0.9, bp1*1.1),
                        ContinuousParameter('Ps2', Ps2*0.9, Ps2*1.1),
                        ContinuousParameter('Ft2', Ft2*0.9, Ft2*1.1),
                        ContinuousParameter('C2', C2*0.9, C2*1.1),
                        ContinuousParameter('eta02', eta02*0.9, eta02*1.1),
                        ContinuousParameter('ap2', ap2*0.9, ap2*1.1),
                        ContinuousParameter('bp2', bp2*0.9, bp2*1.1)])


# In[59]:


senstivity_gpbased = MonteCarloSensitivity(model = model_emukit, input_domain = space)
main_effects_gp, total_effects_gp, _ = senstivity_gpbased.compute_effects(num_monte_carlo_points = 1000000)


# In[92]:


mu_plot, var_plot = model_emukit.predict(train)

mu_plot_test, var_plot_test = model_emukit.predict(test)

d = {'GP Monte Carlo main': main_effects_gp,
     'GP Monte Carlo total': total_effects_gp}

df = pd.DataFrame(d) #.plot(kind='bar',figsize=(12, 5))
print(df)
#plt.title('First-order Sobol indexes', fontsize=10)
#plt.ylabel('% of explained output variance',fontsize=8);
labels = [ 'Ps (T-ply)', 'Ft (T-ply)', 'C (T-ply)', 'Eta0 (T-ply)', 'Ap (T-ply)', 'Bp (T-ply)', \
            'Ps (P-ply)', 'Ft (P-ply)', 'C (P-ply)', 'Eta0 (P-ply)', 'Ap (P-ply)', \
            'Bp (P-ply)']
y = df['GP Monte Carlo total'].values

x = [i for i in range(0,len(labels))]
y_ = np.concatenate(y)

gp_pred_train=np.concatenate(mu_plot)
gp_pred_test=np.concatenate(mu_plot_test)
#gp_pred_floor = [np.floor(elem) for elem in gp_prediction]
#gp_pred_ceil = [np.ceil(elem) for elem in gp_prediction]
err_train = NumN_train-gp_pred_train
mse = sum(err_train*err_train)/len(err_train)
print('Coefficient of determination (R-squared) for training set: {}'.format(1-mse/NumN_train.var(0)))

err_test = NumN_test-gp_pred_test
mse = sum(err_test*err_test)/len(err_test)
print('Coefficient of determination (R-squared) for test set: {}'.format(1-mse/NumN_test.var(0)))


fig, axd = plt.subplot_mosaic([['top'],['bottom']],
                              constrained_layout=True)

#ax1 = fig.add_subplot(2,2,1)
axd['top'].plot(x_plot_train, NumN_train.T, 'ob', label='Aniform',fillstyle='none',markersize=6)
axd['top'].plot(x_plot_train, mu_plot, '*k', label='Surrogate', fillstyle='none',markersize=6)
axd['top'].plot(x_plot_test, NumN_test.T, 'ob',fillstyle='none',markersize=6)
axd['top'].plot(x_plot_test, mu_plot_test, '*r', fillstyle='none',markersize=6)

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
axd['top'].legend(ncol=2, loc='best',markerscale=0.6,handletextpad=0.1, \
            columnspacing=0.1, borderaxespad=0.2, frameon=False)


axd['bottom'].bar(x, y_)
axd['bottom'].set_xlabel('Penalty polymer friction parameters',fontsize=10)
axd['bottom'].set_ylabel('Sensitivity Index',fontsize=10)
axd['bottom'].set_xticks(x, minor=False)
axd['bottom'].set_xticklabels( labels, rotation = 90, ha="right", rotation_mode="anchor")
axd['bottom'].tick_params(axis='both', which='both',direction='out',labelsize=10)
axd['bottom'].xaxis.grid(True, linestyle='-',which ='major',alpha=1.0)
axd['bottom'].xaxis.grid(True, linestyle=':',which ='minor')
axd['bottom'].yaxis.grid(True, linestyle='-',which ='major',alpha=1.0)
axd['bottom'].yaxis.grid(True, linestyle=':',which ='minor')
plt.savefig('{}_{}_DPs_{}_Sobol_crossvalidated.png'.format('Node-number', 'GP', len(x_plot_train)),dpi=500,pad_inches=0.01)
plt.show()
