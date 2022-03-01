#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import GPy
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(10)



df0 = pd.read_excel('result1.xlsx',sheet_name='DP180',header=None)
df = df0.drop(0,axis=0).transpose()
labels = [ 'Ps (T-ply)', 'Ft (T-ply)', 'C (T-ply)', 'Eta0 (T-ply)', 'Ap (T-ply)', 'Bp (T-ply)', \
            'Ps (P-ply)', 'Ft (P-ply)', 'C (P-ply)', 'Eta0 (P-ply)', 'Ap (P-ply)', \
            'Bp (P-ply)']
df.columns=labels
inp_dim = 12
X=df.loc[1:180,labels]

test_train_ratio = 0.9

n = int(test_train_ratio*X.shape[0])


df = pd.read_csv('DispZ_NumNode_maxdisp.txt',header=None,delimiter='\t')
df=df.dropna(how='all')
df.columns=['Number_of_nodes', 'Max'  ]
print(df.head())
y=df['Number_of_nodes'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=1-test_train_ratio, random_state=1)


print(np.shape(X_train),np.shape(np.array([Y_train]).T))

#scaler = StandardScaler().fit(X_train)
#X_train_scaled = pd.DataFrame(scaler.transform(X_train))
#X_test_scaled = pd.DataFrame(scaler.transform(X_test))

x_plot_train=[i for i in X_train.index]
x_plot_test=[i for i in X_test.index]

MaxDisp = df['Max']
#MaxDisp=np.array([MaxDisp])
#NumN=np.array([NumN])


#print(train.shape, MaxDisp.shape)
#model_gpy = GPRegression(X_train,np.array([Y_train]).T,GPy.kern.RBF(12, lengthscale=0.08, variance=20), noise_var=1e-10)
model_gpy = GPRegression(X_train,np.array([Y_train]).T,GPy.kern.RBF(inp_dim, lengthscale=0.15, variance=3),\
            normalizer=False, noise_var=1e1)
model_emukit = GPyModelWrapper(model_gpy)



from emukit.core import ContinuousParameter, ParameterSpace


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
#space = ParameterSpace([ContinuousParameter('Ps1', X_train_scaled.max(0)[1], X_train_scaled.min(0)[1]),
#                        ContinuousParameter('Ft1', X_train_scaled.max(0)[1], X_train_scaled.min(0)[1]),
#                        ContinuousParameter('C1', X_train_scaled.max(0)[1], X_train_scaled.min(0)[1]),
#                        ContinuousParameter('eta01', X_train_scaled.max(0)[1], X_train_scaled.min(0)[1]),
#                        ContinuousParameter('ap1', X_train_scaled.max(0)[1], X_train_scaled.min(0)[1]),
#                        ContinuousParameter('bp1', X_train_scaled.max(0)[1], X_train_scaled.min(0)[1]),
#                        ContinuousParameter('Ps2', X_train_scaled.max(0)[1], X_train_scaled.min(0)[1]),
#                        ContinuousParameter('Ft2', X_train_scaled.max(0)[1], X_train_scaled.min(0)[1]),
#                        ContinuousParameter('C2', X_train_scaled.max(0)[1], X_train_scaled.min(0)[1]),
#                        ContinuousParameter('eta02', X_train_scaled.max(0)[1], X_train_scaled.min(0)[1]),
#                        ContinuousParameter('ap2', X_train_scaled.max(0)[1], X_train_scaled.min(0)[1]),
#                        ContinuousParameter('bp2', X_train_scaled.max(0)[1], X_train_scaled.min(0)[1])
#                        ])

senstivity_gpbased = MonteCarloSensitivity(model = model_emukit, input_domain = space)
main_effects_gp, total_effects_gp, _ = \
            senstivity_gpbased.compute_effects(num_monte_carlo_points = 1000000)


mu_plot, var_plot = model_emukit.predict(np.array(X_train))

mu_plot_test, var_plot_test = model_emukit.predict(np.array(X_test))

d = {'GP Monte Carlo main': main_effects_gp,
     'GP Monte Carlo total': total_effects_gp}

df = pd.DataFrame(d) #.plot(kind='bar',figsize=(12, 5))
print('***Sobol Indices*** \n',df)
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
err_train = Y_train-gp_pred_train
mse = sum(err_train*err_train)/len(err_train)
print('Coefficient of determination (R-squared) for training set: {}'.format(1-mse/Y_train.var(0)))

err_test = Y_test-gp_pred_test
mse = sum(err_test*err_test)/len(err_test)
print('Coefficient of determination (R-squared) for test set: {}'.format(1-mse/Y_test.var(0)))


fig, axd = plt.subplot_mosaic([['top'],['bottom']],figsize=(10,12),
                              constrained_layout=True)

axd['top'].plot(x_plot_train, Y_train.T, 'ob', label='Aniform (train)',fillstyle='none',markersize=8)
axd['top'].plot(x_plot_train, mu_plot, '*k', label='Surrogate (train)', fillstyle='none',markersize=8)
axd['top'].plot(x_plot_test, Y_test.T, 'or', label='Aniform (test)', fillstyle='none',markersize=8)
axd['top'].plot(x_plot_test, mu_plot_test, '*r', label='Surrogate (test)',fillstyle='none',markersize=8)

axd['top'].set_ylabel('Number of nodes', fontsize=18)
axd['top'].set_xlabel('Design point number', fontsize=18)
axd['top'].tick_params(axis='both', which='both',direction='out',labelsize=14)

axd['top'].set_title('Number of nodes in laminate exceeding a chosen Z-axis displacement', fontsize=18)
axd['top'].xaxis.grid(True, linestyle='-',which ='major',alpha=1.0)
axd['top'].xaxis.grid(True, linestyle=':',which ='minor')
axd['top'].yaxis.grid(True, linestyle='-',which ='major',alpha=1.0)
axd['top'].yaxis.grid(True, linestyle=':',which ='minor')
axd['top'].minorticks_on()
axd['top'].legend(ncol=3, loc='best',markerscale=1.0,handletextpad=0.1, \
            columnspacing=0.1, borderaxespad=0.2, frameon=False)


axd['bottom'].bar(x, y_)
axd['bottom'].set_xlabel('Penalty polymer friction parameters',fontsize=18)
axd['bottom'].set_ylabel('Sensitivity Index',fontsize=18)
axd['bottom'].set_xticks(x, minor=False)
axd['bottom'].set_xticklabels( labels, rotation = 90, ha="right", rotation_mode="anchor")
axd['bottom'].tick_params(axis='both', which='both',direction='out',labelsize=14)
axd['bottom'].xaxis.grid(True, linestyle='-',which ='major',alpha=1.0)
axd['bottom'].xaxis.grid(True, linestyle=':',which ='minor')
axd['bottom'].yaxis.grid(True, linestyle='-',which ='major',alpha=1.0)
axd['bottom'].yaxis.grid(True, linestyle=':',which ='minor')
plt.savefig('{}_{}_DPs_{}_Sobol_crossvalidated.png'.format('Node-number', 'GP', len(x_plot_train)),dpi=500,pad_inches=0.01)
plt.show()
