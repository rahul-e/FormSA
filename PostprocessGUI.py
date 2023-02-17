#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from openpyxl import load_workbook
filename='cylinder'
print('Current dir:',os.getcwd())
numsample = 180
start=1
fig, big_axes = plt.subplots(figsize=(10.0, 15.0), nrows=numsample, ncols=1, sharey=False)
for i, big_ax in enumerate(big_axes, start=1):
    # big_ax.set_title("Damping factor = %s" % DF[i], fontsize=16)
    # Turn off axis lines and ticks of the big subplot
    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax._frameon = False

#filename = 'cylinder'
rc=[]
max_RC=[]
for i in range(start,start+numsample):
        ax = 'ax'+str(i)
        ax = fig.add_subplot(numsample,1,(i-start+1))

        cnt = 0
        path = 'sample_'+str(i)+'/results/'
        outfilename = path+'/Displacement.txt'
        if not os.path.exists(outfilename):
            subprocess.call('Aniform.GUI '+path+filename+str(i)+'.afs',shell=True)
        lines = pd.read_csv(path+'Displacement.txt',delimiter='\t',header=None)
        df = pd.DataFrame(lines)
        df.columns = ['A']
        tmpdf = df.A.str.split(expand=True).fillna('%')
        x = tmpdf[tmpdf[1].str.fullmatch('results')].index

        RC = tmpdf.iloc[x[0]+1:x[1],3].astype('float')
        max_RC.append(max(RC))
        n = ax.hist(RC, bins=[0.0,2.0,4.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0,22.0,24.0,26.0], range=(0.0, 25.0))
        ax.grid(which='both')
        ax.minorticks_on()
        rc.append(sum(n[0][-12:]))

with open('DispZ_NumNode_maxdisp.txt','w') as f:
        for i in range(0, numsample): f.write('%0.d\t %0.4f \n' %(rc[i], max_RC[i]))
f.close()


df = pd.read_csv('DesignPoints.txt',header=None,delimiter='\t',engine='python')
df=df.iloc[1].dropna(how='all')

df= df.str.split(expand=True)
df = df.replace(',','', regex=True)
df = df.replace('\[','', regex=True)
df = df.replace('\]','', regex=True)
x=df.iloc[0,1:101].to_numpy()

lines = pd.read_csv('DispZ_NumNode_maxdisp.txt',header=None,delimiter='\t',engine='python')
rc=pd.DataFrame(lines)
rc.columns=['N', 'Max']

x = [i for i in range(0,numsample)]
fig = plt.figure(figsize=(15,15))
plt.plot(x, rc['Max'], 'o')
#plt.xticks(x,rc['T'], rotation = 'vertical',fontsize=12)
plt.yticks(fontsize=12)
#plt.xlabel('Tool temperature (°C)',fontsize=12)
plt.ylabel('Displacement (mm) ',fontsize=12)
plt.title('Maximum Z axis displacement')
plt.grid()
plt.tight_layout()
plt.savefig('DispZ_node_number.png',dpi=500)
plt.show()
