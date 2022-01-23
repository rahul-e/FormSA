import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from openpyxl import load_workbook
filename='cylinder'
print('Current dir:',os.getcwd())
numsample = 76
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

filename = 'model'
rc=[]
for i in range(start,start+numsample):
        ax = 'ax'+str(i)
        ax = fig.add_subplot(numsample,1,(i-start+1))

        cnt = 0
        path = 'sample_'+str(i)+'/results/'
        outfilename = path+'/Crystal.txt'
        if not os.path.exists(outfilename):
            subprocess.call('Aniform.GUI '+path+filename+str(i)+'.afs',shell=True)
        lines = pd.read_csv(path+'Crystal.txt',delimiter='\t',header=None)
        df = pd.DataFrame(lines)
        df.columns = ['A']
        tmpdf = df.A.str.split(expand=True).fillna('%')
        x = tmpdf[tmpdf[1].str.fullmatch('results')].index
        #if (i==1):
        #    RC180 = tmpdf.iloc[x[-8]+1:x[-7],1].astype('float')
        #Syy = tmpdf.iloc[x[-8]+1:x[-7],2].astype('float')
        #Sxy = tmpdf.iloc[x[-8]+1:x[-7],3].astype('float')
        #else:
        RC = tmpdf.iloc[x[-8]+1:x[-7],1].astype('float')
        DelRC = abs(RC)
        n = ax.hist(DelRC, bins=[0.0,0.01,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.98,1.0], range=(0.0, 1.0))
        #print('Sample num is %0.d and the number of nodes is %0.d' %(i, sum(n[0])-n[0][-1]))
        rc.append(sum(n[0])-n[0][-1])

        #print('Sample',i,'\n')
with open('RC.txt','w') as f:
        for i in rc: f.write('%0.d\n' %(i))
f.close()


df = pd.read_csv('DesignPoints.txt',header=None,delimiter='\t',engine='python')
df=df.iloc[1].dropna(how='all')
#df.columns=['Col2']
df= df.str.split(expand=True)
df = df.replace(',','', regex=True)
df = df.replace('\[','', regex=True)
df = df.replace('\]','', regex=True)
x=df.iloc[0,1:101].to_numpy()
label = [np.round(float(u),2) for u in x]
#
lines = pd.read_csv('RC.txt',header=None,delimiter='\t',engine='python')
rc=pd.DataFrame(lines)
rc.columns=['N']
rc['T'] = label[0:76]
rc = rc.sort_values(by='T',axis=0)
x = [i for i in range(0,76)]
fig = plt.figure(figsize=(15,15))
plt.plot(x, rc['N'], 'o')
plt.xticks(x,rc['T'], rotation = 'vertical',fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Tool temperature (°C)',fontsize=12)
plt.ylabel('Number of nodes',fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig('Recrystal_node_number.png',dpi=500)
plt.show()
