import os
#import subprocess
import pandas as pd
#from openpyxl import load_workbook

print('Current dir:',os.getcwd())
for i in range(1,50):

        path = 'sample_'+str(i)+'/results/'

        lines = pd.read_csv(path+'Strain.txt',delimiter='\t',header=None)
        df = pd.DataFrame(lines)
        df.columns = ['A']
        tmpdf = df.A.str.split(expand=True).fillna('%')
        x = tmpdf[tmpdf[1].str.fullmatch('results')].index
        Sxx = tmpdf.iloc[x[-8]+1:x[-7],1].astype('float')
        Syy = tmpdf.iloc[x[-8]+1:x[-7],2].astype('float')
        Sxy = tmpdf.iloc[x[-8]+1:x[-7],3].astype('float')

        print('Sample',i,'\n',Sxx.head(),'\n',Sxx.tail())

        with open('MaxStrain.txt','a') as f:
            f.write('%0.d \t %.4f \t %.4f \t %.4f \n' %(i, Sxx.max(axis=0),Syy.max(axis=0),Sxy.max(axis=0)))
        f.close()
