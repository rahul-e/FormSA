import os
import sys
import shutil
from subprocess import call
import chaospy as cp
import numpy as np
import pandas as pd
import re


def read_AniformModel():


		lines=pd.read_csv(filename+'.afi',delimiter='\t',header=None)
		df = pd.DataFrame(lines)#.astype('string')
		df.columns=['A']
		tmpdf=df.A.str.split(expand=True).fillna('%')
		maxrowlen = 0
		for i in range(0, df.shape[0]):
			if(maxrowlen<len(tmpdf.iloc[i])):
				maxrowlen = len(tmpdf.iloc[i])
				maxrownum = i
		col=['col'+str(i) for i in range(0,maxrowlen)]
		tmpdf.columns=col

		# Penalty Polymer friction
		Ps = [] # Penalty stiffness
		tau0 = []
		c0 = []
		Ft = [] # Film thickness
		C = [] # Another param in Power Law viscosity
		Eta0 = []
		n = []
		ap = []
		bp = []
		p0 = []
		if len(tmpdf['col2'].str.match('PenaltyPolymerILD')):
			PenPolyNum = tmpdf[tmpdf['col2'].str.match('PenaltyPolymerILD')].index
			for i in PenPolyNum:
				Ps.append(tmpdf['col1'].iloc[i+1]) #=tmpdf['col1'].iloc[i+1]+0.1*tmpdf['col1'].iloc[i+1] # Penalty stiffness
				tau0.append(tmpdf['col1'].iloc[i+2])
				tau0.append(tmpdf['col1'].iloc[i+3])
				tau0.append(tmpdf['col1'].iloc[i+4])
				C.append(tmpdf['col1'].iloc[i+5]) #tmpdf['col1'].iloc[i+5]+0.1*tmpdf['col1'].iloc[i+5] # C
				Eta0.append(tmpdf['col1'].iloc[i+6])
				n.append(tmpdf['col1'].iloc[i+7])
				ap.append(tmpdf['col1'].iloc[i+8])
				bp.append(tmpdf['col1'].iloc[i+9])
				p0.append(tmpdf['col1'].iloc[i+10])

		Cond = [] # Conduction coefficient
		Conv = [] # Convection coefficient
		Emi = [] # Emmisivity
		EmiO = []	# Emissivity Other

		if len(tmpdf['col2'].str.match('ConductanceIT')):
			CondNum = tmpdf[tmpdf['col2'].str.match('ConductanceIT')].index
			for i in CondNum:
				Cond.append(tmpdf['col1'].iloc[i+2])

		if len(tmpdf['col2'].str.match('ConvectionIT')):
			ConvNum = tmpdf[tmpdf['col2'].str.match('ConvectionIT')].index
			for i in ConvNum:
				Conv.append(tmpdf['col1'].iloc[i+2])

		if len(tmpdf['col2'].str.match('RadiationIT')):
			RadNum = tmpdf[tmpdf['col2'].str.match('RadiationIT')].index
			for i in RadNum:
				Emi.append(tmpdf['col1'].iloc[i+3])
				EmiO.append(tmpdf['col1'].iloc[i+4])

		# IsoElastic material model for inplane and bending
		Rho = [] # density
		E = [] # Young
		nu = [] # poisson
		# OrthoElastic material model for inplane and bending
		E1 = []
		E2 = []
		nu12 = []
		G12 = []

		if len(tmpdf['col2'].str.match('IsoElasticLD')):
			IsoElasNum = tmpdf[tmpdf['col2'].str.match('IsoElasticLD')].index
			for i in IsoElasNum:
				if len(tmpdf['col0'].str.match('density')):
					Rho.append(tmpdf['col1'].iloc[i+1])
					E.append(tmpdf['col1'].iloc[i+2])
					nu.append(tmpdf['col1'].iloc[i+3])
				else:
					E.append(tmpdf['col1'].iloc[i+1])
					nu.append(tmpdf['col1'].iloc[i+2])

		if len(tmpdf['col2'].str.match('OrthoElasticLD')):
			OrtElasNum = tmpdf[tmpdf['col2'].str.match('OrthoElasticLD')].index
			for i in OrtElasNum:
				E1.append(tmpdf['col1'].iloc[i+1])
				E2.append(tmpdf['col1'].iloc[i+2])
				nu12.append(tmpdf['col1'].iloc[i+3])
				G12.append(tmpdf['col1'].iloc[i+4])

		# Cross Viscosity Fluid model
		CvEta0 = []
		EtaInf = []
		m = []
		n = []

		if len(tmpdf['col2'].str.match('ViscousCrossLD')):
			CrViscNum = tmpdf[tmpdf['col2'].str.match('ViscousCrossLD')].index
			for i in CrViscNum:
				CvEta0.append(tmpdf['col1'].iloc[i+2])
				EtaInf.append(tmpdf['col1'].iloc[i+3])
				m.append(tmpdf['col1'].iloc[i+4])
				n.append(tmpdf['col1'].iloc[i+5])

		# Adhesion between plies
		tension = []
		if len(tmpdf['col2'].str.match('AdhesionILD')):
			AdhNum = tmpdf[tmpdf['col2'].str.match('AdhesionILD')].index
			for i in AdhNum:
				tension.append(tmpdf['col1'].iloc[i+1])

		with open('model'+str(sample_num)+'.afi','w') as f:
			dfAsString=tmpdf.to_string(header=False, index=False)
			f.write(dfAsString)


		#print(Ps,Cond,Conv)
			#print(tmpdf.col0.values,file=f) #for k in range(0,df.shape[0]
		#for k in u:
			#print(s.loc[k])
			#print(s.loc[k].reset_index(drop=True))
			#print(s.loc['*material'].reset_index(drop=True))
			#print(k)

		#tmpdf['col1'].iloc[i+1]=cp.Uniform(-0.1,0.1).sample(1,"R")[0]*Ps[0]+Ps[0]
		#tmpdf['col1'].iloc[i+6]+0.1*tmpdf['col1'].iloc[i+6] # Eta0
		#params = ['material','tau0', 'c0', 'filmthickness', 'C', 'eta0', 'n', 'ap', \
		#'bp', 'p0', 'alpha', 'coefficient', 'Tinf']
		#s = pd.Series(tmpdf['col1'].values, tmpdf['col0'].values)
		#u = np.unique(s.index.values).tolist()
		#print(u)
		#tmpDF=pd.concat([s.loc[k].reset_index(drop=True) for k in u], axis=1, keys=u)
		#print(maxrowlen,maxrownum)
def createDesignMatrix(sample_num):
		distribution=cp.Uniform()

def run_Aniform(sampleID):

		read_AniformModel()

		inpfilename=filename+'.afi'

		file_path = "./sample_%.0f/file.txt" %(sampleID)
		directory = os.path.dirname(file_path)


		if not os.path.exists(directory): # check if the sample exist

			os.makedirs(directory) # make directory for the sample

		path = "./sample_%.0f"%(sampleID)

		wd = os.getcwd()
		meshsubdir = wd+'meshes'
        #print('Current working directory',meshsubdir)
		print('Current working directory',wd)
		try:
			shutil.copy2(os.path.join(wd,inpfilename), path) # copy inputfile to the sample subdir
			shutil.copytree(meshsubdir, path+'/meshes') # copy mesh directory into the sample directory
		except:
			print('Could not copy all files')

		os.chdir(path)
		print('Current working directory',os.getcwd())
		print('Aniform output file exist?',)
		print('Running Aniform')
		#call("AniformP " +inpfilename,shell=True)

		os.chdir('..')
		print('Current working directory',os.getcwd())

def main(args):

		global sample_num

		global filename

		filename = args[-1]
		sample_num = int(args[-2])

		print('Total number of samples', sample_num)

		samples=[i for i in range(0, sample_num)]

		for s in samples:
				run_Aniform(s+1)

if __name__ == '__main__':

		main(sys.argv)
