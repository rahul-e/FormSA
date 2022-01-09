import os
import sys
import shutil
from subprocess import call
import chaospy as cp
import numpy as np
import pandas as pd
import re


def read_AniformModel():

		global Param, tmpdf
		# Ps, tau0, c0, Ft, C, Eta0, n, ap, bp, p0, \
		# Cond, Conv, Emi, EmiO, \
		# Rho, E, nu, \
		# E1, E2, G12, nu12, \
		# CvEta0, EtaInf, m, CVFn, tension, tmpdf, Param

		Param = dict() #  Dictionary of parameters

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
		Param["Ps"]=[] # Penalty stiffness
		Param["tau0"] = []
		Param["c0"] = []
		Param["Ft"] = [] # Film thickness
		Param["C"] = [] # Another param in Power Law viscosity
		Param["eta0"] = []
		Param["n"] = []
		Param["ap"] = []
		Param["bp"] = []
		Param["p0"] = []
		if len(tmpdf['col2'].str.fullmatch('PenaltyPolymerILD')):
			PenPolyNum = tmpdf[tmpdf['col2'].str.fullmatch('PenaltyPolymerILD')].index
			for i in PenPolyNum:
				Param["Ps"].append(tmpdf['col1'].iloc[i+1]) #=tmpdf['col1'].iloc[i+1]+0.1*tmpdf['col1'].iloc[i+1] # Penalty stiffness
				Param["tau0"].append(tmpdf['col1'].iloc[i+2])
				Param["c0"].append(tmpdf['col1'].iloc[i+3])
				Param["Ft"].append(tmpdf['col1'].iloc[i+4])
				Param["C"].append(tmpdf['col1'].iloc[i+5]) #tmpdf['col1'].iloc[i+5]+0.1*tmpdf['col1'].iloc[i+5] # C
				Param["eta0"].append(tmpdf['col1'].iloc[i+6])
				Param["n"].append(tmpdf['col1'].iloc[i+7])
				Param["ap"].append(tmpdf['col1'].iloc[i+8])
				Param["bp"].append(tmpdf['col1'].iloc[i+9])
				Param["p0"].append(tmpdf['col1'].iloc[i+10])

		#for i in range(0,len(Ps)):
		#	Param["Ps"].append(Ps[i])
		#	Param["tau0"].append(tau0[i])

		Cond = [] # Conduction coefficient
		Conv = [] # Convection coefficient
		Emi = [] # Emmisivity
		EmiO = []	# Emissivity Other

		if len(tmpdf['col2'].str.fullmatch('ConductanceIT')):
			CondNum = tmpdf[tmpdf['col2'].str.fullmatch('ConductanceIT')].index
			for i in CondNum:
				Cond.append(tmpdf['col1'].iloc[i+2])

		if len(tmpdf['col2'].str.fullmatch('ConvectionIT')):
			ConvNum = tmpdf[tmpdf['col2'].str.fullmatch('ConvectionIT')].index
			for i in ConvNum:
				Conv.append(tmpdf['col1'].iloc[i+2])

		if len(tmpdf['col2'].str.fullmatch('RadiationIT')):
			RadNum = tmpdf[tmpdf['col2'].str.fullmatch('RadiationIT')].index
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

		if len(tmpdf['col2'].str.fullmatch('IsoElasticLD')):
			IsoElasNum = tmpdf[tmpdf['col2'].str.fullmatch('IsoElasticLD')].index
			for i in IsoElasNum:
				if len(tmpdf['col0'].str.fullmatch('density')):
					Rho.append(tmpdf['col1'].iloc[i+1])
					E.append(tmpdf['col1'].iloc[i+2])
					nu.append(tmpdf['col1'].iloc[i+3])
				else:
					E.append(tmpdf['col1'].iloc[i+1])
					nu.append(tmpdf['col1'].iloc[i+2])

		if len(tmpdf['col2'].str.fullmatch('OrthoElasticLD')):
			OrtElasNum = tmpdf[tmpdf['col2'].str.fullmatch('OrthoElasticLD')].index
			for i in OrtElasNum:
				E1.append(tmpdf['col1'].iloc[i+1])
				E2.append(tmpdf['col1'].iloc[i+2])
				nu12.append(tmpdf['col1'].iloc[i+3])
				G12.append(tmpdf['col1'].iloc[i+4])

		# Cross Viscosity Fluid model
		CvEta0 = []
		EtaInf = []
		m = []
		CVFn = []

		if len(tmpdf['col2'].str.fullmatch('ViscousCrossLD')):
			CrViscNum = tmpdf[tmpdf['col2'].str.fullmatch('ViscousCrossLD')].index
			for i in CrViscNum:
				CvEta0.append(tmpdf['col1'].iloc[i+2])
				EtaInf.append(tmpdf['col1'].iloc[i+3])
				m.append(tmpdf['col1'].iloc[i+4])
				CVFn.append(tmpdf['col1'].iloc[i+5])

		# Adhesion between plies
		tension = []
		if len(tmpdf['col2'].str.fullmatch('AdhesionILD')):
			AdhNum = tmpdf[tmpdf['col2'].str.fullmatch('AdhesionILD')].index
			for i in AdhNum:
				tension.append(tmpdf['col1'].iloc[i+1])

def createDesignMatrix():
		global DPdf
		DesgnPnt = dict()

		percent = 0.1
		Udist=cp.Uniform(-1*percent,1*percent)
		Ndist=cp.Normal()

		print('Current working directory',os.getcwd())

		text=pd.read_csv('MaterialmodelSens.txt',delimiter='\t',header=None)
		df = pd.DataFrame(text).astype('string')
		df.columns=['A']
		df=df.A.str.split(expand=True)
		# print(len(df.iloc[1]))
		Mean=[]

		for i in range(0,df.shape[0]):
			line=df.iloc[i].to_list()
			if 'PenaltyPolymer' in line:
				line.remove('PenaltyPolymer')
				dim=len(line)
				for l in line: DesgnPnt[l]=[]
				print('Number of uncertain variables in Penalty Polymer model', dim)
				#dist = cp.Iid(Udist,dim)

				for l in line:
					if len(set(Param[l]))==1:
						Mean=float(Param[l][0])
						DesgnPnt[l].append((1+Udist.sample(sample_num,'R'))*Mean)
					else:
						for j in range(0,len(Param[l])):
							Mean=float(Param[l][j])
							DesgnPnt[l].append((1+Udist.sample(sample_num,'R'))*Mean)
							#print(Mean,DesgnPnt)


		DPdf = pd.DataFrame.from_dict(DesgnPnt,orient='index')

		#dp = dp.append(DesgnPnt["Eta0"])
		print (DPdf.head())
		print (DPdf.tail())

def write_AniformModel(var):

		desgnId = var-1

		AnInpdf = pd.DataFrame()
		AnInpdf = tmpdf


		print (AnInpdf.head())
		print (AnInpdf.tail())

		params = DPdf.index
		params = list(map(lambda x: x.replace('Ps', 'penaltystiffness'), params))
		params = list(map(lambda x: x.replace('Ft', 'filmthickness'), params))

		for parnam in params:

			if len(AnInpdf['col0'].str.fullmatch(parnam)):
				row = AnInpdf[AnInpdf['col0'].str.fullmatch(parnam)].index
				track=0
				for count, r in enumerate(row):
					#if (AnInpdf['col0'].iloc[row] == parnam)
					#print('OK')
					if parnam == 'penaltystiffness':
						try:
							print('\t Current value of',AnInpdf['col0'].iloc[r], AnInpdf['col1'].iloc[r])
							AnInpdf['col1'].iloc[r] = DPdf[count].loc['Ps'][desgnId]
							print('\t New value of', AnInpdf['col0'].iloc[r], AnInpdf['col1'].iloc[r])
						except:
							print("'None' type object found")
							AnInpdf['col1'].iloc[r] = DPdf[0].loc['Ps'][desgnId] # It is assumed here \
							# that in subsequent occurences of parnam in Input file, the corresponding \
							# value of the parameter (Mean) is the same as that in the very first \
							# occurence. This needs reconsideration as it can result in a bug when, \
							# for eg: PenaltyPolymer is defined three times and for the first occurence \
							# value of Ps is X and for the two subsequent occurencea the value is Y
					elif parnam == 'filmthickness':
						try:
							print('\t Current value of',AnInpdf['col0'].iloc[r], AnInpdf['col1'].iloc[r])
							AnInpdf['col1'].iloc[r] = DPdf[count].loc['Ft'][desgnId]
							print('\t New value of', AnInpdf['col0'].iloc[r], AnInpdf['col1'].iloc[r])
						except:
							print("'None' type object found")
							AnInpdf['col1'].iloc[r] = DPdf[0].loc['Ft'][desgnId]

					elif parnam == 'eta0': #
						if AnInpdf['col2'].iloc[r-6] == 'PenaltyPolymerILD':
							print('Ok', AnInpdf['col2'].iloc[r-6], 'row', r)
							try:
								print('\t Current value of',AnInpdf['col0'].iloc[r], AnInpdf['col1'].iloc[r])
								AnInpdf['col1'].iloc[r] = DPdf[count-track].loc[parnam][desgnId]
								print('\t New value of', AnInpdf['col0'].iloc[r], AnInpdf['col1'].iloc[r])
							except:
								print("'None' type object found")
								AnInpdf['col1'].iloc[r] = DPdf[0].loc[parnam][desgnId]
						else:
							track = track+1
							print('This eta0 is associated with ViscousCrossLD.','track = %.0d' %(track))

					elif parnam == 'n': #

						if AnInpdf['col2'].iloc[r-7] == 'PenaltyPolymerILD':
							print('Ok', AnInpdf['col2'].iloc[r-7], 'row', r, parnam)
							try:
								print('\t Current value of',AnInpdf['col0'].iloc[r], AnInpdf['col1'].iloc[r])
								AnInpdf['col1'].iloc[r] = DPdf[count-track].loc[parnam][desgnId]
								print('\t New value of', AnInpdf['col0'].iloc[r], AnInpdf['col1'].iloc[r])
							except:
								print("'None' type object found")
								AnInpdf['col1'].iloc[r] = DPdf[0].loc[parnam][desgnId]
						else:
							track = track+1
							print("This 'n' is associated with ViscousCrossLD.",'track = %.0d' %(track))

					else:
						try:
							print('\t Current value of',AnInpdf['col0'].iloc[r], AnInpdf['col1'].iloc[r])
							AnInpdf['col1'].iloc[r] = DPdf[count].loc[parnam][desgnId]
							print('\t New value of', AnInpdf['col0'].iloc[r], AnInpdf['col1'].iloc[r])
						except:
							print("'None' type object found")
							AnInpdf['col1'].iloc[r] = DPdf[0].loc[parnam][desgnId]

		# Write input file for current design point
		print('Current working directory', os.getcwd())
		print('Changing to the subdirectory for current sample')

		file_path = "./sample_%.0f/file.txt" %(var)
		directory = os.path.dirname(file_path)

		if not os.path.exists(directory): # check if the sample exist
			os.makedirs(directory) # make directory for the sample
		os.chdir('sample_'+str(var))

		with open(filename+str(var)+'.afi','w') as f:
			dfAsString=AnInpdf.to_string(header=False, index=False)
			f.write(dfAsString)
		f.close()

		os.chdir('..')



def run_Aniform(sampleID):

		inpfilename=filename+'.afi'

		file_path = "./sample_%.0f/file.txt" %(sampleID)
		directory = os.path.dirname(file_path)


		if not os.path.exists(directory): # check if the sample exist

			os.makedirs(directory) # make directory for the sample

		path = "./sample_%.0f"%(sampleID)

		wd = os.getcwd()
		meshsubdir = wd+'\meshes'
        #print('Current working directory',meshsubdir)
		print('Current working directory',wd)
		try:
			shutil.copy2(os.path.join(wd,inpfilename), path) # copy inputfile to the sample subdir
			shutil.copytree(meshsubdir, path+'/meshes') # copy mesh directory into the sample directory
		except:
			print('Could not copy all files')

		os.chdir(path)
		print('Current working directory',os.getcwd())

		print('Check if Aniform output file exist?',)

		outfilename = filename+str(sampleID)+'.out'
		if not os.path.exists(outfilename):
				print('Running Aniform: filename', filename+str(sampleID)+'.afi')
				call("AniformP " +filename+str(sampleID)+'.afi',shell=True)
		else: print('Output file exists')

		os.chdir('..')
		print('Current working directory',os.getcwd())

def main(args):

		global sample_num

		global filename

		filename = args[-1]
		sample_num = int(args[-2]) # Total number of samples

		print('Aniform model input file', filename)
		print('Total number of samples', sample_num)

		read_AniformModel()

		createDesignMatrix()

		samples=[i for i in range(0, sample_num)]

		for s in samples:
				write_AniformModel(s+1)
				run_Aniform(s+1)

if __name__ == '__main__':

		main(sys.argv)
