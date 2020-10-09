#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:29:29 2019

@author: jtjanecek
"""
import argparse
import hdf5storage
import math
import numpy as np
import glob
import os
import sys
from collections import defaultdict
import csv
import pandas as pd
from scipy.stats import norm

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MDTO')
logger.setLevel(logging.DEBUG)

Z = norm.ppf

class MDTO_Group():
	def __init__(self, cli_args):
		''' Initialize a MDTO Group
		params: 
			cli_args: namespace with input and output directories
		'''
		self._id = cli_args.id
		self._input_dir = cli_args.input
		self._output_dir = cli_args.output

		self._stats = []
		self._diffusion_trials = []

		logger.info("Processing ...")
		for logfile in sorted(glob.glob(os.path.join(self._input_dir,'*_MDTO_log.txt'))):
			reader = MDTO_Log_Reader(logfile)
			if not reader.unreadable:
				self._stats.append(reader.GetStats())
				self._diffusion_trials.append(reader.GetDiffusionTrials())
	
	def save(self):
		logger.info("Saving ...")
		df = pd.DataFrame(self._stats)
		df.to_csv(os.path.join(self._output_dir, '{}_measures.csv'.format(self._id)), index=False)

		sub_ids = np.array([diffusion[0] for diffusion in self._diffusion_trials]).flatten().astype(float)
		all_trial_resp = np.array([diffusion[1] for diffusion in self._diffusion_trials])
		all_trial_conds = np.array([diffusion[2] for diffusion in self._diffusion_trials])

		data = {'subjList': sub_ids,
				'rt': all_trial_resp,
				'subList': all_trial_conds,
			}
		hdf5storage.write(data, '.',  os.path.join(self._output_dir,'{}_diffusion_trials'.format(self._id) + '.mat'), matlab_compatible=True)


class MDTO_Log_Reader():

	def __init__(self, logFileLoc: str):
		'''
		Params:
			logFileLoc: the path to the logfile
		'''

		self.logFileLoc = logFileLoc
		self.unreadable = False
		self.numTrials = 160
		
		# Check if the file exists
		if not os.path.exists(self.logFileLoc):
			raise Exception("Log file {} does not exist!".format(self.logFileLoc))
		
		self.subID = os.path.basename(logFileLoc).split("_")[0]

		# Trials 
		self.trials = self._readTrials()		

		self.CheckErrors()
			
		# Udpate to human readable values
		self.UpdateTrialTypes()
		self.UpdateResponses()

	def GetStats(self):
		'''
		Get LDI and DPrime for each valence
		Return:
			dict:
				keys:
					HSim, LSim, Foil
				Values:
					dict: keys: 'LDI', 'D'
		LDI = p(Lure CR)-p(Target Miss)
		'''
		if self.unreadable:
			return
		stats = {'subID': float(self.subID)}
		
		numHSimCR  = len([trial for trial in self.trials if trial['TrialType'] == 'HSim' and trial['TestResp'] == 'New'])
		numHSimAll = len([trial for trial in self.trials if trial['TrialType'] == 'HSim' and trial['TestResp'] != ''])
		numLSimCR  = len([trial for trial in self.trials if trial['TrialType'] == 'LSim' and trial['TestResp'] == 'New'])
		numLSimAll = len([trial for trial in self.trials if trial['TrialType'] == 'LSim' and trial['TestResp'] != ''])
		numFoilCR  = len([trial for trial in self.trials if trial['TrialType'] == 'Foil' and trial['TestResp'] == 'New'])
		numFoilAll = len([trial for trial in self.trials if trial['TrialType'] == 'Foil' and trial['TestResp'] != ''])

		
		numTargHit = len([trial for trial in self.trials if trial['TrialType'] == 'Targ' and trial['TestResp'] == 'Old'])
		numTargAll = len([trial for trial in self.trials if trial['TrialType'] == 'Targ' and trial['TestResp'] != ''])
		
		stats['HSimCR'] = numHSimCR / numHSimAll
		stats['LSimCR'] = numLSimCR / numLSimAll
		stats['FoilCR'] = numFoilCR / numFoilAll
		stats['AllLureCR'] = (numHSimCR + numLSimCR) / (numHSimAll + numLSimAll)

		stats['HSimFA'] = 1 - stats['HSimCR']
		stats['LSimFA'] = 1 - stats['LSimCR']
		stats['FoilFA'] = 1 - stats['FoilCR']
		stats['AllLureFA'] = 1 - stats['AllLureCR']
		
		stats['TargHR'] = numTargHit / numTargAll

		stats['LDI_HSim'] = stats['HSimCR'] - (1 - stats['TargHR'])
		stats['LDI_LSim'] = stats['LSimCR'] - (1 - stats['TargHR'])
		stats['LDI_Foil'] = stats['FoilCR'] - (1 - stats['TargHR'])
		stats['LDI_AllLure'] = stats['AllLureCR'] - (1 - stats['TargHR'])
		
		'''
		If Hit rate is 1, correct with this formula:
			(n - 0.5)/n
		If FA rate is 0, then correct with this formula:
			.5 / n
		'''
		# Check for 0/1 in FA or Hit Rate
		if stats['TargHR'] == 1.0:
			stats['TargHR'] = (numTargAll - .5) / numTargAll
		if stats['HSimFA'] == 0.0:
			stats['HSimFA'] = .5 / numHSimCR
		if stats['LSimFA'] == 0.0:
			stats['LSimFA'] = .5 / numLSimCR		
		if stats['FoilFA'] == 0.0:
			stats['FoilFA'] = .5 / numFoilCR
		if stats['AllLureFA'] == 0.0:
			stats['AllLureFA'] = .5 / (numHSimCR + numLSimCR)
			
		stats['D_Targ_HSim'] = Z(stats['TargHR']) - Z(stats['HSimFA'])
		stats['D_Targ_LSim'] = Z(stats['TargHR']) - Z(stats['LSimFA'])
		stats['D_Targ_Foil'] = Z(stats['TargHR']) - Z(stats['FoilFA'])
		stats['D_Targ_AllLure'] = Z(stats['TargHR']) - Z(stats['AllLureFA'])

		stats['D_HSim_Foil'] = Z(stats['HSimFA']) - Z(stats['FoilFA'])
		stats['D_LSim_Foil'] = Z(stats['LSimFA']) - Z(stats['FoilFA'])
		stats['D_AllLure_Foil'] = Z(stats['AllLureFA']) - Z(stats['FoilFA'])

		stats['B_HSim']    = math.exp((Z(stats['HSimFA'])**2 - Z(stats['TargHR'])**2) / 2)
		stats['B_LSim']    = math.exp((Z(stats['LSimFA'])**2 - Z(stats['TargHR'])**2) / 2)
		stats['B_Foil']    = math.exp((Z(stats['FoilFA'])**2 - Z(stats['TargHR'])**2) / 2)
		stats['B_AllLure'] = math.exp((Z(stats['AllLureFA'])**2 - Z(stats['TargHR'])**2) / 2)

		stats['C_HSim'] = -(Z(stats['TargHR']) + Z(stats['HSimFA'])) / 2
		stats['C_LSim'] = -(Z(stats['TargHR']) + Z(stats['LSimFA'])) / 2
		stats['C_Foil'] = -(Z(stats['TargHR']) + Z(stats['FoilFA'])) / 2
		stats['C_AllLure'] = -(Z(stats['TargHR']) + Z(stats['AllLureFA'])) / 2

		return stats

	def CheckErrors(self):
		'''
		Checks the logfile and makes sure it's not empty
		'''
		if self.unreadable:
			return
		
		if len(self.trials) != self.numTrials:
			logger.warning("Subject {} does not have {} trials! Skipping".format(self.subID, self.numTrials))
			self.unreadable = True
			return
		
		numRespStudy = len([trial for trial in self.trials if trial['StudyResp'] != ''])
		numRespTest = len([trial for trial in self.trials if trial['TestResp'] != ''])
		
		'''
		if numRespStudy < .4*self.numTrials:
			logger.warning("Subject {} does not have enough Study Responses: {}".format(self.subID, numRespStudy))
			self.unreadable = True
			return
		'''
		if numRespTest < .4*self.numTrials:
			logger.warning("Subject {} does not have enough Test Responses: {}".format(self.subID, numRespTest))
			self.unreadable = True
			return
		

	def UpdateResponses(self):
		'''
		Update the responses to be more readable
		Responses can be:
			['v','n']
			or
			['f','j']
			or
			['6','9']
		'''
		if self.unreadable:
			return
		studyResponseMap = {'v': 'Indoor','n': 'Outdoor',
					 'f': 'Indoor', 'j': 'Outdoor',
					 '6': 'Indoor', '9': 'Outdoor',
					 '':'', 'space':'', 'p': ''}
		testResponseMap= {'v': 'Old','n': 'New',
					 'f': 'Old', 'j': 'New',
					 '6': 'Old', '9': 'New',
					 '':'', 'space':'', 'p': ''}
		for trial in self.trials:
			trial['StudyResp'] = studyResponseMap[trial['StudyResp']]
			trial['TestResp'] = testResponseMap[trial['TestResp']]
			if trial['TestResp'] == '':
				trial['DiffResp'] = np.nan
			else:
				trial['DiffResp'] = trial['TestRT'] if trial['TestResp'] == 'New' else -trial['TestRT']

	def UpdateTrialTypes(self):
		'''
		Update the trial types to more easily readable verions
		sF -> Foil
		sR -> Targ
		1  -> HSim
		2  -> LSim
		'''
		if self.unreadable:
			return
		typeMap = {'sF':'Foil', 'sR': 'Targ', '1': 'HSim', '2': 'LSim'}
		for trial in self.trials:
			trial['TrialType'] = typeMap[trial['TrialType']]
			if trial['TrialType'] == 'Targ':
				trial['CorrResp'] = 'Old'
			else:
				trial['CorrResp'] = 'New'

	def GetStatsDataFrame(self):
		'''
		Return a dataframe format for the stats
		
		'''
		cols = ['HSimCR','LSimCR','FoilCR',
		  'HSimFA','LSimFA','FoilFA','TargHR',
		  'LDI_HSim', 'LDI_LSim','LDI_Foil','D_HSim','D_LSim','D_Foil',
		  'AllLureCR', 'AllLureFA', 'LDI_AllLure', 'D_AllLure',
			'B_HSim', 'B_LSim', 'B_Foil', 'B_AllLure',
			'C_HSim', 'C_LSim', 'C_Foil', 'C_AllLure']
		header_cols = ['MDTO_' + col for col in cols]

		stats = self.GetStats()
		df = pd.DataFrame([[self.subID] + [stats[key] for key in cols]], columns = ['subID'] + header_cols) 
		return df

	def _readTrials(self):
		'''
		Read the trials from the logfile. We want two lists,
		one for the study trials, and one for the retrieval trials
		
		Return:
			studyList: list of raw study trials
			testList: list of raw test trials
		'''						
					
		trials = defaultdict(dict)
		with open(self.logFileLoc, 'r') as f:
			# Keep track of the current phase as we read down the log
			currentPhase = None
			for line in f:
				if 'Begin Study' in line:
					currentPhase = 'Study'
				elif 'Begin Test' in line:
					currentPhase = 'Test'
				elif currentPhase == 'Study':
					line = line.split()
					# if its a newline, or the first column is not a number, continue
					if len(line) == 0 or not line[0].isdigit():
						continue
					# study lines should be len of 4 (no response) or 5 (response)
					if len(line) == 4 or len(line) == 5:
						# add trial to study
						studyImg = line[1].split("_")[0]

						trials[studyImg]['StudyTrialNum'] = line[0]
						trials[studyImg]['StudyImg'] = line[1]
						trials[studyImg]['TrialType'] = line[2]
						# no response
						if len(line) == 4:
							trials[studyImg]['StudyResp'] = '' 
							trials[studyImg]['StudyRT'] = '' 
						else:
							trials[studyImg]['StudyResp'] = line[3]
							trials[studyImg]['StudyRT'] = line[4]
				elif currentPhase == 'Test':
					line = line.split()
					# if its a newline or the first col is not a digit
					if len(line) == 0 or not line[0].isdigit():
						continue
					if len(line) == 5 or len(line) == 6:
						# add trial to test
						studyImg = line[1].split("_")[0].replace('b','a')
						trials[studyImg]['TestTrialNum'] = line[0]
						trials[studyImg]['TestImg'] = line[1]	
						# This means it's a Foil condition
						if 'TrialType' not in trials[studyImg].keys():
							trials[studyImg]['StudyTrialNum'] = ''			
							trials[studyImg]['StudyImg'] = ''			
							trials[studyImg]['StudyResp'] = ''			
							trials[studyImg]['StudyRT'] = ''			
							trials[studyImg]['TrialType'] = line[2]

						if trials[studyImg]['TrialType'] != line[2]:
							raise Exception("Study Trial type does not match test trial type: file: {}\n {},{}".format(self.logFileLoc, trials[studyImg]['TrialType'], line))

						# no response
						if len(line) == 5:
							trials[studyImg]['TestResp'] = '' 
							trials[studyImg]['TestRT'] = '' 
						else:
							trials[studyImg]['TestResp'] = line[4]
							trials[studyImg]['TestRT'] = float(line[5])
		return trials.values()

	def GetDiffusionTrials(self):
		'''
		We want to return the following:
		1. subject ID
		2. list of trial new/old for all trials
		3. list of conditions 
		'''
		subj_id = self.subID
		trial_corr = [trial['DiffResp'] for trial in self.trials]
		condMap = {'Foil': 4, 'LSim': 3, 'HSim': 2, 'Targ': 1}
		trial_conds = [condMap[trial['TrialType']] for trial in self.trials]
		return subj_id, trial_corr, trial_conds
	

if __name__ == '__main__':
	import glob

	parser = argparse.ArgumentParser(description='MDT-Object log reader. Analyze your MDT-Object logs')
	parser.add_argument('--id', help='Group ID', required=True)
	parser.add_argument('--input', help='Input directory', required=True)
	parser.add_argument('--output', help='Output directory', required=True)
	#parser.add_argument('--diffusion', default='True')
	cli_args = parser.parse_args()
	'''
	if cli_args.diffusion == 'True':
		dfs = []
		all_subj = []
		all_group = []
		all_trial_corr = []
		all_trial_conds = []
		for group in ['old', 'young']:
			for logfile in glob.glob('{}_logs/*_MDTO_log.txt'.format(group)):
				reader = MDTO_Log_Reader(logfile)
				dfs.append(reader.GetStats())
				subj_id, trial_corr, trial_conds = reader.GetDiffusionTrials()
				all_subj.append(subj_id)
				all_group.append(1 if group == 'old' else 2)
				all_trial_corr.append(trial_corr)
				all_trial_conds.append(trial_conds)

		all_subj = np.array(all_subj).astype(float)
		all_group = np.array(all_group)
		all_trial_corr = np.array(all_trial_corr)
		all_trial_conds = np.array(all_trial_conds)
		
		df = pd.DataFrame(dfs)
		print(df)
		print(df['subID'])
		for label in ['LDI_HSim', 'LDI_LSim']:
			print("{} Old Mean: {}".format(label,df[df['subID'] > 199][label].mean()))
			print("{} Yng Mean: {}".format(label,df[df['subID'] < 199][label].mean()))

		data = {'subjList': all_subj,
				'groupList': all_group,
				'rt': all_trial_corr,
				'subList': all_trial_conds,
				}
		hdf5storage.write(data, '.',  '_'.join(['old','young']) + '.mat', matlab_compatible=True)
	else:
	'''
	group = MDTO_Group(cli_args)
	group.save()
