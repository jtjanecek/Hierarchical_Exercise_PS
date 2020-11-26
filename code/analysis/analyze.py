import scipy.stats as ss
from scipy.stats import norm
import glob
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 150
import pandas as pd
import seaborn as sns
from collections import defaultdict
import os
sns.set()

import logging
logging.basicConfig(level=logging.INFO,
           format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
           datefmt='%m-%d-%y %H:%M')

FIGURE_DIR = os.path.join("..", "..", 'results')
STORAGE_DIR = os.path.join("..","..","storage")


class ChainAnalyzer():
	def __init__(self, model: str):

		self.model_name = model

		self._model_stats = {'model_name': self.model_name}
		logging.info("Initializing analyzer for: " + self.model_name)
		self._initialize_chains()
		logging.info("Done.")

	def _initialize_chains(self):
		self._chains = []
		for hdf_chain in glob.glob(os.path.join(STORAGE_DIR, self.model_name, "*.hdf5")):
			self._chains.append(h5py.File(hdf_chain, 'r'))
		logging.info("Initialized {} chains.".format(len(self._chains)))
		logging.info(self._chains[0].keys())
	
	def close(self):
		for chain in self._chains:
			chain.close()

	def analyze(self):
		self._plot_differences()
		self._plot_dic()
		self._plot_drifts()

	def _plot_drifts(self):
		deltaMu = self._get_vals("deltaMu")	
		print(deltaMu.shape)

	def _plot_histogram(self, data, title, figpath):
		plt.figure()
		plt.hist(data, bins=50, density=True)
		plt.title(title)
		plt.savefig(figpath)
		plt.close()

	def _plot_single_difference(self, vals, prior_mean, prior_sd, title, fig_name):
		''' Plot the difference distributions along with priors
		'''
		plt.figure()
	
		# Plot observered
		plt.hist(vals, bins=50, density=True, label='Posterior')
		#model_stats[model_name]['{}_>0'.format(label)] = sum(vals>0) / len(vals)
		#model_stats[model_name]['{}_<0'.format(label)] = sum(vals<0) / len(vals)

		cur_xlim = plt.gca().get_xlim()
		min_x = min(cur_xlim[0], prior_mean-prior_sd*3)
		max_x = max(cur_xlim[1], prior_mean+prior_sd*3)

		# Plot priors
		x = np.linspace(prior_mean-prior_sd*4,prior_mean+prior_sd*4, 100)
		y_pdf = ss.norm.pdf(x, prior_mean, prior_sd) # the normal pdf
		plt.plot(x, y_pdf, label='Prior')

		# Fit normal distribution
		mu, sd = norm.fit(vals)	
		x = np.linspace(mu-sd*4,mu+sd*4, 100)
		y_pdf = ss.norm.pdf(x, mu, sd) # the normal pdf
		plt.plot(x, y_pdf, label='Posterior Fit')

		self._model_stats["{}_{}".format(fig_name.split(".")[0],'BF_zero')] = ss.norm.pdf([0], prior_mean, prior_sd)[0] / ss.norm.pdf([0], mu, sd)[0]

		plt.xlim([min_x,max_x])
		plt.title(title)
		plt.legend()
		plt.savefig(os.path.join(FIGURE_DIR, 'differences', "{}_{}.png".format(self.model_name, fig_name)))
		plt.close()

	def _get_vals(self, var):
		'''
		Get values from hdf5 files and collapse across chain files
		'''
		vals = None
		for chain in self._chains:
			#logging.info(type(vals))
			if type(vals) != np.ndarray:
				vals = np.array(chain.get(var))	
				#logging.info("None: {}".format(vals.shape))
			else:
				vals = np.concatenate([vals,np.array(chain.get(var))],-1)
				#logging.info("Concat: {}".format(vals.shape))
		return vals

	def _plot_differences(self):

		deltaExercise = self._get_vals('deltaDiffExercise')
		deltaExerciseCarryover = self._get_vals('deltaDiffExerciseCarryover')
		deltaControl = self._get_vals('deltaDiffControl')

		for cond_num, cond_label in enumerate(['Targ', 'HSim', 'LSim', 'Foil']):
			# Exercise diff
			data = deltaExercise[cond_num,:]
			print(data.shape)
			title = 'Delta {} Difference Post Exercise'.format(cond_label)
			label = "deltaDiffExercise_{}".format(cond_label)
			self._plot_single_difference(data, 0, .5, title, label)

			# Carryover Diff
			data = deltaExerciseCarryover[cond_num,:]
			print(data.shape)
			title = 'Delta {} Difference Exercise Carryover'.format(cond_label)
			label = "deltaDiffExerciseCarryover_{}".format(cond_label)
			self._plot_single_difference(data, 0, .5, title, label)

			# Control Diff
			data = deltaControl[cond_num,:]
			print(data.shape)
			title = 'Delta {} Difference Control'.format(cond_label)
			label = "deltaDiffControl_{}".format(cond_label)
			self._plot_single_difference(data, 0, .5, title, label)

		for label, prior_vals in [['alpha', [0, .3]], ['beta',[0, .2]], ['tau',[0, .2]]]:
			prior_mu = prior_vals[0]
			prior_sd = prior_vals[1]

			# Exercise diff
			data = self._get_vals("{}DiffExercise".format(label))
			title = '{} Difference Post Exercise'.format(label.capitalize())
			fig_label = "{}DiffExercise".format(label)
			self._plot_single_difference(data, prior_mu, prior_sd, title, fig_label)
	
			# Carryover diff
			data = self._get_vals("{}DiffExerciseCarryover".format(label))
			title = '{} Difference Exercise Carryover'.format(label.capitalize())
			fig_label = "{}DiffExerciseCarryover".format(label)
			self._plot_single_difference(data, prior_mu, prior_sd, title, fig_label)

			# Control diff
			data = self._get_vals("{}DiffControl".format(label))
			title = '{} Difference Control'.format(label.capitalize())
			fig_label = "{}DiffControl".format(label)
			self._plot_single_difference(data, prior_mu, prior_sd, title, fig_label)

	def _plot_dic(self):
		data = self._get_vals("deviance").flatten()
		print("Data:")
		print(data.shape)
		print(sum(np.isnan(data)))
		self._model_stats['dic'] = data.mean()
		self._plot_histogram(data, "DIC", os.path.join(FIGURE_DIR, "dic", "{}.png".format(self.model_name)))

	def get_stats(self):
		return self._model_stats

	def _calculate_posterior_predictives(self, chains):
		logging.info("Filling ypred....")
		# Load in all the y_pred simulated values (lots of values)
		chains = f.get('chains')
		y_pred = np.zeros((n_subj,n_trials,n_values))
		for s in range(n_subj):
			for tr in range(n_trials):
				exec('y_pred[s,tr,:] = np.array(chains.get("ypred_' + str(s+1) + '_' + str(tr+1) + '")).flatten()')
		logging.info("Loading y")
		# Load in original y
		y = np.array(f.get("data").get("y")).T

if __name__ == '__main__':

	all_stats = []
	for model in ['model_07_C1']:
		analyzer = ChainAnalyzer(model)
		analyzer.analyze()
		print(analyzer.get_stats())
		analyzer.close()

	'''		
	all_stats.append(analyzer.get_stats())

	# Now let's save the stats to a CSV
	logging.info("Saving results to CSV...")
	df = pd.DataFrame(all_stats)
	logging.info(df.head(n=50))
	df.to_csv(os.path.join(FIGURE_DIR,"csvs","results.csv"))
	'''
