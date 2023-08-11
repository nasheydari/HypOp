
"""
This software is Copyright © 2XXX The Regents of the University of California. All Rights Reserved.
Permission to copy, modify, and distribute this software and its documentation for educational, research and
non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the
above copyright notice, this paragraph and the following three paragraphs appear in all copies. Permission
to make commercial use of this software may be obtained by contacting:

Office of Innovation and Commercialization
9500 Gilman Drive, Mail Code 0910
University of California
La Jolla, CA 92093-0910
(858) 534-5815
innovation@ucsd.edu

This software program and documentation are copyrighted by The Regents of the University of California.
The software program and documentation are supplied “as is”, without any accompanying services from
The Regents. The Regents does not warrant that the operation of the program will be uninterrupted or error-
free. The end-user understands that the program was developed for research purposes and is advised not to
rely exclusively on the program for any reason.

IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE
PROVIDED HEREUNDER IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA
HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
ENHANCEMENTS, OR MODIFICATIONS.
"""


import os
import time
import copy
import numpy as np
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

class AdaNS_sampler(object):
	def __init__(self, boundaries, minimum_num_good_samples):
		# minimum number of good samples (b) used to find the value of \alpha for each iteration
		self.minimum_num_good_samples = minimum_num_good_samples
		
		# shape of boundaries: <d, 2>. Specifies the minimum and maximum allowed value of the hyperparameter per dimension.
		self.boundaries = boundaries
		self.dimensions = len(boundaries)
		
		self.all_samples = np.zeros((0, self.dimensions))
		self.all_scores = np.zeros(0)
		
		self.good_samples = np.zeros(0)
		
		# maximum score through all iterations seen so far
		self.max_score = 0
		self.alpha_t = 0

	
	def sample_uniform(self, num_samples=1):
		'''
		function to sample unifromly from all the search-space
			- num_samples: number of samples to take
		'''
		if num_samples>0:
			sample_vectors = np.random.uniform(self.boundaries[:,0], self.boundaries[:,1], size=(num_samples, self.dimensions))
			sample_vectors = np.unique(sample_vectors, axis=0)
			while len(sample_vectors) < num_samples:
				count = num_samples - len(sample_vectors)
				sample_vectors = np.concatenate((sample_vectors, np.random.uniform(self.boundaries[:,0], self.boundaries[:,1], size=(count, self.dimensions))))
				sample_vectors = np.unique(sample_vectors, axis=0)
		else:
			sample_vectors = np.zeros((0, self.dimensions))

		return sample_vectors
	

	def update_good_samples(self, alpha_t=None):
		'''
		function to update the list of good samples after evaluating a new batch of samples
			- alpha_max: \alpha_max parameter
		'''
		self.max_score = np.max(self.all_scores)
		
		if alpha_t is not None:
			score_thr = alpha_t * self.max_score
		else:
			score_thr = self.alpha_t * self.max_score
		
		self.good_samples = self.all_scores>=score_thr
	

	def configure_alpha(self, alpha_max=1.0, verbose=True):
		'''
		function to determine \alpha based on current good samples
			- alpha_max: \alpha_max
		'''
		if np.sum(self.good_samples)<self.minimum_num_good_samples:
			self.max_score = np.max(self.all_scores)
			alpha_t = alpha_max
			

			if self.max_score==0:
				sorted_args = np.argsort(self.all_scores)[::-1]
				indices = sorted_args[:self.minimum_num_good_samples]
				self.good_samples[indices] = True
				assert np.sum(self.good_samples)==self.minimum_num_good_samples, (np.sum(self.good_samples), self.minimum_num_good_samples)

			else:
				itr = 0
				while np.sum(self.good_samples)<self.minimum_num_good_samples and itr<1000:
					if self.max_score < 0:
						alpha_t = alpha_t + 0.05
					else:
						alpha_t = alpha_t - 0.05

					self.update_good_samples(alpha_t)
					itr += 1

				if np.sum(self.good_samples)<self.minimum_num_good_samples:
					sorted_args = np.argsort(self.all_scores)[::-1]
					alpha_t = self.all_scores[sorted_args[self.minimum_num_good_samples-1]] / self.all_scores[sorted_args[0]]
					self.update_good_samples(alpha_t)
				
			if verbose:
				print('changing alpha_t to %0.2f' % (alpha_t))
			self.alpha_t = alpha_t

		return self.alpha_t
	

	def update(self, samples, scores, alpha_max, **kwargs):	
		'''
		function to add newly evaluated samples to the history
			- samples: new samples
			- scores: evaluation score of new samples
			- alpha_max: current \alpha_max
		'''   
		self.all_samples = np.concatenate((self.all_samples, samples), axis=0)
		orig_count = self.all_samples.shape[0]
		self.all_samples, indices = np.unique(self.all_samples, axis=0, return_index=True)
		if self.all_samples.shape[0] < orig_count:
			print(f'==== Removing {orig_count-self.all_samples.shape[0]} duplicate samples')

		self.all_scores = np.concatenate((self.all_scores, scores), axis=0)[indices]
		assert len(self.all_samples)==len(self.all_scores)

		self.update_good_samples(alpha_max)

		return indices


	def run_sampling(self, evaluator, num_samples, n_iter, minimize=False, alpha_max=1.0, early_stopping=np.Infinity, 
		save_path='./sampling', n_parallel=1, plot_contour=False, executor=mp.Pool, verbose=True):
		'''
		Function to maximize given black-box function and save results to ./sampling/
			- evaluator : the objective function to be minimized
			- num_samples: number of samples to take at each iteration
			- n_iter: total number of sampling rounds
			- minimize: if set to True, the objective function will be minimized, otherwise maximized
			- alpha_max: \alpha_max parameter
			- early_stopping: the sampling loop will terminate after this many iterations without improvmenet
			- save_path: path to save the sampling history and other artifcats
			- n_parallel: number of parallel evaluations
			- plot_contour: whether to plot contours of objective functions and the samples
			- executor: function to handle parallel evaluations
		returns: optimal hyperparameters
		'''
		coeff = -1 if minimize else 1

		# set up logging directory
		os.makedirs(save_path, exist_ok=True)

		# set up contour plotting
		contour = None
		if plot_contour:
			if self.dimensions==2:
				path_to_contour = os.path.join(save_path, 'contour')
				os.makedirs(path_to_contour, exist_ok=True)
				contour_file = os.path.join(path_to_contour, 'contour_data.pkl')
				if not os.path.exists(contour_file):
					x = np.linspace(self.boundaries[0,0], self.boundaries[0,1], num=1000)
					y = np.linspace(self.boundaries[1,0], self.boundaries[1,1], num=1000)

					data = np.zeros((len(x), len(y)))
					for i in range(len(x)):
						for j in range(len(y)):
							data[i, j] = evaluator([x[i], y[j]])
					contour = (x, y, data)
					with open(contour_file, 'wb') as f:
						pickle.dump(contour, f)
				else:
					with open(contour_file, 'rb') as f:
						contour = pickle.load(f)
						x, y, data = contour

				max_val = np.max(data)
				max_ind = np.unravel_index(np.argmax(data, axis=None), data.shape)
				print('maximum is %.2f located at (%.2f, %.2f)'%(max_val, x[int(max_ind[0])], y[int(max_ind[1])]))
			else:
				print('=> Contour plotting not possible for %d dimensions.'%self.dimensions)

		# adjusting the per-iteration sampling budget to the parallelism level
		if num_samples % n_parallel != 0:
			num_samples = num_samples - (num_samples % n_parallel) + n_parallel
			print('=> Sampling budget was adjusted to be ' + str(num_samples))
			self.minimum_num_good_samples = num_samples

		
		# apply the sampling algorithm
		best_samples = []
		best_scores = []
		alpha_vals = []
		num_not_improve = 0
		for iteration in range(n_iter):
			if iteration==0:
				samples = self.sample_uniform(num_samples)
				origins = ['U']*len(samples)
				prev_max_score = self.max_score
			else:
				max_score_improv = self.max_score - prev_max_score
				prev_max_score = self.max_score
				samples, origins = self.sample(num_samples, verbose=verbose)

				# if the percentage improvement in the maximum score is smaller than 0.1%, activate early stopping
				if (max_score_improv/prev_max_score) < 0.001:
					num_not_improve += 1 
				else:
					num_not_improve = 0

			if num_not_improve > early_stopping:
				print('=> Activating early stopping')
				break

			# evaluate current batch of samples
			scores = np.zeros(len(samples))
			n_batches = len(samples)//n_parallel if len(samples)%n_parallel==0 else (len(samples)//n_parallel)+1
			with tqdm(total=n_batches) as pbar:
				for i in range(n_batches):
					if n_parallel > 1:
						batch_samples = samples[i*n_parallel:(i+1)*n_parallel]
						with executor() as e:
							scores[i*n_parallel:(i+1)*n_parallel] = list(e.map(evaluator, batch_samples))
					else:
						scores[i] = evaluator(samples[i])
					scores[i*n_parallel:(i+1)*n_parallel] *= coeff
					
					pbar.update(1)
					pbar.set_description('batch %s/%s (samples %s..%s/%s)'%(i+1, num_samples//n_parallel, i*n_parallel, \
													(i+1)*n_parallel, num_samples))			  
			self.update(samples=samples, scores=scores, origins=origins, alpha_max=alpha_max)

			# modify \alpha if necessary, to make sure there are enough "good" samples
			alpha = self.configure_alpha(alpha_max, verbose=verbose)
			alpha_vals.append(alpha)

			# optionally visualize the current samples on the search-space
			if contour is not None:
				plt.figure()
				plt.contourf(contour[0], contour[1], contour[-1])
				plt.colorbar()
				plt.scatter(samples[:,0], samples[:,1], c='k', s=30)
				plt.xlim(self.boundaries[0,:])
				plt.ylim(self.boundaries[1,:])
				plt.savefig(os.path.join(path_to_contour, 'score_contour_iter%d.png'%iteration))
				plt.close()

			# book-keeping
			best_scores.append(np.max(self.all_scores))
			id_best = np.argmax(self.all_scores)
			best_samples.append(self.all_samples[id_best])
			
			if verbose:
				print('=> iter: %d, average score: %.3f, best score: %0.3f' %(iteration, np.mean(scores)*coeff, best_scores[-1]*coeff))

		info = {'best_samples': np.asarray(best_samples),
				'best_scores': np.asarray(best_scores),
				'alpha_vals': alpha_vals,
				'all_samples': self.all_samples,
				'all_scores': self.all_scores,
				'good_samples':self.good_samples}

		path_to_info = os.path.join(save_path, 'history_info.pkl')
		with open(path_to_info, 'wb') as f:
			pickle.dump(info, f)

		id_best_overall = np.argmax(best_scores)
		best_sample_overall = best_samples[id_best_overall]

		if contour is not None:
				plt.figure()
				plt.contourf(contour[0], contour[1], contour[-1])
				plt.colorbar()
				plt.scatter(best_sample_overall[0], best_sample_overall[1], c='r', marker='*', s=100)
				plt.xlim(self.boundaries[0,:])
				plt.ylim(self.boundaries[1,:])
				plt.savefig(os.path.join(path_to_contour, 'score_contour_final.png'))
				plt.close()

		return best_sample_overall, best_scores[id_best_overall]*coeff


class Genetic_sampler(AdaNS_sampler):
	def __init__(self, boundaries, minimum_num_good_samples, p_cross=0.8, p_swap=0.2, p_mutate=0.8, p_tweak=0.05, mutate_scale=0.2):
		'''
			- p_cross: probability of crossing the parent vectors
			- p_swap: probability of swapping each hyperparameter in the parent vectors 
			- p_mutate: probability of mutating the vector
			- p_tweak: probability of mutating each hyperparameter
			- mutate_scale: mutation noise is sampled from N~(0, mutate_scale) 
		'''

		super(Genetic_sampler, self).__init__(boundaries, minimum_num_good_samples)

		self.p_cross = p_cross
		self.p_swap = p_swap
		self.p_mutate = p_mutate
		self.p_tweak = p_tweak
		self.mutate_scale = mutate_scale
	

	def set_params(self, p_cross=None, p_swap=None, p_mutate=None, p_tweak=None, mutate_scale=None):
		if p_cross is not None:
			self.p_cross = p_cross

		if p_swap is not None:
			self.p_swap = p_swap

		if p_mutate is not None:
			self.p_mutate = p_mutate

		if p_tweak is not None:
			self.p_tweak = p_tweak

		if mutate_scale is not None:
			self.mutate_scale = mutate_scale


	def mutate(self, individual):
		'''
		function to mutate individuals in the genetic algorithm
			- individual: input hyperparameter vector to be mutated
		'''
		p = np.random.rand()
		if p < self.p_mutate:
			#--------------- do mutation
			for i in range(len(individual)):
				p = np.random.rand()
				if p <= self.p_tweak:
					#----------- mutate this gene
					noise = np.random.normal(loc=0.0, scale=self.mutate_scale)
					individual[i] = individual[i] + noise

		individual = np.clip(individual, self.boundaries[:,0], self.boundaries[:,1])
		return individual
	

	def crossover(self, father, mother):
		'''
		function to crossover two individuals in the genetic algorithm
			- father, mother: input hyperparameter vectors to be crossovered
		'''
		p = np.random.rand()
		if p <= self.p_cross:	
			#------------------- do crossover
			for i in range(len(father)):
				p = np.random.rand()
				if p < self.p_swap:
					#-------------- swap element of mother & father
					temp = father[i]
					father[i] = mother[i]
					mother[i] = temp
		return father, mother


	def sample(self, num_samples, **kwargs):
		'''
		function to sample from the search-space
			- num_samples: number of samples to take
		'''
		if num_samples==0:
			return np.zeros((0, self.dimensions)).astype(np.int32)

		num_samples = num_samples + np.mod(num_samples, 2)

		if num_samples >= int(np.sum(self.good_samples)+0.001):
			num_samples = int(np.sum(self.good_samples)+0.001)
			num_samples = num_samples - np.mod(num_samples, 2)
			genetic_samples = self.all_samples[self.good_samples][:num_samples]
		else:
			inds = np.where(self.good_samples)[0]
			probs = (self.all_scores[self.good_samples] - np.min(self.all_scores[self.good_samples]))
			avg_good_scores = np.mean(probs)
			probs = probs + avg_good_scores
			if np.sum(probs)==0:
				probs = np.ones_like(probs)
			choices = np.random.choice(inds, size=num_samples, replace=False, p=probs/np.sum(probs))
			genetic_samples = np.asarray([self.all_samples[c] for c in choices])
			
		for index in range(num_samples//2):
		# crossover
			father = genetic_samples[2*index]
			mother = genetic_samples[2*index+1]
			kid1, kid2 = self.crossover(father, mother)

			#----------------- mutation
			kid1 = self.mutate(kid1)
			kid2 = self.mutate(kid2)
			genetic_samples[2*index] = kid1
			genetic_samples[2*index+1] = kid2

		return genetic_samples, None


class Gaussian_sampler(AdaNS_sampler):
	def __init__(self, boundaries, minimum_num_good_samples, u_random_portion=0.2, local_portion=0.4, cross_portion=0.4, pair_selection_method='random'):
		'''
			- u_random_portion: ratio of samples taken uniformly from the entire space
			- local_portion: ratio of samples taken from gaussian distributions using the "local" method
			- cross_portion: ratio of samples taken from gaussian distributions using the "cross" method
			
				(u_random + local_portion + cross_portion) = 1
			
			- pair_selection_method: how to select pairs for cross samples. Options: ['random','top_scores','top_and_nearest','top_and_furthest','top_and_random']
		'''

		super(Gaussian_sampler, self).__init__(boundaries, minimum_num_good_samples)

		# for each sample, specifies how it was created: 'U':uniformly 'L':gaussian local, 'C':gaussian cross
		self.origins = []

		self.u_random_portion = u_random_portion
		self.local_portion = local_portion
		self.cross_portion = cross_portion
		assert (u_random_portion + local_portion + cross_portion) == 1., 'sum of sampling portions must be 1'

		self.pair_selection_method = pair_selection_method
		assert pair_selection_method in ['random','top_scores','top_and_nearest','top_and_furthest','top_and_random'], \
						"pair selection should be one of ['random','top_scores','top_and_nearest','top_and_furthest','top_and_random']"


	def set_params(self, u_random_portion=None, local_portion=None, cross_portion=None, pair_selection_method=None):
		if u_random_portion is not None:
			self.u_random_portion = u_random_portion

		if local_portion is not None:
			self.local_portion = local_portion

		if cross_portion is not None:
			self.cross_portion = cross_portion

		if pair_selection_method is not None:
			self.pair_selection_method = pair_selection_method

		assert (self.u_random_portion + self.local_portion + self.cross_portion) == 1., 'sum of sampling portions must be 1'
		assert pair_selection_method in ['random','top_scores','top_and_nearest','top_and_furthest','top_and_random'], \
								"pair selection should be one of ['random','top_scores','top_and_nearest','top_and_furthest','top_and_random']"


	def sample(self, num_samples, verbose=True, **kwargs):
		'''
		function to sample from the search-space
			- num_samples: number of samples to take
		'''
		if num_samples==0:
			return np.zeros((0, self.dimensions)).astype(np.int32), []

		data = self.all_samples[self.good_samples]
		assert len(np.unique(data, axis=0))==data.shape[0], 'duplicate samples found'
		scores = self.all_scores[self.good_samples] - np.min(self.all_scores[self.good_samples])
		avg_good_scores = np.mean(scores)
		scores = scores + avg_good_scores
		assert np.sum(scores>=0)==len(scores)

		# "Local" samples created with gaussians
		local_sampling = int(num_samples*self.local_portion+0.001)
		
		max_all_dims = np.max(data, axis=0)
		min_all_dims = np.min(data, axis=0)
		
		gaussian_means = data
## change this line
		gaussian_covs = np.asarray([((max_all_dims-min_all_dims)/2.0)**2 for _ in range(len(data))])
		gaussian_mix = GaussianMixture(n_components=data.shape[0], covariance_type='diag',
									  weights_init=np.ones(data.shape[0])/data.shape[0], means_init=data)
		
		try:
			gaussian_mix.fit(X=data)
			gaussian_mix.means_ = gaussian_means
			gaussian_mix.covariances_ = gaussian_covs
			if np.sum(scores)==0:
				print('====== sum of scores was zero')
				gaussian_mix.weights_ = [1./len(scores)] * len(scores)
			else:
				gaussian_mix.weights_ = scores/np.sum(scores)

			if local_sampling>0:
				local_samples  = gaussian_mix.sample(n_samples=local_sampling)[0]
				local_samples  = np.clip(local_samples, self.boundaries[:,0], self.boundaries[:,1])
			else:
				local_samples = np.zeros((0, self.dimensions))
		except:
			local_samples  = self.sample_uniform(num_samples=local_sampling)

		# "Cross" samples created with gaussians	
		cross_sampling = int(num_samples*self.cross_portion+0.001)
		cross_sampling = cross_sampling + np.mod(cross_sampling, 2)
	
		cross_samples = np.zeros((0, self.dimensions))
		if cross_sampling>0:
			pairs = self.get_pairs(num_pairs=cross_sampling)
			for pair in pairs:
				father = self.all_samples[pair[0]]
				mother = self.all_samples[pair[1]]
				gauss_mean = (father + mother)/2.0
				gauss_cov = (np.absolute(father-mother)/2.0)**2
				gauss_cov = np.diag(gauss_cov)
				sample = np.random.multivariate_normal(gauss_mean, gauss_cov)
				sample = np.clip(sample, self.boundaries[:,0], self.boundaries[:,1])
				sample = np.expand_dims(sample, axis=0)
				cross_samples = np.append(cross_samples, sample, axis=0)				

		# "Uniform" samples chosen uniformly random
		random_sampling = int(num_samples*self.u_random_portion+0.001)  
		random_samples = self.sample_uniform(num_samples=random_sampling)
			   
		if verbose:
			print('sampled %d uniformly, %d with local gaussians, %d with cross gaussians'%(len(random_samples), len(local_samples), len(cross_samples)))
		
		origins_random = ['U'] * len(random_samples)
		origins_local = ['L'] * len(local_samples)
		origins_cross = ['C'] * len(cross_samples)
		origins = origins_random + origins_local + origins_cross
				
		sample_vectors = random_samples
		if local_sampling>0:
			sample_vectors = np.concatenate((sample_vectors, local_samples))
			
		if cross_sampling>0:
			sample_vectors = np.concatenate((sample_vectors, cross_samples))

		sample_vectors, indices = np.unique(sample_vectors, axis=0, return_index=True)
		origins = [origins[i] for i in indices]
		while len(sample_vectors) < num_samples:
			count = num_samples - len(sample_vectors)
			print(f'adding {count} more random samples')
			sample_vectors = np.concatenate((sample_vectors, self.sample_uniform(num_samples=count)))
			origins += ['U'] * count
			sample_vectors, indices = np.unique(sample_vectors, axis=0, return_index=True)
			origins = [origins[i] for i in indices]
		
		return sample_vectors, origins


	def update(self, samples, scores, origins, alpha_max):	
		'''
		function to add newly evaluated samples to the history
			- samples: new samples
			- scores: evaluation score of new samples
			- origins: origin of new samples (zoom, genetic, gaussian-local, gaussian-cross, uniform-random)
			- alpha_max: current \alpha_max
		''' 
		indices = super(Gaussian_sampler, self).update(samples, scores, alpha_max)
		self.origins = np.concatenate((self.origins, origins), axis=0)[indices]
	

	def get_pairs(self, num_pairs):
		'''
		function to find pairs of vectors for Gaussian cross sampling
			- num_pairs: number of vector pairs to create
		'''
		pairs = []
		inds = np.where(self.good_samples)[0]
		if self.pair_selection_method == 'random':
			while(len(pairs)<num_pairs):
				choices = np.random.choice(inds, size=2, replace=False)
				pairs.append((choices[0], choices[1]))
		
		elif self.pair_selection_method == 'top_scores':
			scores = self.all_scores[self.good_samples]
			sum_score_mat = np.zeros((len(scores), len(scores)))
			for i, s1 in enumerate(scores[:-1]):
				for j in range(i+1, len(scores)):
					s2 = scores[j]
					sum_score_mat[i][j] = s1 + s2
			indices = np.argsort(sum_score_mat, axis=None)[::-1][:num_pairs]
			pair_inds = np.unravel_index(indices, sum_score_mat.shape)
			for p0, p1 in zip(pair_inds[0], pair_inds[1]):
				pairs.append((inds[p0], inds[p1]))
		
		elif self.pair_selection_method == 'top_and_nearest':
			scores = self.all_scores[self.good_samples]
			samples = self.all_samples[self.good_samples]
			sorted_sample_ids = np.argsort(scores)[::-1] 
			distance_mat = np.zeros((len(scores), len(scores)))
			for i, s1 in enumerate(scores[:-1]):
				for j in range(i, len(scores)):
					s2 = scores[j]
					distance_mat[i][j] = np.sum((samples[i]-samples[j])**2)
					distance_mat[j][i] = np.sum((samples[i]-samples[j])**2)
			for i in range(len(scores)):
				distance_mat[i,i] = np.Infinity
			pair_each_point = np.zeros(len(scores)).astype(np.int32)
			id0 = 0
			while(len(pairs)<num_pairs):
				candidates = distance_mat[sorted_sample_ids[id0]]
				closest = np.argsort(candidates)[pair_each_point[id0]]
				pairs.append((inds[sorted_sample_ids[id0]], inds[closest]))
				pair_each_point[id0] += 1
				id0 += 1
				id0 = np.mod(id0, len(scores))
		
		elif self.pair_selection_method == 'top_and_furthest':
			scores = self.all_scores[self.good_samples]
			samples = self.all_samples[self.good_samples]
			sorted_sample_ids = np.argsort(scores)[::-1] 
			distance_mat = np.zeros((len(scores),len(scores)))
			for i, s1 in enumerate(scores[:-1]):
				for j in range(i,len(scores)):
					s2 = scores[j]
					distance_mat[i][j] = np.sum((samples[i]-samples[j])**2)
					distance_mat[j][i] = np.sum((samples[i]-samples[j])**2)		
			for i in range(len(scores)):
				distance_mat[i,i] = 0
			pair_each_point = np.zeros(len(scores)).astype(np.int32)
			id0 = 0
			while(len(pairs)<num_pairs):
				candidates = distance_mat[sorted_sample_ids[id0]]
				farest = np.argsort(candidates)[::-1][pair_each_point[id0]]
				pairs.append((inds[sorted_sample_ids[id0]], inds[farest]))
				pair_each_point[id0] += 1
				id0 += 1
				id0 = np.mod(id0, len(scores))
		
		elif self.pair_selection_method == 'top_and_random':
			scores = self.all_scores[self.good_samples]
			samples = self.all_samples[self.good_samples]
			sorted_sample_ids = np.argsort(scores)[::-1] 
			id0 = 0
			while len(pairs)<num_pairs:
				id1 = id0
				while(id1==id0):
					id1 = np.random.randint(len(samples))
				pairs.append((inds[sorted_sample_ids[id0]], inds[sorted_sample_ids[id1]]))
				id0 += 1
				id0 = np.mod(id0, len(scores))
		
		return pairs


class Zoom_sampler(AdaNS_sampler):
	# zoom
	def __init__(self, boundaries, minimum_num_good_samples):
		super(Zoom_sampler, self).__init__(boundaries, minimum_num_good_samples)
		
		self.num_regions = 1
		self.region_ids = np.zeros(0)
		self.good_region_ids = np.zeros(0)
		self.per_region_num_goods = np.zeros(self.num_regions)

		# probability of sampling each region, initially uniform
		self.per_region_sampling_probs = np.ones(self.num_regions)/self.num_regions

		# "start" and "end" are vectors indicating per-region boundaries
		self.starts = np.zeros((self.num_regions, self.dimensions))
		self.ends = np.zeros((self.num_regions, self.dimensions))

		self.initialize_starts_ends()

	
	def initialize_starts_ends(self):
		'''
		function to initialize the regions in the search space
		'''
		self.starts = np.expand_dims(self.boundaries[:,0], axis=0)
		self.ends = np.expand_dims(self.boundaries[:,1], axis=0)


	def id_to_vector(self, id, start, end): 
		'''
		function to sample a vector of hyperparameters uniformly at random from a given region in the search space
			- id: region id to sample from
			- start: <d> starting coordinate of the region (d is the search space dimensionality)
			- end: <d> ending coordinate of the region
		'''
		assert id < self.num_regions
		v = np.zeros(self.dimensions)
		for i in range(len(v)):
			v[i] = np.random.uniform(start[i], end[i])
		return v
	

	def vector_to_id(self, v, starts, ends):
		'''
		function to convert a vector of hyperparameters to its corresponding region id based on where in the search space it is located
			- v: input vector of hyperaprameters
			- starts: <N, d> starting coordinates of search-space regions (N is the number of regions and d is the search space dimensionality)
			- ends: <N, d> ending coordinates of search-space regions
		'''
		for i in range(len(starts)):
			start = starts[i]
			end = ends[i]
			n = 0
			for ii, vv in enumerate(v):
				if vv>=start[ii] and vv<=end[ii]:
					n = n+1
			if n==self.dimensions:
				return i			


	def sample_from_all_regions(self, per_regions_samples):
		'''
		function to sample unifromly from all the search-space
			- per_regions_samples: number of samples to take from each region
		'''
		ids = np.asarray([np.ones(per_regions_samples)*i for i in range(self.num_regions)]).astype(np.int32).ravel()
		sample_vectors = np.asarray([self.id_to_vector(i, self.starts[i], self.ends[i]) for i in ids])
		origins = np.zeros(len(ids)).astype(np.int32)
		return sample_vectors, ids


	def sample(self, num_samples):
		'''
		function to sample from the search-space
			- num_samples: number of samples to take
		'''
		
		indices = np.arange(0, self.num_regions)
		if num_samples==0:
			return np.zeros((0, self.dimensions)).astype(np.int32)

		# Choose samples from good regions with non-uniform density
		probs = self.per_region_sampling_probs/np.sum(self.per_region_sampling_probs)
		region_ids = np.random.choice(indices, size=num_samples, replace=True, p=probs)
		sample_vectors = np.asarray([self.id_to_vector(i, self.starts[i], self.ends[i]) for i in region_ids])
		
		return sample_vectors, region_ids


	def update_region_w(self, alpha_max=None):
		'''
		function to update region quality (w in Eq. 9) after evaluating a new batch of samples
			- alpha_max: \alpha_max parameter
		'''
		self.update_good_samples(alpha_max)
		self.good_region_ids = self.region_ids[self.good_samples]
		
		for i in range(self.num_regions):
			self.per_region_num_goods[i] = np.sum(self.good_region_ids==i)
			num_samples = np.sum(self.region_ids==i)
			
			if num_samples>0:
				self.per_region_sampling_probs[i] = self.per_region_num_goods[i]/num_samples

	
	def configure_alpha(self, alpha_max=1.0):
		'''
		function to determine \alpha based on current good samples
			- alpha_max: \alpha_max
		'''
		alpha_t = super(Zoom_sampler, self).configure_alpha(alpha_max)
		self.update_region_w(alpha_max)

		return self.alpha_t

	
	def update(self, samples, ids, scores, alpha_max):	
		'''
		function to add newly evaluated samples to the history
			- samples: new samples
			- ids: region ids of new samples
			- scores: evaluation score of new samples
			- alpha_max: current \alpha_max
		'''   
		indices = super(Zoom_sampler, self).update(samples, scores, alpha_max)
		self.region_ids = np.concatenate((self.region_ids, ids), axis=0)[indices]

		self.update_region_w(alpha_max)


	def find_region_to_divide(self):
		'''
			function to find and return the region with maximum score, if it is beig enough to be divided
		'''
		# find region with maximum score
		sorted_args = np.argsort(self.per_region_sampling_probs)[::-1]
		for i in sorted_args:
			# make sure the region is divisible, and then return it			
			dim_to_divide = np.argmax(self.ends[i]-self.starts[i])
			max_diff = self.ends[i][dim_to_divide]-self.starts[i][dim_to_divide]		
			if max_diff>(0.05*(self.boundaries[dim_to_divide,1]-self.boundaries[dim_to_divide,0])):
				return i, dim_to_divide
		
		return None, None
	

	def split_region(self, max_id, ids):
		'''
			function to recalculate sample regions based on the new regions
				- max_id: divided region
				- ids: divided region and the new region id
		'''
		samples_max_id = self.all_samples[self.region_ids==max_id]
		
		ids_to_change = np.where(self.region_ids==max_id)[0]
		self.per_region_num_goods[max_id] = 0
		
		starts = np.take(self.starts, ids, axis=0)
		ends = np.take(self.ends, ids, axis=0)
		
		for i,s in enumerate(samples_max_id):
			id = self.vector_to_id(s, starts, ends)
			assert(id<2)
			
			self.region_ids[ids_to_change[i]] = ids[id]
		
		self.update_region_w()
		

	def divide(self, max_id, max_dim):
		'''
			find the region with maximum ratio of good samples and zoom into it
		'''
		if max_id is not None:	   
			#------------- divide the given region into 2 regions		
			num_regions_old = self.num_regions
			self.num_regions = self.num_regions + 1
			
			self.per_region_sampling_probs = np.append(self.per_region_sampling_probs, np.expand_dims(0,0), axis=0)
			self.per_region_num_goods = np.append(self.per_region_num_goods, np.expand_dims(0,0), axis=0)  
			self.starts = np.append(self.starts, np.zeros((1, self.dimensions)), axis=0)
			self.ends = np.append(self.ends, np.zeros((1, self.dimensions)), axis=0)

			#------- update the starts and ends
			self.starts[max_id] = np.asarray(self.starts[max_id])
			self.ends[self.num_regions-1] = np.asarray(self.ends[max_id])	   
			
			self.ends[max_id, max_dim] = copy.deepcopy((self.starts[max_id][max_dim]+self.ends[max_id, max_dim])/2)	 
			self.starts[self.num_regions-1] = np.asarray(self.starts[max_id])
			self.starts[self.num_regions-1, max_dim] = copy.deepcopy(self.ends[max_id, max_dim])
			
			ids = [max_id, num_regions_old]
			self.split_region(max_id, ids)


	def run_sampling(self, evaluator, num_samples, n_iter, minimize=False, alpha_max=1.0, early_stopping=np.Infinity, 
		save_path='./sampling', n_parallel=1, plot_contour=False, executor=mp.Pool, verbose=True):
		'''
		Function to maximize given black-box function and save results to ./sampling/
			- evaluator : the objective function to be minimized
			- num_samples: number of samples to take at each iteration
			- n_iter: total number of sampling rounds
			- minimize: if set to True, the objective function will be minimized, otherwise maximized
			- alpha_max: \alpha_max parameter
			- early_stopping: the sampling loop will terminate after this many iterations without improvmenet
			- save_path: path to save the sampling history and other artifcats
			- n_parallel: number of parallel evaluations
			- plot_contour: whether to plot contours of objective functions and the samples
			- executor: function to handle parallel evaluations
		returns: optimal hyperparameters
		'''
		coeff = -1 if minimize else 1

		# set up logging directory
		if not os.path.exists(save_path):
			os.mkdir(save_path)

		# set up contour plotting
		contour = None
		if plot_contour:
			if self.dimensions==2:
				path_to_contour = os.path.join(save_path, 'contour')
				os.makedirs(path_to_contour, exist_ok=True)
				contour_file = os.path.join(path_to_contour, 'contour_data.pkl')
				if not os.path.exists(contour_file):
					x = np.linspace(self.boundaries[0,0], self.boundaries[0,1], num=1000)
					y = np.linspace(self.boundaries[1,0], self.boundaries[1,1], num=1000)

					data = np.zeros((len(x), len(y)))
					for i in range(len(x)):
						for j in range(len(y)):
							data[i, j] = evaluator([x[i], y[j]])
					contour = (x, y, data)
					with open(contour_file, 'wb') as f:
						pickle.dump(contour, f)
				else:
					with open(contour_file, 'rb') as f:
						contour = pickle.load(f)
						x, y, data = contour

				max_val = np.max(data)
				max_ind = np.unravel_index(np.argmax(data, axis=None), data.shape)
				print('maximum is %.2f located at (%.2f, %.2f)'%(max_val, x[int(max_ind[0])], y[int(max_ind[1])]))
			else:
				print('=> Contour plotting not possible for %d dimensions.'%self.dimensions)

		# adjusting the per-iteration sampling budget to the parallelism level
		if num_samples % n_parallel != 0:
			num_samples = num_samples - (num_samples % n_parallel) + n_parallel
			print('=> Sampling budget was adjusted to be ' + str(num_samples))
			self.minimum_num_good_samples = num_samples

		# apply the sampling algorithm
		best_samples = []
		best_scores = []
		alpha_vals = []

		num_not_improve = 0
		prev_max_id = None
		iteration = 0
		while iteration < n_iter:
			id_was_the_same = 0
			
			# the inner loop is executed until a region has too many good samples, then the loop breaks and that region is cut in half
			while True:
				if iteration==0:
					starting_iter = True
					samples, ids = self.sample_from_all_regions(num_samples)
					prev_max_score = self.max_score
				else:
					starting_iter = False
					max_score_improv = self.max_score - prev_max_score
					prev_max_score = self.max_score
					samples, ids = self.sample(num_samples)

					# if the percentage improvement in the maximum score is smaller than 0.1%, activate early stopping
					if (max_score_improv/prev_max_score) < 0.001:
						num_not_improve += 1 
					else:
						num_not_improve = 0

				if num_not_improve > early_stopping:
					print('=> Activating early stopping')
					break

				scores = np.zeros(len(samples))
				n_batches = len(samples)//n_parallel if len(samples)%n_parallel==0 else (len(samples)//n_parallel)+1
				with tqdm(total=n_batches) as pbar:
					for i in range(n_batches):
						if n_parallel > 1:
							batch_samples = samples[i*n_parallel:(i+1)*n_parallel]
							with executor() as e:
								scores[i*n_parallel:(i+1)*n_parallel] = list(e.map(evaluator, batch_samples))
						else:
							scores[i] = evaluator(samples[i])
						scores[i*n_parallel:(i+1)*n_parallel] *= coeff
					
						pbar.update(1)
						pbar.set_description('batch %s/%s (samples %s..%s/%s)'%(i+1, num_samples//n_parallel, i*n_parallel, \
														(i+1)*n_parallel, num_samples))					  
				self.update(samples=samples, ids=ids, scores=scores, alpha_max=alpha_max)

				# modify \alpha if necessary, to make sure there are enough "good" samples
				alpha = self.configure_alpha(alpha_max)
				alpha_vals.append(alpha)

				# optionally visualize the current samples on the search-space
				if contour is not None:
					fig, ax = plt.subplots()
					plt.contourf(contour[0], contour[1], contour[-1])
					plt.colorbar()
					plt.scatter(samples[:,0], samples[:,1], c='k', s=30)
					for i in range(len(self.starts)):
						rect = matplotlib.patches.Rectangle(self.starts[i], self.ends[i][0]-self.starts[i][0], self.ends[i][1]-self.starts[i][1],
							linewidth=1, edgecolor='r', facecolor='none')
						ax.add_patch(rect)
					plt.xlim(self.boundaries[0,:])
					plt.ylim(self.boundaries[1,:])
					plt.savefig(os.path.join(path_to_contour, 'score_contour_iter%d.png'%iteration))
					plt.close()
				
				max_id, max_dim = self.find_region_to_divide()		
				selection_scores = np.asarray(self.per_region_sampling_probs)

				# book-keeping
				best_scores.append(np.max(self.all_scores))
				id_best = np.argmax(self.all_scores)
				best_samples.append(self.all_samples[id_best])
			
				if verbose:
					print('=> iter: %d, average score: %.3f, best score: %0.3f' %(iteration, np.mean(scores)*coeff, best_scores[-1]*coeff))

				if max_id==prev_max_id:
					id_was_the_same += 1
				else:
					id_was_the_same = 0
				prev_max_id = max_id
				iteration +=1

				# if we have more than one region but one region is dominating all samples, break it in half
				if len(selection_scores)>1 and np.max(selection_scores/np.sum(selection_scores))>=0.8:
					break
				
				# if no new samples are as good as the best seen before, break the best region in half
				if id_was_the_same>=1:
					break

			# early stopping
			if num_not_improve > early_stopping:
				break

			if max_id is not None:
				# print('Zooming into', self.starts[max_id], self.ends[max_id], 'at dim', max_dim)
				self.divide(max_id, max_dim)

		info = {'best_samples': np.asarray(best_samples),
				'best_scores': np.asarray(best_scores),
				'alpha_vals': alpha_vals,
				'all_samples': self.all_samples,
				'all_scores': self.all_scores,
				'good_samples':self.good_samples}

		path_to_info = os.path.join(save_path, 'history_info.pkl')
		with open(path_to_info, 'wb') as f:
			pickle.dump(info, f)

		id_best_overall = np.argmax(best_scores)
		best_sample_overall = best_samples[id_best_overall]

		if contour is not None:
				plt.figure()
				plt.contourf(contour[0], contour[1], contour[-1])
				plt.colorbar()
				plt.scatter(best_sample_overall[0], best_sample_overall[1], c='r', marker='*', s=30)
				plt.xlim(self.boundaries[0,:])
				plt.ylim(self.boundaries[1,:])
				plt.savefig(os.path.join(path_to_contour, 'score_contour_final.png'))
				plt.close()

		return best_sample_overall, best_scores[id_best_overall]*coeff