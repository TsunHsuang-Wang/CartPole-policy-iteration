from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 
import utils

class CartPole_Optimizer(object):
	def __init__(self, policy, env, train_config, discount=0.99):
		'''
			train_config:
				max_iters:
				n_episodes:
				path_max_len:
				thresh: 
		'''
		self._policy = policy
		self._env = env
		self._max_iters = train_config['max_iters']
		self._n_episodes = train_config['n_episodes']
		self._path_max_len = train_config['path_max_len']
		self._thresh = train_config['thresh']
		self._summary_log_path = train_config['summary_log_path']
		self._discount = discount

	def sample_path(self):
		# for each timestep we have a triplet
		obs = []
		rewards = []
		acts = []

		ob = self._env.reset() # initialization
		for _ in range(self._path_max_len):
			# take action according to current policy
			act = self._policy.act(ob.reshape(1,-1))
			next_ob, r, done, _ = self._env.step(act) # ob we get here is to decide action to take in the next timestep
			# store infos about current timestep
			obs.append(ob)
			acts.append(act)
			rewards.append(r)

			if done: # path terminates
				break
			# At timestep t, triplet = (ob_t, act_t, r_t),
			# take act_t according to ob_t and get r_t as return
			ob = next_ob

		return {
			'observations': np.array(obs), \
			'actions': np.array(acts), \
			'rewards': np.array(rewards) \
		}

	def process_paths(self, paths):
		for p in paths:
			G = utils.return_of_a_path(p['rewards'], self._discount)

			# data preprocessing skipped now

			# add new attribute to a path ('paths' is passed by reference)
			p['returns'] = G
			p['advantages'] = G

		# flatten paths into a shape=[sum_over_all_paths(timestep_of_a_path),]
		obs_flattened = np.concatenate([p['observations'] for p in paths])
		acts_flattened = np.concatenate([p['actions'] for p in paths])
		adv_flattened = np.concatenate([p['advantages'] for p in paths])

		return {
			'observations': obs_flattened, \
			'actions': acts_flattened, \
			'advantages': adv_flattened \
		}

	def train(self):
		# For each iteration we sample n_episodes paths for stochastic gradient (fetching batch).
		# For each path among n_episodes, we fix our policy of taking actions for path_max_len 
		# timesteps. This is done to stablize our optimization process, comparing to updating our
		# policy for taking actions at every timestep.
		summary_writer = tf.train.SummaryWriter(self._summary_log_path, \
		                                        graph=tf.get_default_graph())
		for i in range(self._max_iters):
			# sample path and do preprocessing
			paths = []
			for _ in range(self._n_episodes):
				paths.append(self.sample_path())
			data = self.process_paths(paths)
			# train one iteration using Monte Carlo policy iteration 
			# (assuming undone paths with max path length terminate)
			loss, summary = self._policy.train(data['observations'], data['actions'], data['advantages'])
			# for every iteration, write summary log to log_path
			summary_writer.add_summary(summary, i)
			# stop criteria
			avg_return = np.mean([np.sum(p['rewards']) for p in paths])
			print('Iteration {}: average return = {}'.format(i, avg_return))
			if avg_return >= self._thresh:
				print('Training process done!!')
				return

		print('Reach maximum iteration number {}, stop training'.format(self._max_iters))
