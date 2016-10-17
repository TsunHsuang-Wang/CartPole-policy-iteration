from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from policy.policy import *

# policy estimator using multiple layer fully-connected neural network
class MultiLayerNN_Policy(Policy):
	def __init__(self, session, optimizer, scope, NN_config):
		'''
			NN_config:
				in: dimension of observations
				hidden_1: dimension of the 1st hidden layer
				out: size of action space
		'''
		# inherit from class 'Policy'
		Policy.__init__(self, session, optimizer, scope, NN_config['in'])
		# define our NN function approximator for policy, project from observation space to action space
		# Overall scheme: 1.NN architecture, 2.Loss, 3.Train op, 4.Take action op, 5.Summary(optional)
		with tf.variable_scope(self._scope):
			# 1. NN architecture
			with tf.variable_scope('hidden_1'):
				weights = tf.get_variable('weights', shape=[NN_config['in'], NN_config['hidden_1']], 
											initializer=tf.random_normal_initializer())
				biases = tf.get_variable('biases', shape=[NN_config['hidden_1']],
											initializer=tf.random_normal_initializer())
				layer = tf.add(tf.matmul(self._observations, weights), biases)
				layer = tf.tanh(layer)
			with tf.variable_scope('out'):
				weights = tf.get_variable('weights', shape=[NN_config['hidden_1'], NN_config['out']], 
											initializer=tf.random_normal_initializer())
				biases = tf.get_variable('biases', shape=[NN_config['out']],
											initializer=tf.random_normal_initializer())
				self._policy = tf.add(tf.matmul(layer, weights), biases)
				self._policy = tf.nn.softmax(self._policy)

			# 3. Loss
			n_acts = tf.shape(self._policy)[1] # size of action space e.g.for CartPole, n_acts=2 (+1,-1)
			total_timestep = tf.shape(self._policy)[0] # sum_over_all_paths(path_len)
			# action index over our nn_policy, fetching probability of taking certain action in a specific timestep
			# In CartPole case, if now we have 3 episodes(paths) in an iteration, for each episode we have path_len
			# =[3,2,4]. And now we get act_idx with shape 3+2+4=9 and a stochastic policy obtained from our NN_function
			# _approximator with shape=[9,2] (2=n_acts). If we want to fetch prob of taking action 0 in 1st timestep
			# in 2nd episode, we compute idx=3(1st episode len)+1(1st timestep in 2nd episode)+0(take action 0)=4,
			# and finally get prob through idx over flattened stochastic policy with shape=[9*2=18]. The following 
			# simply do the same thing over all 3+2+4=9 timestep, fetching their prob according to certain actions.
			act_idx = tf.mul(tf.range(0, total_timestep), n_acts)
			act_idx = tf.add(act_idx, self._actions) # flattened index
			probs = tf.gather(tf.reshape(self._policy, [-1]), act_idx) # indexing over flattened stochastic policy
			log_probs = tf.log(tf.add(probs, 1e-8)) # adding 1e-8 to make sure we don't take log over 0
			# negative sign due to we get more reward over time. If we are solving problems like approaching the
			# goal as soon as possible, we use a positive sign.
			self._loss = -tf.reduce_sum(tf.mul(log_probs, self._advantages))

			# 3. Train op --> optimize loss
			# grads = self._opt.compute_gradients(self._loss)
			# Op to calculate every variable gradient
			grads = tf.gradients(self._loss, tf.trainable_variables())
			grads = list(zip(grads, tf.trainable_variables()))
			self._train_op = self._opt.apply_gradients(grads, name="train_op")

			# 4. Take action op --> can output prob distribution of an action
			self._act_op = self._policy[0, :]

			# 5. Summaries
			# visualize prob distribution outputed from NN policy approximator
			tf.histogram_summary('out/prob', self._policy)
			# visualize gradient of every trainable variables
			for grad, var in grads:
			    tf.histogram_summary(var.name + '/gradient', grad)
			self._merged_summary_op = tf.merge_all_summaries()

	def __str__(self):
		return 'I am an instance of Policy/MultiLayerNN_Policy.'
