from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
from scipy.signal import lfilter

def print_all_trainable_vars():
	t_vars = tf.trainable_variables()
	for i in xrange(len(t_vars)):
		print(t_vars[i].name)

def print_all_vars():
	all_vars = tf.all_variables()
	for i in xrange(len(all_vars)):
		print(all_vars[i].name)

def print_ops(graph):
	ops = graph.get_operations()
	for i in xrange(len(ops)):
		print(ops[i].name)

def get_space_size(env):
	ob_space = env.observation_space
	act_space = env.action_space
	if isinstance(ob_space, Box): # observation space
		ob_dim = np.prod(ob_space.shape)
	else:
		raise ValueError('Observation space must be Box')
	if isinstance(act_space, Discrete): # action space
		n_acts = act_space.n
	else:
		raise ValueError('Action space must be Discrete')

	return ob_dim, n_acts

def return_of_a_path(x, discount_rate):
	# input x is rewards given at every timestep in an episode(path)
	# compute return of every timestep in an episode
    return lfilter([1],[1,-0.99],x[::-1])[::-1]
