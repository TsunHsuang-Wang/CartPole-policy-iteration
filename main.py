from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 
import gym

from utils import *
from policy.multilayerNN_policy import MultiLayerNN_Policy
from cartpole_optimizer import CartPole_Optimizer

sess = tf.Session()
opt = tf.train.AdamOptimizer(learning_rate=0.01)
scope = 'CartPole_Policy'
env = gym.make('CartPole-v0')
ob_dim, n_acts = get_space_size(env)
NN_config = {
	'in': ob_dim, \
	'hidden_1': 8, \
	'out': n_acts \
}
p = MultiLayerNN_Policy(session=sess, optimizer=opt, scope=scope, NN_config=NN_config)

sess.run(tf.initialize_all_variables())

train_config = {
	'max_iters': 200, \
	'n_episodes': 100, \
	'path_max_len': 200, \
	'thresh': 100, \
	'summary_log_path': '/tmp/tensorflow_logs'
}
opt = CartPole_Optimizer(p, env, train_config)
opt.train()