from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Base class of policy estimator(function approximator)
class Policy(object):
    def __init__(self, session, optimizer, scope, in_dim):
        self._scope = scope
        # input placeholder for training stage,
        # a batch contains n_episode of paths, where each path may not have the same length
        # 	observations: all observations in a batch, shape=[sum(path_len), observation_dim]
        # 	actions: all actions taken in a batch, shape=[sum(path_len), 1]
        # 	advantages: returns at each time step of all paths in a batch, shape=[sum(path_len)]
        # 	*** inputs are flattened to sum_over_all_paths(path_len) because we have no fixed path length
        self._observations = tf.placeholder(tf.float32, shape=[None, in_dim], name=scope+'/observation')
        self._actions = tf.placeholder(tf.int32, name=scope+"/actions")
        self._advantages = tf.placeholder(tf.float32, name=scope+"/advantages")
        # optimizer to solve function approximator of policy
        self._opt = optimizer
        # tensorflow session used
        self._sess = session
        # undefined policy function approximator
        # 	input: observation, shape=[sum(path_len), observation_dimension]
        # 	output: probability distribution in action space, shape=[sum(path_len), n_action_choices]
        self._policy = None
        # undefined op to take an action
        self._act_op = None
        # undefined loss
        self._loss = None
        # undefined train_op to minimize loss using optimizer
        self._train_op = None 
        # merged summary op for visualization (optional)
        self._merged_summary_op = None

    def act(self, ob):
        if self._act_op == None:
            raise ValueError('Act_op is not defined yet.')

        # only allow one observation as input to take action
        assert ob.shape[0] == 1
        # session run on act_op with input ob, giving out acts_prob (probability of doing actions)
        acts_prob = self._sess.run(self._act_op, feed_dict={self._observations:ob})
        # make a stochastic action according to probability distribution acts_prob
        cs = np.cumsum(acts_prob)
        act_idx = sum(cs < np.random.rand())

        return act_idx

    def train(self, observations, actions, advantages):
        if self._loss == None or self._train_op is None:
            raise ValueError('Loss or Train_op is not defined yet.')

        loss, summary, _ = self._sess.run([self._loss, self._merged_summary_op, self._train_op], \
                            feed_dict={ self._observations: observations, \
                                        self._actions: actions, \
                                        self._advantages: advantages})
        return loss, summary

    def __str__(self):
    	return 'I am an instance of Policy.'
