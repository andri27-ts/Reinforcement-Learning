import gym
import numpy as np
from collections import namedtuple
import collections
import torch
import torch.nn as nn
import torch.optim as optim

import time

from neural_net import DQN
from central_control import CentralControl
from buffers import ReplayBuffer


class DQNAgent():
	'''
	Agent class. It control all the agent functionalities
	'''
	rewards = []
	total_reward = 0
	birth_time = 0
	n_iter = 0
	n_games = 0
	ts_frame = 0
	ts = time.time()

	Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward', 'done'], verbose=False, rename=False)

	def __init__(self, env, device, hyperparameters, summary_writer=None):
		'''
		Agent initialization. It create the CentralControl that control all the low
		'''

		# The CentralControl is the 'brain' of the agent
		self.cc = CentralControl(env.observation_space.shape, env.action_space.n, hyperparameters['gamma'], hyperparameters['n_multi_step'], hyperparameters['double_DQN'],
				hyperparameters['noisy_net'], hyperparameters['dueling'], device)

		self.cc.set_optimizer(hyperparameters['learning_rate'])

		self.birth_time = time.time()

		self.iter_update_target = hyperparameters['n_iter_update_target']
		self.buffer_start_size = hyperparameters['buffer_start_size']

		self.epsilon_start = hyperparameters['epsilon_start']
		self.epsilon = hyperparameters['epsilon_start']
		self.epsilon_decay = hyperparameters['epsilon_decay']
		self.epsilon_final = hyperparameters['epsilon_final']

		self.accumulated_loss = []
		self.device = device

		# initialize the replay buffer (i.e. the memory) of the agent
		self.replay_buffer = ReplayBuffer(hyperparameters['buffer_capacity'], hyperparameters['n_multi_step'], hyperparameters['gamma'])
		self.summary_writer = summary_writer

		self.noisy_net = hyperparameters['noisy_net']

		self.env = env

	def act(self, obs):
		'''
		Greedy action outputted by the NN in the CentralControl
		'''
		return self.cc.get_max_action(obs)

	def act_eps_greedy(self, obs):
		'''
		E-greedy action
		'''

		# In case of a noisy net, it takes a greedy action
		if self.noisy_net:
			return self.act(obs)

		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		else:
			return self.act(obs)

	def add_env_feedback(self, obs, action, new_obs, reward, done):
		'''
		Acquire a new feedback from the environment. The feedback is constituted by the new observation, the reward and the done boolean.
		'''

		# Create the new memory and update the buffer
		new_memory = self.Memory(obs=obs, action=action, new_obs=new_obs, reward=reward, done=done)
		self.replay_buffer.append(new_memory)

		# update the variables
		self.n_iter += 1
		# decrease epsilon
		self.epsilon = max(self.epsilon_final, self.epsilon_start - self.n_iter/self.epsilon_decay)
		self.total_reward += reward

	def sample_and_optimize(self, batch_size):
		'''
		Sample batch_size memories from the buffer and optimize them
		'''

		if len(self.replay_buffer) > self.buffer_start_size:
			# sample
			mini_batch = self.replay_buffer.sample(batch_size)
			# optimize
			l_loss = self.cc.optimize(mini_batch)
			self.accumulated_loss.append(l_loss)

		# update target NN
		if self.n_iter % self.iter_update_target == 0:
			self.cc.update_target()

	def reset_stats(self):
		'''
		Reset the agent's statistics
		'''
		self.rewards.append(self.total_reward)
		self.total_reward = 0
		self.accumulated_loss = []
		self.n_games += 1


	def print_info(self):
		'''
		Print information about the agent
		'''
		fps = (self.n_iter-self.ts_frame)/(time.time()-self.ts)
		print('%d %d rew:%d mean_rew:%.2f eps:%.2f, fps:%d, loss:%.4f' % (self.n_iter, self.n_games, self.total_reward, np.mean(self.rewards[-40:]), self.epsilon, fps, np.mean(self.accumulated_loss)))

		self.ts_frame = self.n_iter
		self.ts = time.time()

		if self.summary_writer != None:
			self.summary_writer.add_scalar('reward', self.total_reward, self.n_games)
			self.summary_writer.add_scalar('mean_reward', np.mean(self.rewards[-40:]), self.n_games)
			self.summary_writer.add_scalar('10_mean_reward', np.mean(self.rewards[-10:]), self.n_games)
			self.summary_writer.add_scalar('esilon', self.epsilon, self.n_games)
			self.summary_writer.add_scalar('loss', np.mean(self.accumulated_loss), self.n_games)
