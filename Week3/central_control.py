import numpy as np
from collections import namedtuple
import collections
import torch
import torch.nn as nn
import torch.optim as optim

import time

from neural_net import DQN, DuelingDQN


class CentralControl():

	def __init__(self, observation_space_shape, action_space_shape, gamma, n_multi_step, double_DQN, noisy_net, dueling, device):
		if dueling:
			# Dueling NN
			self.target_nn = DuelingDQN(observation_space_shape, action_space_shape).to(device)
			self.moving_nn = DuelingDQN(observation_space_shape, action_space_shape).to(device)
		else:
			# Normal NN
			self.target_nn = DQN(observation_space_shape, action_space_shape, noisy_net).to(device)
			self.moving_nn = DQN(observation_space_shape, action_space_shape, noisy_net).to(device)

		self.device = device
		self.gamma = gamma
		self.n_multi_step = n_multi_step
		self.double_DQN = double_DQN

	def set_optimizer(self, learning_rate):
		self.optimizer = optim.Adam(self.moving_nn.parameters(), lr=learning_rate)

	def optimize(self, mini_batch):
		'''
		Optimize the NN
		'''
		# reset the grads
		self.optimizer.zero_grad()
		# caluclate the loss of the mini batch
		loss = self._calulate_loss(mini_batch)
		loss_v = loss.item()

		# do backpropagation
		loss.backward()
		# one step of optimization
		self.optimizer.step()

		return loss_v

	def update_target(self):
		'''
		Copy the moving NN in the target NN
		'''
		self.target_nn.load_state_dict(self.moving_nn.state_dict())
		self.target_nn = self.moving_nn

	def get_max_action(self, obs):
		'''
		Forward pass of the NN to obtain the action of the given observations
		'''
		# convert the observation in tensor
		state_t = torch.tensor(np.array([obs])).to(self.device)
		# forawrd pass
		q_values_t = self.moving_nn(state_t)
		# get the maximum value of the output (i.e. the best action to take)
		_, act_t = torch.max(q_values_t, dim=1)
		return int(act_t.item())


	def _calulate_loss(self, mini_batch):
		'''
		Calculate mini batch's MSE loss.
		It support also the double DQN version
		'''

		states, actions, next_states, rewards, dones = mini_batch

		# convert the data in tensors
		states_t = torch.as_tensor(states, device=self.device)
		next_states_t = torch.as_tensor(next_states, device=self.device)
		actions_t = torch.as_tensor(actions, device=self.device)
		rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
		done_t = torch.as_tensor(dones, dtype=torch.uint8, device=self.device)

		# Value of the action taken previously (recorded in actions_v) in the state_t
		state_action_values = self.moving_nn(states_t).gather(1, actions_t[:,None]).squeeze(-1)
		# NB gather is a differentiable function

		# Next state value with Double DQN. (i.e. get the value predicted by the target nn, of the best action predicted by the moving nn)
		if self.double_DQN:
			double_max_action = self.moving_nn(next_states_t).max(1)[1]
			double_max_action = double_max_action.detach()
			target_output = self.target_nn(next_states_t)
			next_state_values = torch.gather(target_output, 1, double_max_action[:,None]).squeeze(-1) # NB: [:,None] add an extra dimension

		# Next state value in the normal configuration
		else:
			next_state_values = self.target_nn(next_states_t).max(1)[0]

		next_state_values = next_state_values.detach() # No backprop

		# Use the Bellman equation
		expected_state_action_values = rewards_t + (self.gamma**self.n_multi_step) * next_state_values
		# compute the loss
		return nn.MSELoss()(state_action_values, expected_state_action_values)
