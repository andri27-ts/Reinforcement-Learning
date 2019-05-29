import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter, init
from torch.nn import functional as F
import math

class NoisyLinear(nn.Linear):
	'''
	Noisy Linear layer -> NOISY NETWORKS FOR EXPLORATION https://arxiv.org/pdf/1706.10295.pdf

	NB: IT DOESN T WORKS. PROBLEMS WITH THE EPSILON PARAMETERES INITIALIZATION
	'''


	def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
		super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
		self.sigma_init = sigma_init

		self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
		self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
		if bias:
			self.sigma_bias = Parameter(torch.Tensor(out_features))
			self.register_buffer('epsilon_bias', torch.zeros(out_features))
		self.reset_parameters()

	def reset_parameters(self):
		'''
		Initialize the biases and weights
		'''
		if hasattr(self, 'sigma_bias'):
			init.constant_(self.sigma_bias, self.sigma_init)
			init.constant_(self.sigma_weight, self.sigma_init)

		std = math.sqrt(3/self.in_features)
		init.uniform_(self.weight, -std, std)
		init.uniform_(self.bias, -std, std)

	def forward(self, input):
		if self.bias is not None:
			## NB: in place operation. PyTorch is not happy with that!! CHANGE IT
			self.epsilon_bias.data.normal_()

			# new bias with noise
			bias = self.bias + self.sigma_bias*self.epsilon_bias
		else:
			bias = self.bias

		## NB: in place operation. PyTorch is not happy with that!! CHANGE IT
		self.epsilon_weight.data.normal_()
		# new weight with noise
		weight = self.weight + self.sigma_weight*self.epsilon_weight
		# create the linear layer it the added noise
		return F.linear(input, weight, bias)


class DuelingDQN(nn.Module):
	'''
	Dueling DQN -> http://proceedings.mlr.press/v48/wangf16.pdf
	'''

	def __init__(self, input_shape, n_actions):
		super(DuelingDQN, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU())

		conv_out_size = self._get_conv_out(input_shape)
		# Predict the actions advantage
		self.fc_a = nn.Sequential(
			nn.Linear(conv_out_size, 512),
			nn.ReLU(),
			nn.Linear(512, n_actions))

		# Predict the state value
		self.fc_v = nn.Sequential(
			nn.Linear(conv_out_size, 512),
			nn.ReLU(),
			nn.Linear(512, 1))

	def _get_conv_out(self, shape):
		o = self.conv(torch.zeros(1, *shape)) # apply convolution layers..
		return int(np.prod(o.size())) # ..to obtain the output shape

	def forward(self, x):
		batch_size = x.size()[0]
		conv_out = self.conv(x).view(batch_size, -1) # apply convolution layers and flatten the results

		adv = self.fc_a(conv_out)
		val = self.fc_v(conv_out)

		# Sum the state value with the advantage of each action (NB: the mean has been subtracted from the advantage. It is used in the paper)
		return val + adv - torch.mean(adv, dim=1, keepdim=True)


class DQN(nn.Module):
	'''
	Deep Q newtork following the architecture used in the DeepMind paper (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
	'''

	def __init__(self, input_shape, n_actions, noisy_net):
		super(DQN, self).__init__()

		# 3 convolutional layers. Take an image as input (NB: the BatchNorm layers aren't in the paper but they increase the convergence)
		self.conv = nn.Sequential(
			nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU())

		# Compute the output shape of the conv layers
		conv_out_size = self._get_conv_out(input_shape)

		# 2 fully connected layers
		if noisy_net:
			# In case of NoisyNet use noisy linear layers
			self.fc = nn.Sequential(
				NoisyLinear(conv_out_size, 512),
				nn.ReLU(),
				NoisyLinear(512, n_actions))
		else:
			self.fc = nn.Sequential(
				nn.Linear(conv_out_size, 512),
				nn.ReLU(),
				nn.Linear(512, n_actions))

	def _get_conv_out(self, shape):
		# Compute the output shape of the conv layers
		o = self.conv(torch.zeros(1, *shape)) # apply convolution layers..
		return int(np.prod(o.size())) # ..to obtain the output shape

	def forward(self, x):
		batch_size = x.size()[0]
		conv_out = self.conv(x).view(batch_size, -1) # apply convolution layers and flatten the results
		return self.fc(conv_out) # apply fc layers
