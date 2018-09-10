import numpy as np
import collections



class ReplayBuffer():
	'''
	Replay Buffer class to keep the agent memories memorized in a deque structure.
	'''
	def __init__(self, size, n_multi_step, gamma):
		self.buffer = collections.deque(maxlen=size)
		self.n_multi_step = n_multi_step
		self.gamma = gamma

	def __len__(self):
		return len(self.buffer)

	def append(self, memory):
		'''
		append a new 'memory' to the buffer
		'''
		self.buffer.append(memory)

	def sample(self, batch_size):
		'''
		Sample batch_size memories from the buffer.
		NB: It deals the N-step DQN
		'''
		# randomly pick batch_size elements from the buffer
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)

		states = []
		actions = []
		next_states = []
		rewards = []
		dones = []

		# for each indices
		for i in indices:
			sum_reward = 0
			states_look_ahead = self.buffer[i].new_obs
			done_look_ahead = self.buffer[i].done

			# N-step look ahead loop to compute the reward and pick the new 'next_state' (of the n-th state)
			for n in range(self.n_multi_step):
				if len(self.buffer) > i+n:
					# compute the n-th reward
					sum_reward += (self.gamma**n) * self.buffer[i+n].reward
					if self.buffer[i+n].done:
						states_look_ahead = self.buffer[i+n].new_obs
						done_look_ahead = True
						break
					else:
						states_look_ahead = self.buffer[i+n].new_obs
						done_look_ahead = False

			# Populate the arrays with the next_state, reward and dones just computed
			states.append(self.buffer[i].obs)
			actions.append(self.buffer[i].action)
			next_states.append(states_look_ahead)
			rewards.append(sum_reward)
			dones.append(done_look_ahead)

		return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), np.array(next_states, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8))
