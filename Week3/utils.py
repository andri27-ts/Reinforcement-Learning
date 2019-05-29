import numpy as np
import gym

def test_game(env, agent, test_episodes):
	reward_games = []
	for _ in range(test_episodes):
		obs = env.reset()
		rewards = 0
		while True:
			action = agent.act(obs)
			next_obs, reward, done, _ = env.step(action)
			obs = next_obs
			rewards += reward

			if done:
				reward_games.append(rewards)
				obs = env.reset()
				break

	return np.mean(reward_games)
