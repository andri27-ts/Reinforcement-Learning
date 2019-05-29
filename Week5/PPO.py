import numpy as np
import gym
from tensorboardX import SummaryWriter

import datetime
from collections import namedtuple
from collections import deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

class A2C_policy(nn.Module):
    '''
    Policy neural network
    '''
    def __init__(self, input_shape, n_actions):
        super(A2C_policy, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU())

        self.mean_l = nn.Linear(32, n_actions[0])
        self.mean_l.weight.data.mul_(0.1)

        self.var_l = nn.Linear(32, n_actions[0])
        self.var_l.weight.data.mul_(0.1)

        self.logstd = nn.Parameter(torch.zeros(n_actions[0]))

    def forward(self, x):
        ot_n = self.lp(x.float())
        return F.tanh(self.mean_l(ot_n))

class A2C_value(nn.Module):
    '''
    Actor neural network
    '''
    def __init__(self, input_shape):
        super(A2C_value, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1))


    def forward(self, x):
        return self.lp(x.float())


class Env:
    '''
    Environment class
    '''
    game_rew = 0
    last_game_rew = 0
    game_n = 0
    last_games_rews = [-200]
    n_iter = 0

    def __init__(self, env_name, n_steps, gamma, gae_lambda, save_video=False):
        super(Env, self).__init__()

        # create the new environment
        self.env = gym.make(env_name)
        self.obs = self.env.reset()

        self.n_steps = n_steps
        self.action_n = self.env.action_space.shape
        self.observation_n = self.env.observation_space.shape[0]
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    # CHANGED
    def steps(self, agent_policy, agent_value):
        '''
        Execute the agent n_steps in the environment
        '''
        memories = []
        for s in range(self.n_steps):
            self.n_iter += 1

            # get the agent policy
            ag_mean = agent_policy(torch.tensor(self.obs))

            # get an action following the policy distribution
            logstd = agent_policy.logstd.data.cpu().numpy()
            action = ag_mean.data.cpu().numpy() + np.exp(logstd) * np.random.normal(size=logstd.shape)
            #action = np.random.normal(loc=ag_mean.data.cpu().numpy(), scale=torch.sqrt(ag_var).data.cpu().numpy())
            action = np.clip(action, -1, 1)

            state_value = float(agent_value(torch.tensor(self.obs)))

            # Perform a step in the environment
            new_obs, reward, done, _ = self.env.step(action)

            # Update the memories with the last interaction
            if done:
                # change the reward to 0 in case the episode is end
                memories.append(Memory(obs=self.obs, action=action, new_obs=new_obs, reward=0, done=done, value=state_value, adv=0))
            else:
                memories.append(Memory(obs=self.obs, action=action, new_obs=new_obs, reward=reward, done=done, value=state_value, adv=0))


            self.game_rew += reward
            self.obs = new_obs

            if done:
                print('#####',self.game_n, 'rew:', int(self.game_rew), int(np.mean(self.last_games_rews[-100:])), np.round(reward,2), self.n_iter)

                # reset the environment
                self.obs = self.env.reset()
                self.last_game_rew = self.game_rew
                self.game_rew = 0
                self.game_n += 1
                self.n_iter = 0
                self.last_games_rews.append(self.last_game_rew)

        # compute the discount reward of the memories and return it
        return self.generalized_advantage_estimation(memories)

    def generalized_advantage_estimation(self, memories):
        '''
        Calculate the advantage diuscounted reward as in the paper
        '''
        upd_memories = []
        run_add = 0

        for t in reversed(range(len(memories)-1)):
            if memories[t].done:
                run_add = memories[t].reward
            else:
                sigma = memories[t].reward + self.gamma * memories[t+1].value - memories[t].value
                run_add = sigma + run_add * self.gamma * self.gae_lambda

            ## NB: the last memoy is missing
            # Update the memories with the discounted reward
            upd_memories.append(Memory(obs=memories[t].obs, action=memories[t].action, new_obs=memories[t].new_obs, reward=run_add + memories[t].value, done=memories[t].done, value=memories[t].value, adv=run_add))

        return upd_memories[::-1]


def log_policy_prob(mean, std, actions):
    # policy log probability
    act_log_softmax = -((mean-actions)**2)/(2*torch.exp(std).clamp(min=1e-4)) - torch.log(torch.sqrt(2*math.pi*torch.exp(std)))
    return act_log_softmax

def compute_log_policy_prob(memories, nn_policy, device):
    '''
    Run the policy on the observation in the memory and compute the policy log probability
    '''
    n_mean = nn_policy(torch.tensor(np.array([m.obs for m in memories], dtype=np.float32)).to(device))
    n_mean = n_mean.type(torch.DoubleTensor)
    logstd = agent_policy.logstd.type(torch.DoubleTensor)

    actions = torch.DoubleTensor(np.array([m.action for m in memories])).to(device)

    return log_policy_prob(n_mean, logstd, actions)

def clipped_PPO_loss(memories, nn_policy, nn_value, old_log_policy, adv, epsilon, writer, device):
    '''
    Clipped PPO loss as in the paperself.
    It return the clipped policy loss and the value loss
    '''

    # state value
    rewards = torch.tensor(np.array([m.reward for m in memories], dtype=np.float32)).to(device)
    value = nn_value(torch.tensor(np.array([m.obs for m in memories], dtype=np.float32)).to(device))
    # Value loss
    vl_loss = F.mse_loss(value.squeeze(-1), rewards)

    new_log_policy = compute_log_policy_prob(memories, nn_policy, device)
    rt_theta = torch.exp(new_log_policy - old_log_policy.detach())

    adv = adv.unsqueeze(-1) # add a dimension because rt_theta has shape: [batch_size, n_actions]
    pg_loss = -torch.mean(torch.min(rt_theta*adv, torch.clamp(rt_theta, 1-epsilon, 1+epsilon)*adv))

    return pg_loss, vl_loss

def test_game(tst_env, agent_policy, test_episodes):
    '''
    Execute test episodes on the test environment
    '''

    reward_games = []
    steps_games = []
    for _ in range(test_episodes):
        obs = tst_env.reset()
        rewards = 0
        steps = 0
        while True:
            ag_mean = agent_policy(torch.tensor(obs))
            action = np.clip(ag_mean.data.cpu().numpy().squeeze(), -1, 1)

            next_obs, reward, done, _ = tst_env.step(action)
            steps += 1
            obs = next_obs
            rewards += reward

            if done:
                reward_games.append(rewards)
                steps_games.append(steps)
                obs = tst_env.reset()
                break

    return np.mean(reward_games), np.mean(steps_games)


Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward', 'done', 'value', 'adv'], verbose=False, rename=False)

# Hyperparameters
ENV_NAME = 'BipedalWalker-v2'
#ENV_NAME = 'BipedalWalkerHardcore-v2'

MAX_ITER = 500000

BATCH_SIZE = 64
PPO_EPOCHS = 7
device = 'cpu'
CLIP_GRADIENT = 0.2
CLIP_EPS = 0.2

TRAJECTORY_SIZE = 2049
GAE_LAMBDA = 0.95
GAMMA = 0.99

## Test Hyperparameters
test_episodes = 5
best_test_result = -1e5
save_video_test = True
N_ITER_TEST = 100

POLICY_LR = 0.0004
VALUE_LR = 0.001
now = datetime.datetime.now()
date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)

load_model = False
checkpoint_name = "checkpoints/..."

if __name__ == '__main__':
    # Create the environment
    env = Env(ENV_NAME, TRAJECTORY_SIZE, GAMMA, GAE_LAMBDA)

    writer_name = 'PPO_'+ENV_NAME+'_'+date_time+'_'+str(POLICY_LR)+'_'+str(VALUE_LR)+'_'+str(TRAJECTORY_SIZE)+'_'+str(BATCH_SIZE)
    writer = SummaryWriter(log_dir='content/runs/'+writer_name)

    # create the test environment
    test_env = gym.make(ENV_NAME)
    if save_video_test:
        test_env = gym.wrappers.Monitor(test_env,  "VIDEOS/TEST_VIDEOS_"+writer_name, video_callable=lambda episode_id: episode_id%10==0)

    # initialize the actor-critic NN
    agent_policy = A2C_policy(test_env.observation_space.shape, test_env.action_space.shape).to(device)
    agent_value = A2C_value(test_env.observation_space.shape).to(device)

    # initialize policy and value optimizer
    optimizer_policy = optim.Adam(agent_policy.parameters(), lr=POLICY_LR)
    optimizer_value = optim.Adam(agent_value.parameters(), lr=VALUE_LR)

    # Do you want to load a trained model?
    if load_model:
        print('> Loading checkpoint {}'.format(checkpoint_name))
        checkpoint = torch.load(checkpoint_name)
        agent_policy.load_state_dict(checkpoint['agent_policy'])
        agent_value.load_state_dict(checkpoint['agent_value'])
        optimizer_policy.load_state_dict(checkpoint['optimizer_policy'])
        optimizer_value.load_state_dict(checkpoint['optimizer_value'])


    experience = []
    n_iter = 0

    while n_iter < MAX_ITER:
        n_iter += 1

        batch = env.steps(agent_policy, agent_value)

        # Compute the policy probability with the old policy network
        old_log_policy = compute_log_policy_prob(batch, agent_policy, device)

        # Gather the advantage from the memory..
        batch_adv = np.array([m.adv for m in batch])
        # .. and normalize it to stabilize network
        batch_adv = (batch_adv - np.mean(batch_adv)) / (np.std(batch_adv) + 1e-7)
        batch_adv = torch.tensor(batch_adv).to(device)

        # variables to accumulate losses
        pol_loss_acc = []
        val_loss_acc = []

        # execute PPO_EPOCHS epochs
        for s in range(PPO_EPOCHS):
            # compute the loss and optimize over mini batches of size BATCH_SIZE
            for mb in range(0, len(batch), BATCH_SIZE):
                mini_batch = batch[mb:mb+BATCH_SIZE]
                minib_old_log_policy = old_log_policy[mb:mb+BATCH_SIZE]
                minib_adv = batch_adv[mb:mb+BATCH_SIZE]

                # Compute the PPO clipped loss and the value loss
                pol_loss, val_loss = clipped_PPO_loss(mini_batch, agent_policy, agent_value, minib_old_log_policy, minib_adv, CLIP_EPS, writer, device)

                # optimize the policy network
                optimizer_policy.zero_grad()
                pol_loss.backward()
                optimizer_policy.step()

                # optimize the value network
                optimizer_value.zero_grad()
                val_loss.backward()
                optimizer_value.step()

                pol_loss_acc.append(float(pol_loss))
                val_loss_acc.append(float(val_loss))

        # add scalars to the tensorboard
        writer.add_scalar('pg_loss', np.mean(pol_loss_acc), n_iter)
        writer.add_scalar('vl_loss', np.mean(val_loss_acc), n_iter)
        writer.add_scalar('rew', env.last_game_rew, n_iter)
        writer.add_scalar('10rew', np.mean(env.last_games_rews[-100:]), n_iter)

        # Test the agent
        if n_iter % N_ITER_TEST == 0:
            test_rews, test_stps = test_game(test_env, agent_policy, test_episodes)
            print(' > Testing..', n_iter,test_rews, test_stps)
            # if it achieve the best results so far, save the models
            if test_rews > best_test_result:
                torch.save({
                    'agent_policy': agent_policy.state_dict(),
                    'agent_value': agent_value.state_dict(),
                    'optimizer_policy': optimizer_policy.state_dict(),
                    'optimizer_value': optimizer_value.state_dict(),
                    'test_reward': test_rews
                }, 'checkpoints/checkpoint_'+writer_name+'.pth.tar')
                best_test_result = test_rews
                print('=> Best test!! Reward:{:.2f}  Steps:{}'.format(test_rews, test_stps))

            writer.add_scalar('test_rew', test_rews, n_iter)


    writer.close()
