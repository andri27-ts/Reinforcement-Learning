
'''
# Needed only if you run it on Google Colab
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1024, 768))
display.start()
import os
os.environ["DISPLAY"] = ":" + str(display.display) + "." + str(display.screen)'''


from sklearn.preprocessing import StandardScaler
import roboschool

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from tqdm import tqdm
import datetime
import time

import gym
import numpy as np

class NNDynamicModel(nn.Module):
    '''
    Model that predict the next state, given the current state and action
    '''
    def __init__(self, input_dim, obs_output_dim):
        super(NNDynamicModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, obs_output_dim)
        )

    def forward(self, x):
        return self.mlp(x.float())


class NNRewardModel(nn.Module):
    '''
    Model that predict the reward given the current state and action
    '''
    def __init__(self, input_dim, reward_output_dim):
        super(NNRewardModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, reward_output_dim)
        )

    def forward(self, x):
        return self.mlp(x.float())

def gather_random_trajectories(num_traj, env_name):
    '''
    Run num_traj random trajectories to gather information about the next state and reward.
    Data used to train the models in a supervised way.
    '''
    dataset_random = []
    env = gym.make(env_name)

    game_rewards = []
    for n in range(num_traj):

        obs = env.reset()
        while True:
            sampled_action = env.action_space.sample()
            new_obs, reward, done, _ = env.step(sampled_action)

            dataset_random.append([obs, new_obs, reward, done, sampled_action])

            obs = new_obs
            game_rewards.append(reward)

            if done:
                break

    # print some stats
    print('Mean R:',np.round(np.sum(game_rewards)/num_traj,2), 'Max R:', np.round(np.max(game_rewards),2), np.round(len(game_rewards)/num_traj))

    return dataset_random

def model_MSEloss(y_truth, y_pred, device):
    '''
    Compute the MSE (Mean Squared Error)
    '''
    y_truth = torch.FloatTensor(np.array(y_truth)).to(device)
    return F.mse_loss(y_pred.view(-1).float(), y_truth.view(-1))


def train_dyna_model(random_dataset, rl_dataset, env_model, rew_model, batch_size, max_model_iter, num_examples_added, ENV_LEARNING_RATE, REW_LEARNING_RATE, device):
    '''
    Train the two models that predict the next state and the expected reward
    '''

    env_optimizer = optim.Adam(env_model.parameters(), lr=ENV_LEARNING_RATE)
    rew_optimizer = optim.Adam(rew_model.parameters(), lr=REW_LEARNING_RATE)

    if len(rl_dataset) > 0:
        '''
        # To use only a fraction of the random dataset
        rand = np.arange(len(random_dataset))
        np.random.shuffle(rand)
        rand = rand[:int(len(rl_dataset)*0.8)] # 80% of rl dataset

        d_concat = np.concatenate([np.array(random_dataset)[rand], rl_dataset], axis=0)'''

        # Concatenate the random dataset with the RL dataset. Used only in the aggregation iterations
        d_concat = np.concatenate([random_dataset, rl_dataset], axis=0)
    else:
        d_concat = np.array(random_dataset)

    # Split the dataset into train(80%) and test(20%)
    D_train = d_concat[:int(-num_examples_added*1/5)]
    D_valid = d_concat[int(-num_examples_added*1/5):]

    print("len(D):", len(d_concat), 'len(Dtrain)', len(D_train))

    # Shuffle the dataset
    sff = np.arange(len(D_train))
    np.random.shuffle(sff)
    D_train = D_train[sff]


    # Create the input and output for the train
    X_train = np.array([np.concatenate([obs,act]) for obs,_,_,_,act in D_train]) # Takes obs and action
    # Reward's output
    y_rew_train = np.array([[rw] for _,_,rw,_,_ in D_train])
    # Next state output
    y_env_train = np.array([no for _,no,_,_,_ in D_train])
    y_env_train = y_env_train - np.array([obs for obs,_,_,_,_ in D_train]) # y(state) = s(t+1) - s(t)

    # Create the input and output array for the validation
    X_valid = np.array([np.concatenate([obs,act]) for obs,_,_,_,act in D_valid]) # Takes obs and action
    # Reward output
    y_rew_valid = np.array([[rw] for _,_,rw,_,_ in D_valid])
    # Next state output
    y_env_valid = np.array([no for _,no,_,_,_ in D_valid])
    y_env_valid = y_env_valid - np.array([obs for obs,_,_,_,_ in D_valid]) # y(state) = s(t+1) - s(t)

    # Standardize the input features by removing the mean and scaling to unit variance
    input_scaler = StandardScaler()
    X_train = input_scaler.fit_transform(X_train)
    X_valid = input_scaler.transform(X_valid)

    # Standardize the outputs by removing the mean and scaling to unit variance

    env_output_scaler = StandardScaler()
    y_env_train = env_output_scaler.fit_transform(y_env_train)
    y_env_valid = env_output_scaler.transform(y_env_valid)

    rew_output_scaler = StandardScaler()
    y_rew_train = rew_output_scaler.fit_transform(y_rew_train)
    y_rew_valid = rew_output_scaler.transform(y_rew_valid)

    # store all the scalers in a variable to later uses
    norm = (input_scaler, env_output_scaler, rew_output_scaler)

    losses_env = []
    losses_rew = []

    # go through max_model_iter supervised iterations
    for it in tqdm(range(max_model_iter)):
        # create mini batches of size batch_size
        for mb in range(0, len(X_train), batch_size):

            if len(X_train) > mb+BATCH_SIZE:
                X_mb = X_train[mb:mb+BATCH_SIZE]

                y_env_mb = y_env_train[mb:mb+BATCH_SIZE]
                y_rew_mb = y_rew_train[mb:mb+BATCH_SIZE]

                # Add gaussian noise with mean 0 and variance 0.0001 as in the paper
                X_mb += np.random.normal(loc=0, scale=0.001, size=X_mb.shape)

                ## Optimization of the 'env_model' neural net

                env_optimizer.zero_grad()
                # forward pass of the model to compute the output
                pred_state = env_model(torch.tensor(X_mb).to(device))
                # compute the MSE loss
                loss = model_MSEloss(y_env_mb, pred_state, device)

                if it == (max_model_iter - 1):
                    losses_env.append(loss.cpu().detach().numpy())

                # backward pass
                loss.backward()
                # optimization step
                env_optimizer.step()


                ## Optimization of the 'rew_model' neural net
                rew_optimizer.zero_grad()
                # forward pass of the model to compute the output
                pred_rew = rew_model(torch.tensor(X_mb).to(device))
                # compute the MSE loss
                loss = model_MSEloss(y_rew_mb, pred_rew, device)

                if it == (max_model_iter - 1):
                    losses_rew.append(loss.cpu().detach().numpy())
                # backward pass
                loss.backward()
                # optimization step
                rew_optimizer.step()

        # Evalute the models every 10 iterations and print the losses
        if it % 10 == 0:
          env_model.eval()
          rew_model.eval()

          pred_state = env_model(torch.tensor(X_valid).to(device))
          pred_rew = rew_model(torch.tensor(X_valid).to(device))
          env_model.train(True)
          rew_model.train(True)

          valid_env_loss = model_MSEloss(y_env_valid, pred_state, device)
          valid_rew_loss = model_MSEloss(y_rew_valid, pred_rew, device)

          print('..', it, valid_env_loss.cpu().detach().numpy(), valid_rew_loss.cpu().detach().numpy())


    ## Evaluate the MSE losses

    env_model.eval()
    rew_model.eval()

    pred_state = env_model(torch.tensor(X_valid).to(device))
    pred_rew = rew_model(torch.tensor(X_valid).to(device))
    env_model.train(True)
    rew_model.train(True)

    valid_env_loss = model_MSEloss(y_env_valid, pred_state, device)
    valid_rew_loss = model_MSEloss(y_rew_valid, pred_rew, device)

    return np.mean(losses_env), np.mean(losses_rew), valid_env_loss.cpu().detach().numpy(), valid_rew_loss.cpu().detach().numpy(), norm


def multi_model_based_control(env_model, rew_model, real_obs, num_sequences, horizon_length, sample_action, norm, device):
    '''
    Use a random-sampling shooting method, generating random action sequences. The first action with the highest reward of the entire sequence is returned
    '''
    best_reward = -1e9
    best_next_action = []

    input_scaler, env_output_scaler, rew_output_scaler = norm

    m_obs = np.array([real_obs for _ in range(num_sequences)])

    # array that contains the rewards for all the sequence
    unroll_rewards = np.zeros((num_sequences, 1))
    first_sampled_actions = []

    env_model.eval()
    rew_model.eval()

    ## Create a batch of size 'num_sequences' (number of trajectories) to roll the models 'horizon_length' times.
    ## i.e. roll a given number of trajectories in a single batch (to increase speed)

    for t in range(horizon_length):
      # sampled actions for each sequence
      sampled_actions = [sample_action() for _ in range(num_sequences)]
      # scale the input
      models_input = input_scaler.transform(np.concatenate([m_obs, sampled_actions], axis=1))
      # compute the next state for each sequence
      pred_obs = env_model(torch.tensor(models_input).to(device))
      # and the reward
      pred_rew = rew_model(torch.tensor(models_input).to(device))

      # inverse scaler transofrmation
      pred_obs = env_output_scaler.inverse_transform(pred_obs.cpu().detach().numpy())
      # and add previous observation
      m_obs = pred_obs + m_obs

      assert(pred_rew.cpu().detach().numpy().shape == unroll_rewards.shape)

      # sum of the expected rewards
      unroll_rewards += pred_rew.cpu().detach().numpy()

      if t == 0:
        first_sampled_actions = sampled_actions

    env_model.train(True)
    rew_model.train(True)

    # Best the position of the sequence with the higher reward
    arg_best_reward = np.argmax(unroll_rewards)
    best_sum_reward = unroll_rewards[arg_best_reward].squeeze()
    # take the first action of this sequence
    best_action = first_sampled_actions[arg_best_reward]

    return best_action, best_sum_reward


ENV_NAME = 'RoboschoolAnt-v1'

# Main loop hyperp
AGGR_ITER = 3
STEPS_PER_AGGR = 20000

# Random MB hyperp
NUM_RAND_TRAJECTORIES = 1000

# 'cuda' or 'cpu'
device = 'cuda'

# Supervised Model Hyperp
ENV_LEARNING_RATE = 1e-3
REW_LEARNING_RATE = 1e-3
BATCH_SIZE = 512
TRAIN_ITER_MODEL = 55

# Controller Hyperp
HORIZION_LENGTH = 10
NUM_ACTIONS_SEQUENCES = 20000

save_video_test = True

now = datetime.datetime.now()
date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)

if __name__ == '__main__':
    writer_name = 'MB_RL_'+ENV_NAME+'_'+date_time
    print('Name:',writer_name, device)

    # create the environment
    env = gym.make(ENV_NAME)
    if save_video_test:
        env = gym.wrappers.Monitor(env,  "VIDEOS/TEST_VIDEOS_"+writer_name, video_callable=lambda episode_id: True)
    obs = env.reset()

    # gather the dataset of random sequences
    rand_dataset = gather_random_trajectories(NUM_RAND_TRAJECTORIES, ENV_NAME)

    rl_dataset = []

    # Initialize the models
    env_model = NNDynamicModel(env.action_space.shape[0] + env.observation_space.shape[0], env.observation_space.shape[0]).to(device)
    rew_model = NNRewardModel(env.action_space.shape[0] + env.observation_space.shape[0], 1).to(device)


    game_reward = 0
    num_examples_added = len(rand_dataset)

    for n_iter in range(AGGR_ITER):

        # supervised training of the dataset (random and rl if it exists)
        train_env_loss, train_rew_loss, valid_env_loss, valid_rew_loss, norm = train_dyna_model(rand_dataset, rl_dataset, env_model, rew_model, BATCH_SIZE, TRAIN_ITER_MODEL, num_examples_added, ENV_LEARNING_RATE, REW_LEARNING_RATE, device)
        print('{} >> Eloss:{:.4f} EV loss:{:.4f} -- Rloss:{:.4f} RV loss:{:.4f}'.format(n_iter, train_env_loss, valid_env_loss, train_rew_loss, valid_rew_loss))

        obs = env.reset()

        num_examples_added = 0
        game_reward = 0
        game_pred_rews = []
        rews = []

        while num_examples_added < STEPS_PER_AGGR:
            while True:

                tt = time.time()
                # Execute the control to roll the sequences and pick the first action of the sequence with the higher reward
                action, pred_rew = multi_model_based_control(env_model, rew_model, obs, NUM_ACTIONS_SEQUENCES, HORIZION_LENGTH, env.action_space.sample, norm, device)
                game_pred_rews.append(pred_rew)

                # one step in the environment with the action returned by the controller
                new_obs, reward, done, _ = env.step(action)

                input_scaler, env_output_scaler, rew_output_scaler = norm

                ## Compute the reward and print some stats
                models_input = input_scaler.transform([np.concatenate([obs, action])])
                rew_model.eval()
                p_rew = rew_model(torch.tensor(models_input).to(device))
                rew_model.train(True)
                unnorm_rew = rew_output_scaler.inverse_transform([float(p_rew.cpu().data[0])]).squeeze()
                print('  >> ',len(game_pred_rews), 'gt:',np.round(reward,3), 'pred:',np.round(unnorm_rew, 3),
                      'sum:', np.round(pred_rew,3), '|', game_reward, np.round(time.time()-tt, 4), HORIZION_LENGTH)

                # add the last step to the RL dataset
                rl_dataset.append([obs, new_obs, reward, done, action])


                num_examples_added += 1
                obs = new_obs
                game_reward += reward

                # if the environment is done, reset it and print some stats
                if done:
                    obs = env.reset()
                    print('  >> R: {:.2f}, Mean sum:{:.2f}, {}'.format(game_reward, np.mean(game_pred_rews), num_examples_added))

                    rews.append(game_reward)
                    game_reward = 0
                    game_pred_rews = []
                    break

        print('  >> Mean: {:.2f}', np.mean(rews))
