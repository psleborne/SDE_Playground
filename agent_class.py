import collections

import numpy as np
import random
from IPython.display import clear_output
import collections
from collections import deque
import progressbar
import torch
import torch.nn as nn

import Simulation as sim

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    # def sample(self, batch_size):
    #     indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    #     states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
    #     states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
    #     actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
    #     rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    #     next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
    #     dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
    #     return torch.from_numpy(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.int64), torch.tensor(rewards, dtype=torch.float32), \
    #         torch.tensor(dones, dtype=torch.uint8), torch.tensor(next_states, dtype=torch.float32)

    # def sample(self, batch_size):
    #     indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    #     states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
    #     return np.array(states, dtype=np.uint8), np.array(actions, dtype=np.uint8), np.array(rewards,
    #                                                                                            dtype=np.uint8), \
    #         np.array(dones, dtype=np.uint8), np.array(next_states, dtype=np.uint8)



    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), np.array(rewards,
                                                                                               dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states, dtype=np.float32)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()


        self.pre_net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.adv_net1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.adv_net2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.val_net1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.val_net2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # print("test")
        x = self.pre_net(x)
        val = self.val_net1(x)
        val = self.val_net2(val)
        adv = self.adv_net1(x)
        adv = self.adv_net2(adv)
        return val + adv - adv.mean()


class Agent:
    def __init__(self, env, exp_buffer, eps=1):
        self.buffer = exp_buffer
        self.lr = 0.0001
        self.env = env
        self.eps = eps
        self.gamma = 0.99
        self._reset()

        self.total_reward = 0

    def _reset(self):
        self.state = self.env.reset(0)[0]
        self.total_reward = 0

    @torch.no_grad()
    def play_step(self, net, device=torch.device("cpu")):
        done_reward = None
        trajectory = []

        # if self.env.X < -50:
        #     action = 1
        #
        # elif self.env.X > 50:
        #     action = 0

        if np.random.rand() <= self.eps:
            action = round(random.random())


        else:
            # print(self.state)
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a, dtype=torch.float32).to(device)
            # print(state_v)
            q_vals_v = net(state_v)
            _, act_v = torch.min(q_vals_v, dim=0)
            action = int(act_v.item())
        new_state, reward, is_done = self.env.step(action, True)
        self.total_reward += reward
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.env.total_cost
            trajectory = self.env.trajectory
            # self._reset()
        return done_reward, trajectory

    # Calculate loss and optimize (this is the replay)
    # @torch.no_grad()
    def cal_loss(self, batch, net, tgt_net, device=torch.device("cpu")):

        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states, dtype=torch.float32).to(device)
        next_states_v = torch.tensor(next_states, dtype=torch.float32).to(device)
        actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        # print("cal_loss state_v = ", states_v)

        state_action_values = net(states_v.unsqueeze(-1)).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = tgt_net(next_states_v.unsqueeze(-1)).min(1)[0]
            next_state_values[done_mask] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards_v

        return nn.MSELoss()(state_action_values, expected_state_action_values)
