import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import numpy as np
import random
import gym


class ExperienceBuffer():
    def __init__(self, size, state_size):
        self.init_states = torch.zeros([size, state_size])
        self.actions = torch.zeros(size, dtype=torch.long)
        self.rewards = torch.zeros(size)
        self.next_states = torch.zeros([size, state_size])
        self.non_terminal = torch.zeros(size)

        self.size = size
        self.full = False
        self.i = 0

    def store(self, init_state, action, reward, next_state, next_terminal):

        assert self.i < self.size

        self.init_states[self.i, :] = init_state
        self.actions[self.i] = action
        self.rewards[self.i] = reward
        self.next_states[self.i, :] = next_state
        self.non_terminal[self.i] = int(not next_terminal)

        self.i += 1
        if self.i == self.size:
            self.full = True
            self.i = 0

    def sample(self, size):
        indexes = np.arange(self.size) if self.full else np.arange(self.i)
        size = min(size, len(indexes))

        chosen = np.random.choice(indexes, size=size, replace=False)
        return {
            "init_states": self.init_states[chosen, :],
            "actions": self.actions[chosen],
            "rewards": self.rewards[chosen],
            "next_states": self.next_states[chosen, :],
            "non_terminal": self.non_terminal[chosen]
        }


class QNetwork(nn.Module):
    def __init__(self, state_size, num_actions):
        super(QNetwork, self).__init__()
        self.in_dim = state_size
        self.out_dim = num_actions
        self.fc1 = nn.Linear(self.in_dim, 32)
        self.fc2 = nn.Linear(32, self.out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class QNetworkAtari(nn.Module):
    def __init__(self, num_actions):
        super(QNetworkAtari, self).__init__()
        self.out_dim = num_actions

        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernal_size=8,
            stride=4),
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernal_size=4,
            stride=2),
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(9*9*32, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = torch.tensor(x)
        x = x.permute(0, 3, 1, 2)
        x = x.float() / 256

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class DQN():

    def __init__(self, env=None, atari=False, gamma=.99,
                 epoch_steps=10**4, writer=None):

        self.env = gym.make('CartPole-v0') if env is None else env
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]

        self.exp_buf = ExperienceBuffer(2000, self.obs_dim)
        if atari:
            self.qnet = QNetworkAtari(self.num_actions)
        else:
            self. qnet = QNetwork(self.obs_dim, self.num_actions)

        self.qnet_opt = optim.Adam(self.qnet.parameters())

        self.gamma = gamma
        self.epoch_steps = epoch_steps

        self.writer = SummaryWriter(f"runs/dqn/"+str(datetime.now())) \
            if writer is None else writer

        self.epoch = 0

    def choose_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.num_actions))
        else:
            qvals = self.qnet.forward(obs)
            return torch.argmax(qvals).item()

    def qnet_loss(self):
        d = self.exp_buf.sample(10)

        nextqs = self.qnet.forward(d["next_states"])
        maxq, _ = torch.max(nextqs, dim=1)
        ys = d["rewards"] + d["non_terminal"] * self.gamma * maxq

        action_mask = F.one_hot(d["actions"], self.num_actions).float()
        all_qs = self.qnet.forward(d["init_states"])
        qs = torch.sum(action_mask * all_qs, dim=1)

        mse = nn.MSELoss()
        return mse(qs, ys)

    def train(self):
        i = 0
        while i < self.training_steps:
            obs = torch.tensor(self.env.reset(), dtype=torch.float)
            done = False

            while not done:
                init_obs = obs
                act = self.choose_action(obs, 0.05)
                obs, rew, done, _ = self.env.step(act)

                obs = torch.tensor(obs, dtype=torch.float)
                self.exp_buf.store(init_obs, act, rew, obs, done)

                self.qnet_opt.zero_grad()
                loss = self.qnet_loss()
                loss.backward()
                print(f"After {i} steps, loss={loss}")
                self.qnet_opt.step()

                i += 1


if __name__ == '__main__':
    dqn = DQN()
    dqn.train()
