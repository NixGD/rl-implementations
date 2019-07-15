import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

import gym
from baselines.common.atari_wrappers import wrap_deepmind

from tensorboardX import SummaryWriter

from tqdm import tqdm


import statistics
from datetime import datetime

import argparse
import os


EPOCHS = 10


class ExperienceBuffer():
    def __init__(self, size, state_size, device):
        self.init_states = torch.zeros([size] + state_size).to(device)
        self.actions = torch.zeros(size, dtype=torch.long).to(device)
        self.rewards = torch.zeros(size).to(device)
        self.next_states = torch.zeros([size] + state_size).to(device)
        self.non_terminal = torch.zeros(size).to(device)

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
            kernel_size=8,
            stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(9*9*32, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        x = x.reshape(-1, 84, 84, 4).permute(0, 3, 1, 2)
        x = x.float() / 256

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DQN():

    def __init__(self, env=None, atari=False, gamma=.99,
                 epoch_steps=2e3, writer=None, buffer_size=2000,
                 device=None, evaluation_runs=5, batch_size = 512,
                 state_sample_size = 1000):

        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("cpu")
        else:
            self.device = device

        self.env = gym.make('CartPole-v0') if env is None else env
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]

        self.batch_size = batch_size
        self.evaluation_runs = evaluation_runs
        self.atari = atari
        if atari:
            print("using atari")
            self.qnet = QNetworkAtari(self.num_actions)
            self.obs_dim = [84, 84, 4]

        else:
            self.qnet = QNetwork(self.obs_dim, self.num_actions)

        self.qnet.to(self.device)

        # TODO: should test if keeping the buffer on the gpu is faster
        self.exp_buf = ExperienceBuffer(buffer_size, self.obs_dim,
            device=torch.device("cpu"))

        self.qnet_opt = optim.Adam(self.qnet.parameters())

        self.gamma = gamma
        self.epoch_steps = epoch_steps

        self.run_time = str(datetime.now())
        self.writer = SummaryWriter(f"runs/dqn/{self.run_time}") \
            if writer is None else writer

        self.epoch = 0

        self.state_sample = None
        self.state_sample_size = state_sample_size


    def choose_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.num_actions))
        else:
            qvals = self.qnet.forward(obs.to(self.device))
            return torch.argmax(qvals).item()

    def qnet_loss(self):
        d = self.exp_buf.sample(self.batch_size)

        for k, x in d.items():
            if type(x) is not torch.Tensor:
                x = torch.tensor(x)
            d[k] = x.to(self.device)

        nextqs = self.qnet.forward(d["next_states"])
        maxq, _ = torch.max(nextqs, dim=1)
        ys = d["rewards"] + d["non_terminal"] * self.gamma * maxq

        all_qs = self.qnet.forward(d["init_states"])
        action_mask = F.one_hot(d["actions"], self.num_actions).float()
        qs = torch.sum(action_mask * all_qs, dim=1)

        mse = nn.MSELoss()
        return mse(qs, ys)

    def _evaluation_run(self):
        obs = self.env.reset()
        done = False
        tot_rew = 0

        while not done:
            obs = torch.tensor(obs, dtype=torch.float)
            act = self.choose_action(obs, 0.05)
            obs, rew, done, _ = self.env.step(act)
            tot_rew += rew

        return tot_rew


    def evaluate(self, epsilon=0.05, render=False):
        if self.evaluation_runs:
            rews = [
                self._evaluation_run() for _ in
                tqdm(range(self.evaluation_runs))
            ]

            print("")
            mean_rew, max_rew = statistics.mean(rews), max(rews)

            self.writer.add_scalar("mean eval reward", mean_rew, self.epoch)
            self.writer.add_scalar("max eval reward", max_rew, self.epoch)
            print(f"Epoch {self.epoch}:")
            print(f"  rewards: {rews}")
            print(f"  mean reward: {mean_rew}")
            print(f"  max reward:  {max_rew}")

        if self.state_sample is None:
            d = self.exp_buf.sample(self.state_sample_size)
            states = d["init_states"]
            self.state_sample = states.to(self.device)

        # Calculate the expected reward from the best-action in a constant
        # sample of different states
        sample_qs = self.qnet.forward(self.state_sample)
        sample_max_qs, _ = torch.max(sample_qs, dim=1)
        mean_q = sample_max_qs.mean().item()
        self.writer.add_scalar("mean sample q", mean_q, self.epoch)
        print(f"mean q of sampled states is {mean_q:.6}")

    def train_epoch(self):
        self.epoch += 1
        i = 0

        with tqdm(total=self.epoch_steps) as pbar:
            while i < self.epoch_steps:
                obs = torch.tensor(self.env.reset(), dtype=torch.float)
                for _ in range(6):
                    _ = self.env.step(0)
                done = False

                while (not done) and (i < self.epoch_steps):
                    init_obs = obs
                    act = self.choose_action(obs, 0.05)
                    obs, rew, done, _ = self.env.step(act)
                    assert not rew

                    obs = torch.tensor(obs, dtype=torch.float)
                    self.exp_buf.store(init_obs, act, rew, obs, done)

                    self.qnet_opt.zero_grad()
                    loss = self.qnet_loss()
                    loss.backward()
                    self.qnet_opt.step()

                    step_num = i + (self.epoch-1)*self.epoch_steps
                    self.writer.add_scalar("loss", loss, step_num)

                    i += 1

                    pbar.update(1)

        self.evaluate()

        filename = f"models/{self.run_time}/{self.epoch}.pt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            torch.save({
                'epoch': self.epoch,
                'qnet_state_dict': self.qnet.state_dict(),
                'qnet_opt_state_dict': self.qnet_opt.state_dict(),
                'state_sample': self.state_sample
                }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
    parser.add_argument('--minimal', action='store_true',
                    help='Lowers parameters to give a quick system test')
    args = parser.parse_args()

    env = gym.make('Pong-v0')
    wrapped = wrap_deepmind(env, frame_stack=True)

    dqn_args = {
        "env" : wrapped,
        "atari" : True
    }

    if args.disable_cuda:
        dqn_args["device"] = torch.device('cpu')

    if args.minimal:
        dqn_args["epoch_steps"] = 50
        dqn_args["evaluation_runs"] = 1
        EPOCHS = 1

    dqn = DQN(**dqn_args)
    for i in range(EPOCHS):
        dqn.train_epoch()
