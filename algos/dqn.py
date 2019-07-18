import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import gym
from baselines.common.atari_wrappers import wrap_deepmind

from tensorboardX import SummaryWriter
from tqdm import tqdm

import statistics
from datetime import datetime
import random
import copy
import argparse
import os


EPOCHS = 50


class ExperienceBuffer():
    def __init__(self, size, state_size, device):
        self.size = size
        self.state_size = state_size
        self.device = device

        self.storage = []
        self.full = False
        self.i = 0

    def store(self, init_state, act, rew, next_state, next_terminal):

        assert self.i < self.size

        data = (init_state, act, rew, next_state, not next_terminal)
        if self.full:
            self.storage[self.i] = data
        else:
            self.storage.append(data)

        self.i += 1
        if self.i == self.size:
            self.full = True
            self.i = 0

    def sample(self, size):

        sampled_data = random.choices(self.storage, k=size)
        s0s, acts, rews, s1s, non_terms = [], [], [], [], []

        for s0, act, rew, s1, non_term in sampled_data:
            s0s.append(np.array(s0, copy=False))
            acts.append(act)
            rews.append(rew)
            s1s.append(np.array(s1, copy=False))
            non_terms.append(non_term)

        return {
            "init_states": np.array(s0s),
            "actions": acts,
            "rewards": rews,
            "next_states": np.array(s1s),
            "non_terminal": non_terms
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

    def __init__(self, env=None, atari=False, lr=1e-4, gamma=.99,
                 epoch_steps=1e4, writer=None, buffer_size=10000,
                 device=None, evaluation_runs=2, batch_size=32,
                 state_sample_size=512, prefill_buffer_size=10000,
                 sync_frequency=1000, save_models=True):

        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("cpu")
        else:
            self.device = device

        # writer never closes!
        self.run_time = str(datetime.now())
        self.writer = SummaryWriter(f"runs/dqn/{self.run_time}") \
            if writer is None else writer

        print(f"Run time: {self.run_time}")

        self.env = gym.make('CartPole-v0') if env is None else env
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]

        self.sync_frequency = sync_frequency
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
        self.target_net = copy.deepcopy(self.qnet).to(self.device)

        # TODO: should test if keeping the buffer on the gpu is faster
        self.exp_buf = ExperienceBuffer(buffer_size, self.obs_dim,
                                        device=torch.device("cpu"))
        self.initialize_buffer(prefill_buffer_size)

        self.qnet_opt = optim.Adam(self.qnet.parameters(), lr=lr)

        self.gamma = gamma
        self.epoch_steps = epoch_steps
        self.epoch = 0
        self.save_models = save_models

        self.state_sample = None
        self.state_sample_size = state_sample_size

    def initialize_buffer(self, steps=10000):
        done = True
        obs = None
        print("\nInitializing buffer:")
        for i in tqdm(range(steps)):
            if done:
                obs = self.env.reset()
            init_obs = obs
            act = random.choice(range(self.num_actions))
            obs, rew, done, _ = self.env.step(act)
            self.exp_buf.store(init_obs, act, rew, obs, done)

    def choose_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.num_actions))
        else:
            t_obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            qvals = self.qnet.forward(t_obs)
            return torch.argmax(qvals).item()

    def qnet_loss(self):
        d = self.exp_buf.sample(self.batch_size)

        for k, x in d.items():
            if type(x) is not torch.Tensor:
                x = torch.tensor(x)
            d[k] = x.to(self.device)

        nextqs = self.target_net.forward(d["next_states"].float())
        maxq, _ = torch.max(nextqs, dim=1)
        ys = d["rewards"].float() + d["non_terminal"].float() * self.gamma * maxq

        all_qs = self.qnet.forward(d["init_states"].float())
        action_mask = F.one_hot(d["actions"].long(), self.num_actions).float()
        qs = torch.sum(action_mask * all_qs, dim=1)

        mse = nn.MSELoss()
        return mse(qs, ys)

    def _evaluation_run(self, epsilon=0.05):
        obs = self.env.reset()
        done = False
        tot_rew = 0

        while not done:
            act = self.choose_action(obs, epsilon)
            obs, rew, done, _ = self.env.step(act)
            tot_rew += rew

        return tot_rew

    def evaluate(self, rewards, epsilon=0.05, render=False):
        print(f"\nEpoch {self.epoch} training statistics:")

        if rewards:
            mean_train_rew = statistics.mean(rewards)
            self.writer.add_scalar("mean train rewards",
                                   mean_train_rew, self.epoch)
            if len(rewards) <= 5:
                print(f"  train rewards: {rewards}")
            print(f"  mean train rewards: {mean_train_rew}")

        print(f"\nEvaluating...")

        if self.evaluation_runs:
            rews = [
                self._evaluation_run() for _ in
                tqdm(range(self.evaluation_runs))
            ]

            print(f"\nEpoch {self.epoch} evaluation statistics:")
            mean_rew, max_rew = statistics.mean(rews), max(rews)

            self.writer.add_scalar("mean eval reward", mean_rew, self.epoch)
            self.writer.add_scalar("max eval reward", max_rew, self.epoch)
            print(f"  eval rewards: {rews}")
            print(f"  mean eval reward: {mean_rew}")
            print(f"  max eval reward:  {max_rew}")

        if self.state_sample is None:
            d = self.exp_buf.sample(self.state_sample_size)
            states = torch.tensor(d["init_states"]).float()
            self.state_sample = states.to(self.device)

        # Calculate the expected reward from the best-action in a constant
        # sample of different states
        sample_qs = self.qnet.forward(self.state_sample)
        sample_max_qs, _ = torch.max(sample_qs, dim=1)
        mean_q = sample_max_qs.mean().item()
        self.writer.add_scalar("mean sample q", mean_q, self.epoch)
        print(f"  mean q of sampled states is {mean_q:.6}")

    def decayed_epsilon(self, step_num):
        decay_steps = 1e5
        end_value = 0.02
        if step_num < decay_steps:
            return 1 - (1-end_value) * step_num / decay_steps
        else:
            return end_value

    def sync_target_net(self):
        self.target_net.load_state_dict(self.qnet.state_dict())

    def train_epoch(self):
        self.epoch += 1
        i = 0

        rewards = []

        loss_to_log = []
        with tqdm(total=self.epoch_steps) as pbar:
            while i < self.epoch_steps:
                obs = self.env.reset()
                done = False
                tot_rew = 0

                while (not done) and (i < self.epoch_steps):
                    step_num = i + (self.epoch-1)*self.epoch_steps

                    init_obs = obs
                    act = self.choose_action(obs, self.decayed_epsilon(step_num))
                    obs, rew, done, _ = self.env.step(act)

                    tot_rew += rew
                    self.exp_buf.store(init_obs, act, rew, obs, done)

                    self.qnet_opt.zero_grad()
                    loss = self.qnet_loss()
                    loss.backward()
                    self.qnet_opt.step()

                    if not step_num % self.sync_frequency:
                        self.sync_target_net()

                    loss_to_log.append(loss.item())
                    if not step_num % 100:
                        mean_loss = statistics.mean(loss_to_log)
                        self.writer.add_scalar("loss", mean_loss, step_num)
                        loss_to_log = []

                    i += 1
                    pbar.update(1)

                if done:
                    rewards.append(tot_rew)
                    self.writer.add_scalar("episode reward", tot_rew, step_num)

        self.evaluate(rewards)

        if self.save_models:
            self.save_model()

    def save_model(self):
        filename = f"models/{self.run_time}/{self.epoch}.pt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            torch.save({
                'epoch': self.epoch,
                'qnet_state_dict': self.qnet.state_dict(),  # ~3MB
                'qnet_opt_state_dict': self.qnet_opt.state_dict(),  # ~5MB
                'state_sample': self.state_sample  # ~15MB
                }, f)

    def close_writer(self):
        self.writer.close()


EPOCHS = 50

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--minimal', action='store_true',
                        help='Lowers parameters to give a quick system test')
    parser.add_argument('--no-skip',  action='store_true',
                        help='Uses PongNoFrameskip-v4 as the enviroment')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to run for')
    args = parser.parse_args()

    envname = 'PongNoFrameskip-v4' if args.no_skip else 'Pong-v0'
    env = gym.make(envname)

    # We don't use episode_life because pong doesn't have life and sometimes
    # we want to reset when the enviroment isn't done.
    wrapped = wrap_deepmind(env, frame_stack=True, episode_life=False)

    dqn_args = {
        "env": wrapped,
        "atari": True
    }

    if args.disable_cuda:
        dqn_args["device"] = torch.device('cpu')

    if args.minimal:
        dqn_args["epoch_steps"] = 50
        dqn_args["evaluation_runs"] = 1
        dqn_args["prefill_buffer_size"] = 100
        dqn_args["save_models"] = False
        dqn_args["writer"] = SummaryWriter(f"runs/tmp/{str(datetime.now())}")
        args.epochs = 1

    dqn = DQN(**dqn_args)
    for i in range(args.epochs):
        print(f"\n\nEpoch {i+1}/{args.epochs}")
        dqn.train_epoch()
    dqn.close_writer()
