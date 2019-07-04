import torch
import torch.nn as nn
import torch.optim as optim

from statistics import mean

import gym


class VpgBuffer():
    def __init__(self, size, num_actions):
        self.actions = torch.zeros(size)
        self.log_probs = torch.zeros([size, num_actions])
        self.rewards = torch.zeros(size)

        self.episode_rewards = []

        self.size = size
        self.i = 0

    def append(self, log_prob, action, reward):
        assert self.i < self.size, "Buffer Full!"
        self.log_probs[self.i] = log_prob
        self.actions[self.i] = action
        self.episode_rewards.append(reward)

        self.i += 1

        return self.i == self.size

    def end_trajectory(self):
        length = len(self.episode_rewards)
        start = self.i - length
        tot_reward = sum(self.episode_rewards)

        self.rewards[start:self.i] = tot_reward

        self.episode_rewards = []

        return tot_reward

    def get_data(self):
        return self.log_probs, self.actions, self.rewards


class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Mlp, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(self.in_dim, 32)
        self.fc2 = nn.Linear(32, self.out_dim)
        self.tanh = nn.Tanh()
        self.sm = nn.LogSoftmax(dim=0)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.sm(out)
        return out

    def step(self, obs):
        obs = torch.tensor(obs, dtype=torch.float)
        logit = self.forward(obs)
        action = torch.multinomial(torch.exp(logit), 1).item()
        return action, logit


class Vpg():

    def __init__(self, lr=1e-2):
        self.env = gym.make('CartPole-v0')

        assert isinstance(self.env.observation_space, gym.spaces.Box), \
            "A continuous state space is required"
        assert isinstance(self.env.action_space, gym.spaces.Discrete), \
            "A discrete action space is required"

        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]
        self.agent = Mlp(self.obs_dim, self.num_actions)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)

    @staticmethod
    def loss(buffer: VpgBuffer):
        """ Given a VPG Buffer with experience data, will return a "loss"
        fucntion which has the appropriate gradient at the current point.
        """

        log_probs, actions, rewards = buffer.get_data()

        # Create 1-hot mask in shape of actions (num steps, num actions)
        num_actions = log_probs.shape[1]
        action_mask = nn.functional.one_hot(actions.long(), num_actions)

        # Use mask to find probabilities of actions taken
        masked_probs = torch.sum(action_mask.float() * log_probs, dim=1)

        return - torch.mean(rewards * masked_probs)

    def run_epoch(self, batch_size=5000):

        buf = VpgBuffer(batch_size, self.num_actions)

        rews = []
        full = False

        while not full:
            obs = self.env.reset()
            done = False

            while not done and not full:
                act, logit = self.agent.step(obs)

                obs, rew, done, _ = self.env.step(act)

                full = buf.append(logit, act, rew)

            rew = buf.end_trajectory()
            rews.append(rew)

        avg_rew = mean(rews)
        print("Epoch avg reward: \t {}".format(avg_rew))

        self.optimizer.zero_grad()
        loss = self.loss(buf)
        loss.backward()


        self.optimizer.step()


if __name__ == '__main__':
    vpg = Vpg()

    for i in range(50):
        vpg.run_epoch()
