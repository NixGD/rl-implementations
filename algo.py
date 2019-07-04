import torch
import torch.nn as nn
import torch.optim as optim

import gym


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
        self.env = gym.make('MountainCar-v0')

        assert isinstance(self.env.observation_space, gym.spaces.Box), \
            "A continuous state space is required"
        assert isinstance(self.env.action_space, gym.spaces.Discrete), \
            "A discrete action space is required"

        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]
        self.agent = Mlp(self.obs_dim, self.num_actions)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)

    def loss(self, logits, acts, rews):
        """
        logits  log likelihoods
                size [n, T, k] where k is number of actions
        acts    actions      [n, T]
        rews    rewards      [n]
        """
        # Create 1-hot mask in shape (n,T,k)
        action_mask = nn.functional.one_hot(acts.long(), self.num_actions)

        # TODO / IMPROVE: use future rewards, or some such.
        # Currently using total trajectory reward, so we reduce on dim 1 too
        logliks = torch.sum(action_mask.float() * logits, dim=[1, 2])

        return - torch.mean(rews * logliks)

    def run_epoch(self, batch_size=100, max_steps=200):
        logits = torch.zeros(batch_size, max_steps, self.num_actions)

        # TODO: make sure this deals properly with too-short episodes.
        acts = torch.zeros((batch_size, max_steps))
        rews = torch.zeros(batch_size)

        for i in range(batch_size):
            obs = self.env.reset()
            done = False

            cum_rew = 0
            for step in range(max_steps):
                act, logit = self.agent.step(obs)
                acts[i, step] = act
                logits[i, step, :] = logit

                obs, rew, done, _ = self.env.step(act)
                cum_rew += rew

                if done:
                    break

            rews[i] = cum_rew

        avg_rew = torch.mean(rews)
        print("Epoch avg reward: \t {}".format(avg_rew))

        self.optimizer.zero_grad()
        loss = self.loss(logits, acts, rews)
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    vpg = Vpg()

    for i in range(50):
        vpg.run_epoch()
