import torch
import torch.nn as nn
import torch.optim as optim

from statistics import mean

from tensorboardX import SummaryWriter

import gym


class VpgBuffer():
    def __init__(self, size, num_observations, num_actions):
        self.observations = torch.zeros([size, num_observations])
        self.actions = torch.zeros(size)
        # hmm... only store used log-prob?
        self.log_probs = torch.zeros([size, num_actions])

        self.rewards = torch.zeros(size)
        self.traj_rewards = torch.zeros(size)
        self.togo_rewards = torch.zeros(size)

        self.size = size

        self.last_reset = 0
        self.i = 0

    def append(self, logprob, obs, act, rew):
        assert self.i < self.size, "Buffer Full!"
        self.log_probs[self.i] = logprob
        self.observations[self.i, :] = torch.Tensor(obs)
        # print(torch.Tensor(obs))
        self.actions[self.i] = act
        self.rewards[self.i] = rew

        self.i += 1

        return self.i == self.size

    def end_trajectory(self):
        traj_rews = self.rewards[self.last_reset:self.i]

        # Future rewards
        cum_rews = torch.cumsum(traj_rews, dim=0)
        total_rew = cum_rews[-1]

        # offset cum-rewards
        offset_cum_rews = torch.zeros_like(cum_rews)
        offset_cum_rews[1:] = cum_rews[:-1]
        offset_cum_rews[0] = 0

        togo_rews = total_rew - offset_cum_rews
        self.togo_rewards[self.last_reset:self.i] = togo_rews

        # Total rewards
        self.traj_rewards[self.last_reset:self.i] = total_rew

        self.last_reset = self.i
        return total_rew.item()

    def get_data(self):
        return {
            "log_probs": self.log_probs,
            "actions": self.actions,
            "togo_rewards": self.togo_rewards,
            "traj_rewards": self.traj_rewards,
            "observations": self.observations,
        }


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
        return out.squeeze()

    def generate_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float)
        logit = self.sm(self.forward(obs))
        action = torch.multinomial(torch.exp(logit), 1).item()
        return action, logit


class Vpg():

    def __init__(self, lr=1e-2, method="value baseline", writer=None):
        self.env = gym.make('CartPole-v0')

        assert isinstance(self.env.observation_space, gym.spaces.Box), \
            "A continuous state space is required"
        assert isinstance(self.env.action_space, gym.spaces.Discrete), \
            "A discrete action space is required"

        assert method in ["trajectory", "togo", "value baseline"]

        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]

        self.agent = Mlp(self.obs_dim, self.num_actions)
        self.actor_opt = optim.Adam(self.agent.parameters(), lr=lr)

        self.method = method
        if self.method == "value baseline":
            self.value_estimator = Mlp(self.obs_dim, 1)
            self.value_opt = optim.Adam(self.value_estimator.parameters(), lr=lr)

        self.writer = writer if writer is not None else SummaryWriter()

    def actor_loss(self, buffer: VpgBuffer):
        """ Given a VPG Buffer with experience data, will return a "loss"
        fucntion which has the appropriate gradient at the current point.
        """

        d = buffer.get_data()
        weights = d["traj_rewards"]
        if self.method == "togo":
            weights = d["togo_rewards"]
        if self.method == "value baseline":
            value_est = self.value_estimator.forward(d["observations"])
            weights = d["togo_rewards"] - value_est

            self.update_value_estimator(value_est, d["togo_rewards"])

        # Create 1-hot mask in shape of actions (num steps, num actions)
        action_mask = nn.functional.one_hot(d["actions"].long(), self.num_actions)

        # Use mask to find probabilities of actions taken
        masked_probs = torch.sum(action_mask.float() * d["log_probs"], dim=1)

        return - torch.mean(weights * masked_probs)

    def run_epoch(self, epoch_num, batch_size=5000):

        buf = VpgBuffer(batch_size, self.obs_dim, self.num_actions)

        rews = []
        full = False

        while not full:
            obs = self.env.reset()
            done = False

            while not done and not full:
                act, logit = self.agent.generate_action(obs)
                obs, rew, done, _ = self.env.step(act)
                full = buf.append(logprob=logit, obs=obs, act=act, rew=rew)

            rew = buf.end_trajectory()
            rews.append(rew)

        avg_rew = mean(rews)

        self.writer.add_scalar("Average reward", avg_rew, epoch_num)
        if not epoch_num % 5:
            print(f"Epoch {epoch_num} avg reward: \t {avg_rew}")

        self.actor_opt.zero_grad()
        loss = self.actor_loss(buf)
        loss.backward()
        self.actor_opt.step()

    def update_value_estimator(self, estimates, togo_rewards):
        self.value_opt.zero_grad()
        value_loss = nn.MSELoss()
        value_loss(estimates, togo_rewards).backward(retain_graph=True)
        self.value_opt.step()

    def run(self, epocs):
        for i in range(50):
            self.run_epoch(i)


if __name__ == '__main__':

    with SummaryWriter(comment="tj") as wr:
        Vpg(method="trajectory",
            writer=wr).run(50)

    with SummaryWriter(comment="tg") as wr:
        Vpg(method="togo",
            writer=wr).run(50)

    with SummaryWriter(comment="vb") as wr:
        Vpg(method="value baseline",
            writer=wr).run(50)
