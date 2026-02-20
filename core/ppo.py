import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gymnasium as gym
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, device, state_dim, emb_size, action_dim, action_std):
        self.device = device
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, action_dim),
            nn.Tanh()
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, 1)
        )

        self.action_var = torch.full((action_dim,), action_std * action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)

        distribution = MultivariateNormal(action_mean, cov_mat)
        action = distribution.sample()
        action_logprob = distribution.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        distribution = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = distribution.log_prob(action)
        distribution_entropy = distribution.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), distribution_entropy

    def get_latent(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [B, state_dim]
        returns: [B, emb_size] (output after second Tanh)
        """
        x = self.actor[0](state)
        x = self.actor[1](x)
        x = self.actor[2](x)
        x = self.actor[3](x)
        return x


class PPO:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.device = self.args.device

        # Handle both Dict and Box observation spaces
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            self.state_dim = self.env.observation_space["state"].shape[0]
        else:
            self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.policy = ActorCritic(self.device, self.state_dim, self.args.emb_size, self.action_dim, self.args.action_std).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr, betas=self.args.betas)

        self.policy_old = ActorCritic(self.device, self.state_dim, self.args.emb_size, self.action_dim, self.args.action_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        if isinstance(state, dict):
            state = state["state"]
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.policy_old.act(state, memory).cpu().data.numpy().flatten()
        try:
            low = self.env.action_space.low
            high = self.env.action_space.high
            action = np.clip(action, low, high)
        except Exception:
            pass
        return action

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(np.array(rewards, dtype=np.float32)).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards.float().squeeze()

        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()

        for _ in range(self.args.K_epochs):
            logprobs, state_values, distribution_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) \
                   + self.args.loss_value_c * self.MseLoss(state_values, rewards) \
                   - self.args.loss_entropy_c * distribution_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

