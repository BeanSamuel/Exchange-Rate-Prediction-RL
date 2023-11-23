from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Categorical

from .utils import neg_log_p, eval_no_grad, Identity, RunningMeanStd


class Mlp(nn.Module):
    def __init__(
         self,
         in_size, hidden_size, out_size=None,
         activation: nn.Module = nn.ReLU(),
         output_activation: nn.Module = nn.Identity()
    ):
        super().__init__()
        model = []
        self.sizes = sizes = [in_size] + hidden_size
        for x, y in zip(sizes[:-1], sizes[1:]):
            model.append(nn.Linear(x, y))
            model.append(deepcopy(activation))
        if out_size is not None:
            model.append(nn.Linear(sizes[-1], out_size))
        self.model = nn.Sequential(*model)
        self.out_act = output_activation

    def forward(self, x):
        return self.out_act(self.model(x))

    def set_spectral_norm(self):
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                self.model[i] = nn.utils.spectral_norm(layer)


class ActorCriticModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.obs_size = config['num_obs']
        self.action_size = config['num_actions']
        self.value_size = config['num_values']
        self.actor = self.Actor(self.obs_size, config['actor_mlp'], self.action_size)
        self.critic = self.Critic(self.obs_size, config['critic_mlp'], self.value_size)

        normalize = lambda x: (x - x.mean()) / (x.std() + 1e-8)

        self.normalize_obs = RunningMeanStd(self.obs_size) if config['normalize_obs'] else Identity()
        self.normalize_value = RunningMeanStd(self.value_size) if config['normalize_value'] else Identity()
        self.normalize_advantage = normalize if config['normalize_advantage'] else Identity()
        self.preproc_advantage = lambda x: self.normalize_advantage(x.mean(dim=-1))

    class Actor(nn.Module):
        def __init__(self, obs_size, mlp_size, action_size):
            super().__init__()
            self.mu = Mlp(obs_size, mlp_size, 9, output_activation=nn.Softmax())

        def forward(self, x):
            return self.mu(x)

    class Critic(nn.Module):
        def __init__(self, obs_size, mlp_size, value_size):
            super().__init__()
            self.value = Mlp(obs_size, mlp_size, value_size)

        def forward(self, x):
            return self.value(x)

    @eval_no_grad
    def get_action(self, obs, train=False, test=False):
        obs = self.normalize_obs(obs)
        mu = self.actor(obs)
        if train:
            return mu
        elif test:
            return torch.argmax(mu, dim=-1)
        else:
            action_dist = Categorical(mu)
            action = action_dist.sample()
            return action, -action_dist.log_prob(action)

    @eval_no_grad
    def get_value(self, obs, train=False):
        obs = self.normalize_obs(obs)
        value = self.critic(obs)
        if train:
            return value
        else:
            return self.normalize_value(value, unnorm=True)
