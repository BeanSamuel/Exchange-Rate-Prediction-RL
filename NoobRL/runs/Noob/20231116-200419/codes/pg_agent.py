import torch
from .ppo_agent import PPOAgent
torch.autograd.set_detect_anomaly(True)


class PGAgent(PPOAgent):

    def _actor_loss(self, _, neglogp, reward):
        return (neglogp * reward).sum()

    def _critic_loss(self, old_value, value, return_batch):
        return 0
