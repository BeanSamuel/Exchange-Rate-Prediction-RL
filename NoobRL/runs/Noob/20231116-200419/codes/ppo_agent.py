import os
import shutil
import time
import torch
from torch import optim
from torch.distributions import Categorical

from .utils import neg_log_p
from .dataset import Dataset
from .experience import VecEnvExperienceBuffer
from .actor_critic_model import ActorCriticModel

from utils.runner import Runner
torch.autograd.set_detect_anomaly(True)


class PPOAgent:
    def __init__(self, params, env):
        print(f'\n------------------------------------ {self.__class__.__name__} ------------------------------------')
        self.config = config = params['config']

        self.device = config.get('device', 'cuda:0')

        # save
        self.save_freq = config.get('save_frequency', 0)

        # normalize
        self.normalize_obs = self.config['normalize_obs']
        self.normalize_value = self.config.get('normalize_value', False)
        self.normalize_advantage = config['normalize_advantage']

        # learning
        self.lr = config['learning_rate']
        self.num_actors = env.num_envs
        self.horizon_length = config['horizon_length']
        self.seq_len = self.config.get('seq_length', 4)
        self.max_epochs = self.config.get('max_epochs', -1)
        self.mini_epochs_num = self.config['mini_epochs']
        self.minibatch_size = self.config.get('minibatch_size')
        self.batch_size = self.horizon_length * self.num_actors
        assert (self.batch_size % self.minibatch_size == 0)

        self.e_clip = config['e_clip']
        self.clip_action = self.config.get('clip_actions', True)
        self.clip_value = config['clip_value']
        self.tau = self.config['tau']
        self.gamma = self.config['gamma']
        self.critic_loss_coef = config['critic_loss_coef']
        self.bounds_loss_coef = self.config.get('bounds_loss_coef', None)

        # env
        self.env = env
        self.build_env_info()

        # model
        self.build_model(params['model'])
        self.optimizer = optim.AdamW(self.model.parameters(), self.lr, eps=1e-08, weight_decay=0)

        # buffers
        self.dataset = Dataset(self.batch_size, self.minibatch_size, self.device)
        self.experience_buffer = VecEnvExperienceBuffer([self.horizon_length, self.num_actors], self.env_info, self.device)

        # counter
        self.epoch_num = 0

        self.env.agent = self

    def build_env_info(self):
        self.env_info = dict(
            num_obs=self.env.num_obs,
            num_actions=self.env.num_actions,
            num_values=self.env.num_values,
        )

    def build_model(self, config):
        model = config.get('model', ActorCriticModel)
        config['normalize_obs'] = self.normalize_obs
        config['normalize_value'] = self.normalize_value
        config['normalize_advantage'] = self.normalize_advantage
        config.update(self.env_info)
        self.model = model(config).to(self.device)
        print(self.model)

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    def preproc_action(self, action):
        return action.clone()

    def env_step(self, action):
        _action = self.preproc_action(action)
        obs, reward, done, infos = self.env.step(_action)

        obs = obs.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        for k in infos.keys():
            if isinstance(infos[k], torch.Tensor):
                infos[k] = infos[k].to(self.device)

        return obs, reward, done, infos

    def env_reset_done(self):
        obs = self.env.reset_done()
        return obs.to(self.device)

    def play_steps(self):
        for n in range(self.horizon_length):
            obs = self.env_reset_done()
            self.experience_buffer.update_data('obs', n, obs)

            value = self.model.get_value(obs)
            action, neglogp = self.model.get_action(obs)
            obs, reward, done, infos = self.env_step(action)

            next_value = self.model.get_value(obs)

            self.experience_buffer.update_data('value', n, value)
            self.experience_buffer.update_data('action', n, action)
            self.experience_buffer.update_data('neglogp', n, neglogp)
            self.experience_buffer.update_data('reward', n, reward)
            self.experience_buffer.update_data('next_obs', n, obs)
            self.experience_buffer.update_data('done', n, done)
            self.experience_buffer.update_data('next_value', n, next_value)

            self.post_step(n, infos)

        mb_done = self.experience_buffer.datas['done']
        mb_value = self.experience_buffer.datas['value']
        mb_next_value = self.experience_buffer.datas['next_value']
        mb_reward = self.experience_buffer.datas['reward']

        mb_value, mb_return, mb_adv = self.compute_return(mb_done, mb_value, mb_reward, mb_next_value)

        self.experience_buffer.datas['value'] = mb_value
        self.experience_buffer.datas['return'] = mb_return
        self.experience_buffer.datas['advantage'] = mb_adv
        batch_dict = self.experience_buffer.get_data()

        return batch_dict

    def train_epoch(self):
        self.set_eval()
        play_time_start = time.time()
        batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()

        self.set_train()
        self.curr_frames = self.batch_size
        self.dataset.update(batch_dict)

        for mini_ep in range(0, self.mini_epochs_num):
            for i in range(len(self.dataset)):
                self.update(self.dataset[i])

        self.post_epoch()

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return play_time, update_time, total_time

    def train(self):
        self.last_mean_rewards = -100500
        total_time = 0
        self.frame = 0

        while True:
            self.epoch_num += 1
            play_time, update_time, epoch_time = self.train_epoch()

            total_time += epoch_time
            scaled_time = epoch_time
            scaled_play_time = play_time
            curr_frames = self.curr_frames
            self.frame += curr_frames
            fps_step = curr_frames / scaled_play_time
            fps_total = curr_frames / scaled_time
            print(f'fps step: {fps_step:.1f} fps total: {fps_total:.1f}')

            if self.save_freq > 0:
                if self.epoch_num % self.save_freq == 0:
                    Runner.save_model('Epoch' + str(self.epoch_num))

            if self.epoch_num > self.max_epochs:
                print('MAX EPOCHS NUM!')
                return

    def test(self):
        self.set_eval()
        score = self.env.test()
        print('total profit:', score)

    def post_step(self, n, infos):
        pass

    def post_epoch(self):
        Runner.logger.upload()
        if self.epoch_num % 10 == 0:
            self.env.test()

    def compute_return(self, done, value, reward, next_value):
        last_gae_lam = 0
        adv = torch.zeros_like(reward)
        done = done.float()

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - done[t]
            not_done = not_done.unsqueeze(1)

            delta = reward[t] + self.gamma * next_value[t] - value[t]
            last_gae_lam = delta + self.gamma * self.tau * not_done * last_gae_lam
            adv[t] = last_gae_lam

        returns = self.model.normalize_value(value + adv)
        value = self.model.normalize_value(value)
        adv = self.model.preproc_advantage(adv)
        return value, returns, adv

    def update(self, input_dict):
        obs = input_dict['obs']
        action = input_dict['action']
        old_value = input_dict['value']
        old_neglogp = input_dict['neglogp']
        advantage = input_dict['advantage']
        returns = input_dict['return']

        mu = self.model.get_action(obs, train=True)

        neglogp = -Categorical(mu).log_prob(action.squeeze(-1))
        value = self.model.get_value(obs, train=True)
        # print(mu.shape, action.shape)
        # print(neglogp.shape)
        # print(torch.exp(old_neglogp[0] - neglogp[0]))

        a_loss = self._actor_loss(old_neglogp, neglogp, advantage)
        c_loss = self._critic_loss(old_value, value, returns)
        b_loss = self._bound_loss(mu)
        loss = a_loss + self.critic_loss_coef * c_loss + self.bounds_loss_coef * b_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Runner.logger.log({
                'loss/total': loss,
                'loss/actor': a_loss,
                'loss/critic': c_loss,
                'value/': value,
            })

    def log_results(self, **kwargs):
        pass

    def _actor_loss(self, old_neglogp, neglogp, advantage):
        ratio = torch.exp(old_neglogp - neglogp).clamp_max(2)  # prevent too large loss
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
        a_loss = torch.max(-surr1, -surr2)
        return a_loss.mean()

    def _critic_loss(self, old_value, value, return_batch):
        if self.clip_value:
            value_pred_clipped = old_value + (value - old_value).clamp(-self.e_clip, self.e_clip)
            value_losses = (value - return_batch) ** 2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - value) ** 2

        return c_loss.mean()
    
    def _bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.maximum(mu - soft_bound, torch.tensor(0, device=self.device)) ** 2
            mu_loss_low = torch.minimum(mu + soft_bound, torch.tensor(0, device=self.device)) ** 2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss.mean()

    def save(self):
        return self.model.state_dict()

    def load(self, datas):
        self.model.load_state_dict(datas)
