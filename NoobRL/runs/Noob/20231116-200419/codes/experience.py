import gym
import torch
import numpy as np


class ExperienceBuffer:
    def __init__(self, shape, env_info, device):
        self.shape = tuple(shape)
        self.num_obs = env_info['num_obs']
        self.num_actions = env_info['num_actions']
        self.num_values = env_info['num_values']
        self.device = device
        self.datas = {}
        self.create_buffer()

    def create_buffer(self):
        self.add_buffer('obs', self.num_obs)
        self.add_buffer('reward', self.num_values)
        self.add_buffer('return', self.num_values)
        self.add_buffer('value', self.num_values)
        self.add_buffer('action', self.num_actions)
        self.add_buffer('neglogp')
        self.add_buffer('done', dtype=torch.long)
        self.add_buffer('next_obs', self.num_obs)
        self.add_buffer('next_value', self.num_values)

    # def create_buffer(self):
    #     self.datas['obs'] = torch.zeros([*self.shape, self.num_obs], device=self.device)
    #     self.datas['reward'] = torch.zeros([*self.shape, self.num_values], device=self.device)
    #     self.datas['return'] = torch.zeros([*self.shape, self.num_values], device=self.device)
    #     self.datas['value'] = torch.zeros([*self.shape, self.num_values], device=self.device)
    #     self.datas['action'] = torch.zeros([*self.shape, self.num_actions], device=self.device)
    #     self.datas['neglogp'] = torch.zeros([*self.shape], device=self.device)
    #     self.datas['done'] = torch.zeros([*self.shape], dtype=torch.long, device=self.device)
    #     self.datas['next_obs'] = torch.zeros([*self.shape, self.num_obs], device=self.device)
    #     self.datas['next_value'] = torch.zeros([*self.shape, self.num_values], device=self.device)

    def add_buffer(self, name, shape=(), dtype=torch.float):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.datas[name] = torch.zeros(self.shape + shape, dtype=dtype, device=self.device)

    def update_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_data(self, *args, **kwargs):
        raise NotImplementedError


class VecEnvExperienceBuffer(ExperienceBuffer):
    def update_data(self, key, idx, value):
        self.datas[key][idx] = value

    def get_data(self):
        batch_dict = {}
        for k, v in self.datas.items():
            s = v.shape
            batch_dict[k] = v.transpose(0, 1).reshape(s[0] * s[1], *s[2:])
        return batch_dict


class AsyncExperienceBuffer(ExperienceBuffer):
    def __init__(self, num_actors, env_info, max_size, device):
        super().__init__([max_size * 2], env_info, device)
        self.size = max_size
        self.run_idx = torch.zeros([num_actors], dtype=torch.long, device=self.device)

    def create_buffer(self):
        super().create_buffer()
        self.status = torch.zeros(self.shape, dtype=torch.long, device=self.device)
        self.datas['steps'] = torch.zeros([*self.shape], dtype=torch.long, device=self.device)

    def update_data(self, **kwargs):
        raise NotImplementedError

    def pre_update_data(self, env_ids, datas: dict):
        idx = (self.status == 0).nonzero().squeeze(-1)[:len(env_ids)]
        self.run_idx[env_ids] = idx
        for k, v in datas.items():
            self.datas[k][idx] = v
        self.status[idx] = -1

    def post_update_data(self, env_ids, datas: dict):
        idx = self.run_idx[env_ids]
        for k, v in datas.items():
            self.datas[k][idx] = v
        self.status[self.status > 0] += 1
        self.status[idx] = 1
        # ToDo: check is needed
        self.status[idx[datas['steps'] <= 0]] = 0

    def full(self):
        return torch.sum(self.status > 0) >= self.size

    def get_data(self):
        if not self.full():
            raise
        idx = self.status.topk(self.size, sorted=False)[1]
        data = {k: v[idx] for k, v in self.datas.items()}
        self.status[idx] = 0
        return data


if __name__ == '__main__':
    T = torch.Tensor
    TL = lambda x: T(x).to(dtype=torch.long)
    Z = torch.zeros
    R = torch.rand
    env_info = {'action_space': Z(2), 'observation_space': Z(3), 'value_size': 1}
    buf = AsyncExperienceBuffer(5, env_info, 5, 'cpu')
    buf.pre_update_data(TL([1, 3]), {'obs': T([[1, 1, 1], [2, 2, 2]])})
    buf.pre_update_data(TL([2, 0]), {'obs': T([[3, 3, 3], [4, 4, 4]])})
    buf.post_update_data(TL([2, 0]), {'action': T([[3, 3], [4, 4]])})
    buf.post_update_data(TL([1, 3]), {'action': T([[1, 1], [2, 2]])})
    buf.pre_update_data(TL([2, 0]), {'obs': T([[3, 3, 3], [4, 4, 4]])})
    buf.post_update_data(TL([2, 0]), {'action': T([[3, 3], [4, 4]])})
    print(buf.run_idx)
    print(buf.datas['obs'], buf.datas['action'])
    print(buf.status)
    print(buf.get_data())
    print(buf.status)

