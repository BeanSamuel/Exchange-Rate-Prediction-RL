import os
import time

import torch
import shutil
import random
import numpy as np
from datetime import datetime

from utils.hydra_cfg.hydra_utils import *
from utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from utils.wandb_logger import WandbLogger


class StopException(Exception):
    pass


class _Runner:
    def __init__(self):
        pass

    def init(self, cfg):
        self.cfg_dict = omegaconf_to_dict(cfg)

        self.test = cfg.test
        self.checkpoint = cfg.checkpoint
        self.clear_cmd()
        self.task_name = cfg.task.name
        self.start_time = datetime.now().strftime('%Y%m%d-%H%M%S')

        # create save dir
        self.save = cfg.save
        self.run_name = self.start_time
        self.task_dir = os.path.join('./runs', self.task_name)
        if self.save:
            self.run_dir = os.path.join(self.task_dir, self.run_name)
            os.makedirs(self.run_dir, exist_ok=True)

        # set seed
        cfg.seed = 42
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        # logger
        self.logger = WandbLogger(self.task_name, self.start_time, cfg.wandb)

        # backup code
        if self.save:
            code_path = './learning'
            if code_path is not None:
                shutil.copytree(code_path, os.path.join(self.run_dir, 'codes'))

        # dump config dict
        if self.save:
            with open(os.path.join(self.run_dir, 'config.yaml'), 'w') as f:
                f.write(OmegaConf.to_yaml(cfg))

        # get env & agent
        from utils.task_util import get_env_agent
        self.env, self.agent = get_env_agent(self.cfg_dict)

        if self.test:
            if self.checkpoint == '':
                self.checkpoint = max(os.listdir(self.task_dir))

        # load checkpoint
        if self.checkpoint:
            self.load_model(self.checkpoint)

        if cfg.render:
            self.write_cmd('render')

    def run(self):
        try:
            if self.test:
                self.agent.test()
            else:
                self.agent.train()
            self.stop()
        except StopException:
            pass

    def stop(self):
        self.save_model('FinalEpoch')
        self.logger.stop()
        raise StopException

    def read_cmd(self):
        try:
            with open('./controller', 'r') as f:
                return f.read().rstrip()
        except:
            return ''

    def write_cmd(self, cmd):
        try:
            with open('./controller', 'w') as f:
                return f.write(cmd)
        except:
            pass

    def clear_cmd(self):
        open('./controller', 'w').close()

    def close(self):
        pass

    def control(self):
        cmd = self.read_cmd()
        if cmd == 'save':
            self.clear_cmd()
            self.save_model(f'Epoch{self.agent.epoch_num}')
        elif cmd == 'stop':
            self.stop()
        elif cmd == 'record':
            self.clear_cmd()
            self.env.record(f'Epoch{self.agent.epoch_num}')
        elif cmd == 'close':
            self.stop()
            self.close()

        self.env.render = cmd == 'render'

    def get_save_dir(self, sub_dir, epoch_dir=False):
        if epoch_dir:
            save_dir = os.path.join(self.run_dir, sub_dir, f'Epoch{self.agent.epoch_num}')
        else:
            save_dir = os.path.join(self.run_dir, sub_dir)

        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def save_model(self, name):
        if self.save:
            path = os.path.join(self.get_save_dir('model'), name)
            torch.save({'agent': self.agent.save(), 'env': self.env.save()}, path)
            print(f'Save model to {path}')

    def load_model(self, name, epoch=None):
        epoch = 'FinalEpoch' if epoch is None else f'Epoch{epoch}'
        model_dir = os.path.join(self.task_dir, name, 'model', epoch)
        datas = torch.load(model_dir)
        self.agent.load(datas['agent'])
        self.env.load(datas['env'])


Runner = _Runner()
