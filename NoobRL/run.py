import cProfile

from utils.hydra_cfg.hydra_utils import *
from utils.runner import Runner

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def run(cfg):
    Runner.init(cfg)
    if cfg.profile:
        cProfile.runctx("Runner.run()", globals(), locals(), "profile.pstat")
    else:
        Runner.run()


def runs(cfg):
    #
    # # policy gradient
    # cfg.train.name = 'PGAgent'
    #
    # # reward
    # cfg.train.params.config.tau = 0
    # cfg.train.params.config.gamma = 0
    # run(cfg)
    #
    # cfg.train.params.config.tau = 0.75
    # cfg.train.params.config.gamma = 0.75
    # run(cfg)
    #
    # # mlp size
    # cfg.train.params.model.actor_mlp = [32, 32]
    # cfg.train.params.model.critic_mlp = [32, 32]
    # cfg.train.params.config.learning_rate = 1e-3
    # cfg.train.params.config.minibatch_size = 512
    # run(cfg)
    #
    # # batch size
    # cfg.train.params.model.actor_mlp = [256, 256]
    # cfg.train.params.model.critic_mlp = [256, 256]
    # cfg.train.params.config.learning_rate = 1e-3
    # cfg.train.params.config.minibatch_size = 64
    # run(cfg)
    #
    # # lr
    # cfg.train.params.model.actor_mlp = [256, 256]
    # cfg.train.params.model.critic_mlp = [256, 256]
    # cfg.train.params.config.learning_rate = 1e-2
    # cfg.train.params.config.minibatch_size = 512
    # run(cfg)

    # ppo
    cfg.train.name = 'PPOAgent'
    cfg.train.params.model.actor_mlp = [256, 256]
    cfg.train.params.model.critic_mlp = [256, 256]
    cfg.train.params.config.learning_rate = 1e-3
    cfg.train.params.config.minibatch_size = 512
    run(cfg)


@hydra.main(config_name="config", config_path="./cfg")
def parse_hydra_configs(cfg: DictConfig):
    if cfg.debug:
        cfg.wandb = cfg.debug == "wandb"
        cfg.save = cfg.debug == "save"
        cfg.task.env.num_envs = 1

        runs(cfg)

    elif cfg.test:
        cfg.wandb = False
        cfg.save = False
        cfg.task.env.num_envs = 1
        cfg.train.params.config.minibatch_size = 1
        runs(cfg)

    else:
        runs(cfg)

    Runner.close()


if __name__ == "__main__":
    parse_hydra_configs()
