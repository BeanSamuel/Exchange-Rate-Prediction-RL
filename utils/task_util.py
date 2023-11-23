from env import ForexEnv
from learning.ppo_agent import PPOAgent
from learning.pg_agent import PGAgent


def get_env_agent(config):

    env_map = {
        'Noob': ForexEnv,
    }

    agent_map = {
        'PPOAgent': PPOAgent,
        'PGAgent': PGAgent
    }

    env = env_map[config['task']['name']](config)
    agent = agent_map[config['train']['name']](params=config['train']['params'], env=env)
    return env, agent
