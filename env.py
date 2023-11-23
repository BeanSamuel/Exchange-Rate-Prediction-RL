from copy import deepcopy
from time import time
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pandas as pd
import torch
from torch.distributions import Categorical

from utils.runner import Runner


class Actions(Enum):
    Buy_NTD = 0
    Buy_AUD = 1
    Buy_CAD = 2
    Buy_EUR = 3
    Buy_GBP = 4
    Buy_HKD = 5
    Buy_JPY = 6
    Buy_SGD = 7
    Buy_USD = 8


class Positions(Enum):
    # 代表持有幣別
    NTD = 0
    AUD = 1
    CAD = 2
    EUR = 3
    GBP = 4
    HKD = 5
    JPY = 6
    SGD = 7
    USD = 8

    def opposite(self, action):
        return Positions(action)


class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, df, window_size, render_mode=None):
        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.render_mode = render_mode
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None

        self._last_position = None
        self._action = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))
        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.NTD
        self._position_history = (self.window_size * [None]) + [self._position]
        self._action = 0
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        # print(action)
        self._action = action
        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._truncated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = False

        if action != self._position.value:
            trade = True

        if trade:
            self._last_position = self._position
            self._position = self._position.opposite(action)
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, self._truncated, info

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position
        )

    def _get_observation(self):
        return self.signal_features[self._current_tick - self.window_size:self._current_tick]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def choice_price_col(self, position, buy_or_sell="買入"):
        foreign_price = None
        if position == Positions.AUD:
            foreign_price = self.prices[f'AUD即期{buy_or_sell}'].to_numpy()
        elif position == Positions.CAD:
            foreign_price = self.prices[f'CAD即期{buy_or_sell}'].to_numpy()
        elif position == Positions.EUR:
            foreign_price = self.prices[f'EUR即期{buy_or_sell}'].to_numpy()
        elif position == Positions.GBP:
            foreign_price = self.prices[f'GBP即期{buy_or_sell}'].to_numpy()
        elif position == Positions.HKD:
            foreign_price = self.prices[f'HKD即期{buy_or_sell}'].to_numpy()
        elif position == Positions.JPY:
            foreign_price = self.prices[f'JPY即期{buy_or_sell}'].to_numpy()
        elif position == Positions.SGD:
            foreign_price = self.prices[f'SGD即期{buy_or_sell}'].to_numpy()
        elif position == Positions.USD:
            foreign_price = self.prices[f'USD即期{buy_or_sell}'].to_numpy()
        return foreign_price

    def render(self, mode='human'):

        def _plot_position():
            # 有買賣
            if self._action != self._position.value:

                # 現在不是持有台幣(即有買入外幣)
                if self._position != Positions.NTD:
                    # 買入用紅色
                    buy_price_col = self.choice_price_col(self._position)
                    plt.scatter(self._current_tick, buy_price_col[self._current_tick], color='red')

                # 上一步不是持有台幣(即有賣出外幣)
                if self._last_position != Positions.NTD:
                    # 賣出用綠色
                    sell_price_col = self.choice_price_col(self._last_position)
                    plt.scatter(self._current_tick, sell_price_col[self._current_tick], color='green')

        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices['AUD即期買入'].to_numpy(), label="AUD")
            plt.plot(self.prices['CAD即期買入'].to_numpy(), label="CAD")
            plt.plot(self.prices['EUR即期買入'].to_numpy(), label="EUR")
            plt.plot(self.prices['GBP即期買入'].to_numpy(), label="GBP")
            plt.plot(self.prices['HKD即期買入'].to_numpy(), label="HKD")
            plt.plot(self.prices['JPY即期買入'].to_numpy(), label="JPY")
            plt.plot(self.prices['SGD即期買入'].to_numpy(), label="SGD")
            plt.plot(self.prices['USD即期買入'].to_numpy(), label="USD")
            # plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.0, 1.0))

            # 起始點標藍色
            plt.scatter(self._current_tick, self.prices['AUD即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['CAD即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['EUR即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['GBP即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['HKD即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['JPY即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['SGD即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['USD即期買入'].to_numpy()[self._current_tick], color='blue')

        _plot_position()

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata['render_fps']) - process_time
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."

        plt.pause(pause_time)

    def render_all(self, title=None):

        plt.cla()
        plt.plot(self.prices['AUD即期買入'].to_numpy(), label="AUD")
        plt.plot(self.prices['CAD即期買入'].to_numpy(), label="CAD")
        plt.plot(self.prices['EUR即期買入'].to_numpy(), label="EUR")
        plt.plot(self.prices['GBP即期買入'].to_numpy(), label="GBP")
        plt.plot(self.prices['HKD即期買入'].to_numpy(), label="HKD")
        plt.plot(self.prices['JPY即期買入'].to_numpy(), label="JPY")
        plt.plot(self.prices['SGD即期買入'].to_numpy(), label="SGD")
        plt.plot(self.prices['USD即期買入'].to_numpy(), label="USD")
        plt.legend(bbox_to_anchor=(1.0, 1.0))

        last_positions = Positions.NTD

        for i, position in enumerate(self._position_history):
            if position != None:
                # 有買賣
                if position != last_positions:
                    # 現在不是持有台幣(即有買入外幣)
                    if position != Positions.NTD:
                        price_col = self.choice_price_col(position)
                        plt.scatter(i, price_col[i], color='red')

                    # 上一步不是持有台幣(即有賣出外幣)
                    if last_positions != Positions.NTD:
                        price_col = self.choice_price_col(last_positions)
                        plt.scatter(i, price_col[i], color='green')

                    last_positions = self._position_history[i]

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError


class ForexEnv(TradingEnv):

    def __init__(self, cfg):
        self.config = cfg
        self.cfg = cfg = cfg['task']['env']
        self.train_df = pd.read_csv(cfg['train_data'])
        self.train_df.replace("-", 0, inplace=True)
        self.test_df = pd.read_csv(cfg['test_data'])
        self.test_df.replace("-", 0, inplace=True)

        self.frame_bound = cfg['frame_bound']
        self.num_envs = cfg['num_envs']
        self.window_size = cfg['window_size']
        super().__init__(self.train_df, self.window_size, None)
        self.num_obs = int(np.prod(self.observation_space.shape)) + 9
        self.num_actions = int(np.prod(self.action_space.shape))
        self.num_values = 1

        self.obs = torch.zeros([self.num_envs, self.num_obs], dtype=torch.float)
        self.reset()

    def reset_done(self):
        if self._truncated:
            Runner.logger.log({'total profit': self._total_profit})
            self.obs, _ = self.reset()
            self.compute_obs()
        return self.obs

    def compute_obs(self):
        ct_obs = [0] * 9
        ct_obs[self._position.value] = 1
        self.obs = torch.tensor(self.obs)
        obs = list(self.obs.flatten()) + ct_obs
        self.obs = torch.tensor(obs, dtype=torch.float).reshape(1, self.num_obs)

    def step(self, action):
        self.obs, rew, reset, _ = super().step(action[0].item())
        Runner.logger.log({'reward': rew})
        self.compute_obs()
        rew = torch.tensor(rew, dtype=torch.float).reshape(1, 1)
        reset = torch.tensor(reset, dtype=torch.long).reshape(1, 1)
        return self.obs, rew, reset, {}

    def _update_profit(self, action):

        # 有交易
        if action != self._position.value:
            # 原本非台幣
            if self._position != Positions.NTD:
                # 此處賣出為銀行方，等於投資者的買入
                buy_price_col = self.choice_price_col(self._position, "賣出")
                buy_price = buy_price_col[self._last_trade_tick]

                # 此處買入為銀行方，等於投資者的賣出
                sell_price_col = self.choice_price_col(self._position, "買入")
                sell_price = sell_price_col[self._current_tick]
                self._total_profit = (self._total_profit / buy_price) * sell_price

        # 結束
        if self._truncated:
            if action != Actions.Buy_NTD.value:
                buy_price_col = self.choice_price_col(Positions(action), "賣出")
                buy_price = buy_price_col[self._last_trade_tick]

                sell_price_col = self.choice_price_col(Positions(action), "買入")
                sell_price = sell_price_col[self._current_tick]

                self._total_profit = (self._total_profit / buy_price) * sell_price

    def get_total_profit(self):
        return self._total_profit

    def _calculate_reward(self, action):
        reward = 0
        if self._position == Positions.NTD:
          reward = 0

        else:
          price_col = self.choice_price_col(self._position)
          current_price = price_col[self._current_tick]
          last_day_price = price_col[self._current_tick-1]
          reward = (current_price - last_day_price) / last_day_price

        return reward * 100

        # reward = 0
        #
        # if action != self._position.value:
        #     # 原本非台幣
        #     if self._position != Positions.NTD:
        #         # 此處賣出為銀行方，等於投資者的買入
        #         buy_price_col = self.choice_price_col(self._position, "賣出")
        #         buy_price = buy_price_col[self._last_trade_tick]
        #
        #         # 此處買入為銀行方，等於投資者的賣出
        #         sell_price_col = self.choice_price_col(self._position, "買入")
        #         sell_price = sell_price_col[self._current_tick]
        #         reward = (self._total_profit / buy_price) * sell_price - self._total_profit
        #
        # # 結束
        # elif self._truncated:
        #     if action != Actions.Buy_NTD.value:
        #         buy_price_col = self.choice_price_col(Positions(action), "賣出")
        #         buy_price = buy_price_col[self._last_trade_tick]
        #
        #         sell_price_col = self.choice_price_col(Positions(action), "買入")
        #         sell_price = sell_price_col[self._current_tick]
        #
        #         reward = (self._total_profit / buy_price) * sell_price - self._total_profit
        #
        # return reward * 1000

    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.iloc[start:end, :].filter(like='即期')

        # 這邊可修改想要使用的 feature
        signal_features = self.df.iloc[:, 1:].to_numpy()[start:end]
        return prices, signal_features

    def test(self):
        frame_bounds = [(10, 100), (10, 300), (10, 800)]
        mean_profit = 0

        for frame_bound in frame_bounds:
            cfg = deepcopy(self.config)
            cfg['task']['env']['train_data'] = self.cfg['test_data']
            cfg['task']['env']['frame_bound'] = frame_bound
            env = ForexEnv(cfg)
            env.obs, _ = env.reset()
            env.compute_obs()

            while True:
                action = self.agent.model.get_action(env.obs, test=True)
                obs, reward, done, info = env.step(action)
                if done:
                    mean_profit += env.get_total_profit()
                    break

        mean_profit /= len(frame_bounds)

        Runner.logger.log({'test profit': mean_profit})

        return mean_profit

    def save(self):
        return None

    def load(self, x):
        pass



