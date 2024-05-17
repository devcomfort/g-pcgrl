import gymnasium as gym
import numpy as np
from gymnasium.core import Wrapper

from gym_pcgrl.envs.reps.narrow_graph import triangular
from gym_pcgrl.wrappers import OneHotEncoding


class OneHotGraph(OneHotEncoding):
    def __init__(self, game, name, **kwargs):
        super().__init__(game, name, **kwargs)
        self.dim = kwargs["node_types"]


class GraphWrapper(Wrapper):
    # use with graph-wide representation
    def __init__(self, game, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game, **kwargs)
        else:
            self.env = game
        self.env = OneHotGraph(self.env, 'map', **kwargs)
        self.width = kwargs["width"]
        self.size = triangular(self.width)
        self.num_tiles = kwargs["node_types"]
        gym.Wrapper.__init__(self, self.env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=[self.width, self.width, self.num_tiles],
                                                dtype=np.float32)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs, stats = self.env.reset()
        obs = self.transform(obs)
        return obs, stats

    def transform(self, obs):
        return obs[self.name]  # * 255


class GraphNarrowWrapper(GraphWrapper):
    # use with graph-wide representation
    def __init__(self, game, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game, **kwargs)
        else:
            self.env = game
        self.env = OneHotGraph(self.env, 'map', **kwargs)
        self.width = kwargs["width"]
        self.size = triangular(self.width)
        self.num_tiles = kwargs["node_types"]
        gym.Wrapper.__init__(self, self.env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=[2, self.width, self.num_tiles], dtype=np.float32)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs, stats = self.env.reset()
        obs = self.transform(obs)
        return obs, stats

    def transform(self, obs):
        m = obs["map"].copy()
        y, x = obs["pos"][0], obs["pos"][1]
        row = m[y]
        row[y + 1:] = m[:, y][y + 1:]
        col = m[x]
        col[x + 1:] = m[:, x][x + 1:]
        return np.array([row, col])
