import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_pcgrl.envs.helper import get_int_prob, get_string_map
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS

"""
The PCGRL GYM Environment
"""

def get_action_size(action_space):
    if isinstance(action_space, spaces.MultiDiscrete):
        return np.prod(action_space.nvec)
    else:
        return action_space.n


class PcgrlEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, prob="binary", rep="narrow", **kwargs):
        self._prob = PROBLEMS[prob](**kwargs)
        self._rep = REPRESENTATIONS[rep](**kwargs)
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self._prob._width * self._prob._height), 8)
        self.render_mode = "rgb_array"
        self._max_iterations = 100

        self._heatmap = np.zeros((self._prob._height, self._prob._width))

        self.viewer = None

        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())

        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height,
                                                                 self.get_num_tiles())

    def get_rep(self):
        return self._rep

    def get_prob(self):
        return self._prob

    def seed(self, seed=None):
        seed = self._rep.seed(seed)
        self._prob.seed(seed)
        return [seed]

    def reset(self, options=None, seed=None):
        self._changes = 0
        self._iteration = 0
        self._rep.reset(self._prob._width, self._prob._height,
                        get_int_prob(self._prob._prob, self._prob.get_tile_types()))
        self._rep_stats = self._prob.get_stats(self._rep._map)

        self._prob.reset(self._rep_stats)

        observation = self._rep.get_observation()

        return observation, self.get_rep_stats()

    def get_border_tile(self):
        return self._prob.get_tile_types().index(self._prob._border_tile)

    def get_num_tiles(self):
        return len(self._prob.get_tile_types())

    def adjust_param(self, **kwargs):
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage')))
            self._max_changes = max(int(percentage * self._prob._width * self._prob._height), 1)

        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)
        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height,
                                                                 self.get_num_tiles())

        if "max_iterations" in kwargs:
            self._max_iterations = kwargs["max_iterations"]  # 100#190 # TODO, care that
        if "max_changes" in kwargs:
            self._max_changes = kwargs["max_changes"]  # 8

    def step(self, action):

        self._iteration += 1
        # save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        change, x, y = self._rep.update(action)
        if change > 0:
            self._changes += change
            self._rep_stats = self._prob.get_stats(self._rep._map)

        observation = self._rep.get_observation()
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        done = self._prob.get_episode_over(self._rep_stats,
                                           old_stats) or self._changes >= self._max_changes or self._iteration >= self._max_iterations

        info = self._prob.get_debug_info(self._rep_stats, old_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        # return the values
        return observation, reward, done, False, info

    def render(self, mode='human'):
        if mode == "graph":
            return self._prob.render(self.get_map())

        tile_size = 16
        img = self._prob.render(get_string_map(self._rep._map, self._prob.get_tile_types()))
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        if self.render_mode == 'rgb_array':
            return img
        elif self.render_mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            if not hasattr(img, 'shape'):
                img = np.array(img)
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_map(self):
        return self._rep._map

    def set_map(self, m):
        self._rep._map = m
        self._rep_stats = self._prob.get_stats(self._rep._map)  # without string map

    def get_rep_stats(self):
        return self._rep_stats
